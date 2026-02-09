use axum::{
    Router,
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Request, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use std::{net::SocketAddr, process::Stdio, sync::Arc, time::Duration};
use tokio::{
    process::{Child, Command},
    sync::Mutex,
    time::{Instant, sleep},
};

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    upstream_base: String,
    child: Arc<Mutex<Option<Child>>>,
}

fn env_u16(name: &str, default: u16) -> u16 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_isize(name: &str, default: isize) -> isize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<isize>().ok())
        .unwrap_or(default)
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

async fn healthz() -> &'static str {
    "ok"
}

async fn shutdown(State(state): State<AppState>) -> impl IntoResponse {
    // Best-effort kill of the managed llama-server child process.
    // Use a single concrete response type: (StatusCode, String)
    let mut lock = state.child.lock().await;

    if let Some(mut child) = lock.take() {
        let resp: (StatusCode, String) = match child.kill().await {
            Ok(()) => (StatusCode::OK, "llama-server terminated".to_string()),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("kill failed: {e}"),
            ),
        };

        // Reap the process to avoid a zombie (best-effort).
        let _ = child.wait().await;

        resp
    } else {
        (StatusCode::OK, "no child process".to_string())
    }
}

async fn proxy_v1_chat_completions(State(state): State<AppState>, req: Request<Body>) -> Response {
    proxy_request(state, req, "/v1/chat/completions").await
}

async fn proxy_v1_completions(State(state): State<AppState>, req: Request<Body>) -> Response {
    proxy_request(state, req, "/v1/completions").await
}

async fn proxy_request(state: AppState, req: Request<Body>, path: &str) -> Response {
    let uri: Uri = match format!("{}{}", state.upstream_base, path).parse() {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, "bad upstream uri").into_response(),
    };

    let (parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, "failed to read request body").into_response(),
    };

    let mut rb = state.client.request(parts.method, uri.to_string());

    // forward headers (skip Host and connection-specific headers)
    rb = rb.headers(filter_headers(parts.headers));

    let resp = match rb.body(bytes).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("upstream request failed: {e}"),
            )
                .into_response();
        }
    };

    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut out_headers = HeaderMap::new();
    for (k, v) in resp.headers().iter() {
        out_headers.insert(k, v.clone());
    }

    let out_bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                format!("upstream body read failed: {e}"),
            )
                .into_response();
        }
    };

    (status, out_headers, out_bytes).into_response()
}

fn filter_headers(headers: HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (k, v) in headers.iter() {
        if k == axum::http::header::HOST
            || k == axum::http::header::CONNECTION
            || k == axum::http::header::CONTENT_LENGTH
            || k == axum::http::header::TRANSFER_ENCODING
        {
            continue;
        }
        out.insert(k, v.clone());
    }

    // Ensure content-type if caller forgot it (llama-server expects JSON)
    if !out.contains_key(axum::http::header::CONTENT_TYPE) {
        out.insert(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
    }

    out
}

async fn wait_for_upstream(
    client: &reqwest::Client,
    base: &str,
    timeout_s: u64,
    poll_ms: u64,
    child_slot: &Arc<Mutex<Option<Child>>>,
) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(timeout_s);
    let url = format!("{base}/health");

    let mut last_progress_log = Instant::now();
    let progress_every = Duration::from_secs(10);

    loop {
        // Fail fast if the child process has exited.
        {
            let mut lock = child_slot.lock().await;
            if let Some(child) = lock.as_mut() {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        return Err(format!(
                            "llama-server exited before becoming ready: {status}"
                        ));
                    }
                    Ok(None) => {}
                    Err(e) => {
                        return Err(format!("failed to check llama-server status: {e}"));
                    }
                }
            }
        }

        if Instant::now() > deadline {
            return Err(format!(
                "timeout waiting for llama-server readiness after {timeout_s}s (still returning non-200 from {url})"
            ));
        }

        match client.get(&url).send().await {
            Ok(resp) => {
                let status = resp.status();

                // Ready: any 2xx is good enough for our proxy.
                if status.is_success() {
                    return Ok(());
                }

                // Common during model load:
                // llama.cpp documents /health returning 503 while loading. :contentReference[oaicite:1]{index=1}
                if status.as_u16() == 503 {
                    // Consume the body (best-effort) so keep-alive connections behave well.
                    // We don't require JSON parsing here to avoid adding dependencies.
                    let _ = resp.text().await; // reqwest Response::text() :contentReference[oaicite:2]{index=2}
                } else {
                    // Any other non-success status is unexpected: include body to help diagnose.
                    let body = resp.text().await.unwrap_or_default();
                    return Err(format!(
                        "unexpected /health response from llama-server: HTTP {status} body={body}"
                    ));
                }
            }
            Err(_) => {
                // Connection errors are expected early in startup; keep waiting.
            }
        }

        if last_progress_log.elapsed() >= progress_every {
            eprintln!(
                "waiting for llama-server readiness (polling {url}); this can take minutes for large models"
            );
            last_progress_log = Instant::now();
        }

        sleep(Duration::from_millis(poll_ms.max(50))).await;
    }
}

async fn spawn_llama_server(
    child_slot: Arc<Mutex<Option<Child>>>,
    llama_server_path: String,
    model_path: String,
    llama_host: String,
    llama_port: u16,
    ctx: usize,
    n_gpu_layers: isize,
) -> Result<(), String> {
    let mut cmd = Command::new(&llama_server_path);

    // llama-server args
    cmd.arg("-m")
        .arg(&model_path)
        .arg("--host")
        .arg(&llama_host)
        .arg("--port")
        .arg(llama_port.to_string())
        .arg("-c")
        .arg(ctx.to_string());

    // -1 means "don’t pass -ngl"
    if n_gpu_layers >= 0 {
        cmd.arg("-ngl").arg(n_gpu_layers.to_string());
    }

    cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());

    let child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn llama-server: {e}"))?;

    let mut lock = child_slot.lock().await;
    *lock = Some(child);
    Ok(())
}

#[tokio::main]
async fn main() {
    // Config
    let bind_host = env_string("BIND_HOST", "0.0.0.0");
    let bind_port = env_u16("BIND_PORT", 3000);

    let llama_server_path = env_string("LLAMA_SERVER_PATH", "/usr/local/bin/llama-server");
    let model_path = env_string("MODEL_PATH", "/models/granite-4.0-h-tiny-Q8_0.gguf");
    let llama_host = env_string("LLAMA_HOST", "127.0.0.1");
    let llama_port = env_u16("LLAMA_PORT", 8080);

    let ctx = env_usize("CTX", 8192);

    // default 99, set N_GPU_LAYERS=-1 to omit -ngl
    let n_gpu_layers: isize = env_isize("N_GPU_LAYERS", 99);

    // Readiness tuning (important for large GGUF loads on CPU)
    let ready_timeout_s = env_u64("LLAMA_READY_TIMEOUT_SECS", 1800); // 30 minutes
    let ready_poll_ms = env_u64("LLAMA_READY_POLL_MS", 250);

    let upstream_base = format!("http://{llama_host}:{llama_port}");

    // Client for proxying requests (can be long-running)
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .expect("reqwest client");

    // Separate client for readiness checks (keep it snappy)
    let health_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("reqwest health client");

    let child_slot: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));

    // Start llama-server
    if let Err(e) = spawn_llama_server(
        child_slot.clone(),
        llama_server_path,
        model_path,
        llama_host.clone(),
        llama_port,
        ctx,
        n_gpu_layers,
    )
    .await
    {
        eprintln!("{e}");
        std::process::exit(1);
    }

    // Wait until llama-server is ready
    if let Err(e) = wait_for_upstream(
        &health_client,
        &upstream_base,
        ready_timeout_s,
        ready_poll_ms,
        &child_slot,
    )
    .await
    {
        eprintln!("{e}");
        std::process::exit(1);
    }

    let state = AppState {
        client,
        upstream_base,
        child: child_slot.clone(),
    };

    let app = Router::new()
        .route("/healthz", get(healthz))
        // Management: stop llama-server
        .route("/shutdown", post(shutdown))
        // OpenAI-ish endpoints (proxy)
        .route("/v1/chat/completions", post(proxy_v1_chat_completions))
        .route("/v1/completions", post(proxy_v1_completions))
        .with_state(state);

    let addr: SocketAddr = format!("{bind_host}:{bind_port}")
        .parse()
        .expect("bind address");

    axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app)
        .await
        .unwrap();
}

// Unit tests in a separate file.
#[cfg(test)]
mod test_api;
