use axum::{
    Router,
    body::{Body, Bytes},
    http::{HeaderMap, Request, StatusCode},
    routing::{get, post},
};
use std::{sync::Arc, time::Duration};
use tokio::sync::Mutex;
use tower::util::ServiceExt; // for `oneshot`
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_string_contains, header, method, path},
};

use super::*;

/// Build the same router as `main()` (but test-local, no socket bind).
fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/shutdown", post(shutdown))
        .route("/v1/chat/completions", post(proxy_v1_chat_completions))
        .route("/v1/completions", post(proxy_v1_completions))
        .with_state(state)
}

fn test_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client for tests")
}

fn state_with_upstream(upstream_base: String) -> AppState {
    AppState {
        client: test_client(),
        upstream_base,
        child: Arc::new(Mutex::new(None)),
    }
}

async fn read_body(resp: axum::response::Response) -> (StatusCode, HeaderMap, Bytes) {
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read response body");
    (status, headers, body)
}

#[tokio::test]
async fn healthz_ok() {
    let state = state_with_upstream("http://127.0.0.1:1".to_string());
    let app = build_app(state).into_service::<Body>();

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let (status, _headers, body) = read_body(resp).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(std::str::from_utf8(&body).unwrap(), "ok");
}

#[tokio::test]
async fn shutdown_when_no_child() {
    let state = state_with_upstream("http://127.0.0.1:1".to_string());
    let app = build_app(state).into_service::<Body>();

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/shutdown")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let (status, _headers, body) = read_body(resp).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(std::str::from_utf8(&body).unwrap(), "no child process");
}

#[tokio::test]
async fn shutdown_kills_and_reaps_child() {
    // Arrange: create a real child process we can kill.
    let child = tokio::process::Command::new("sleep")
        .arg("1000")
        .spawn()
        .expect("spawn sleep");

    let state = state_with_upstream("http://127.0.0.1:1".to_string());

    // Put child into the managed slot.
    {
        let mut lock = state.child.lock().await;
        *lock = Some(child);
    }

    // Act: call shutdown.
    let resp = build_app(state.clone())
        .into_service::<Body>()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/shutdown")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let (status, _headers, body) = read_body(resp).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        std::str::from_utf8(&body).unwrap(),
        "llama-server terminated"
    );

    // Assert: slot cleared.
    let lock = state.child.lock().await;
    assert!(lock.is_none());
    drop(lock);

    // Assert: second shutdown is idempotent.
    let resp2 = build_app(state)
        .into_service::<Body>()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/shutdown")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let (status2, _headers2, body2) = read_body(resp2).await;
    assert_eq!(status2, StatusCode::OK);
    assert_eq!(std::str::from_utf8(&body2).unwrap(), "no child process");
}

#[tokio::test]
async fn filter_headers_strips_connection_specific_and_sets_content_type() {
    let mut in_h = HeaderMap::new();
    in_h.insert(axum::http::header::HOST, "example.com".parse().unwrap());
    in_h.insert(
        axum::http::header::CONNECTION,
        "keep-alive".parse().unwrap(),
    );
    in_h.insert(axum::http::header::CONTENT_LENGTH, "123".parse().unwrap());
    in_h.insert(
        axum::http::header::TRANSFER_ENCODING,
        "chunked".parse().unwrap(),
    );
    in_h.insert("x-forward-me", "yes".parse().unwrap());

    let out = filter_headers(in_h);

    assert!(!out.contains_key(axum::http::header::HOST));
    assert!(!out.contains_key(axum::http::header::CONNECTION));
    assert!(!out.contains_key(axum::http::header::CONTENT_LENGTH));
    assert!(!out.contains_key(axum::http::header::TRANSFER_ENCODING));

    // preserved
    assert_eq!(out.get("x-forward-me").unwrap(), "yes");

    // ensured
    assert_eq!(
        out.get(axum::http::header::CONTENT_TYPE).unwrap(),
        "application/json"
    );
}

#[tokio::test]
async fn proxy_chat_completions_passthrough_status_headers_body() {
    // Upstream mock server.
    let mock = MockServer::start().await;

    let upstream_body = r#"{"id":"cmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}"#;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        // prove we forward caller headers that are not stripped
        .and(header("x-test", "1"))
        // prove the JSON body is forwarded
        .and(body_string_contains(r#""model":"granite""#))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .insert_header("x-upstream", "yes")
                .set_body_raw(upstream_body, "application/json"),
        )
        .mount(&mock)
        .await;

    let state = state_with_upstream(mock.uri());
    let app = build_app(state).into_service::<Body>();

    let req_body = r#"{"model":"granite","messages":[{"role":"user","content":"hi"}]}"#;

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("x-test", "1")
                .header("content-type", "application/json")
                .body(Body::from(req_body))
                .unwrap(),
        )
        .await
        .unwrap();

    let (status, headers, body) = read_body(resp).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers.get("x-upstream").unwrap(), "yes");
    assert!(
        headers
            .get(axum::http::header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("application/json")
    );
    assert_eq!(std::str::from_utf8(&body).unwrap(), upstream_body);
}

#[tokio::test]
async fn proxy_completions_passes_through_upstream_error_status() {
    let mock = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(500)
                .insert_header("content-type", "application/json")
                .set_body_raw(r#"{"error":"upstream"}"#, "application/json"),
        )
        .mount(&mock)
        .await;

    let state = state_with_upstream(mock.uri());
    let app = build_app(state).into_service::<Body>();

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"granite","prompt":"x"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let (status, _headers, body) = read_body(resp).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        std::str::from_utf8(&body).unwrap(),
        r#"{"error":"upstream"}"#
    );
}

#[tokio::test]
async fn proxy_returns_bad_gateway_when_upstream_unreachable() {
    // Port 0 is valid in a URL but not connectable; reqwest should fail quickly.
    let state = state_with_upstream("http://127.0.0.1:0".to_string());
    let app = build_app(state).into_service::<Body>();

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"granite","messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    let (status, _headers, body) = read_body(resp).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(
        std::str::from_utf8(&body)
            .unwrap()
            .starts_with("upstream request failed:"),
        "unexpected body: {}",
        std::str::from_utf8(&body).unwrap()
    );
}
