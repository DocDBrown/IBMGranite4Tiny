# syntax=docker/dockerfile:1.6
ARG RUST_VERSION=1.88.0
ARG APP_NAME=granite-4-tiny
ARG MODEL_FILE=granite-4.0-h-tiny-Q8_0.gguf

############################
# Builder (Rust app)
############################
FROM rust:${RUST_VERSION}-bookworm AS builder
ARG APP_NAME
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./

RUN mkdir -p src && printf '%s\n' 'fn main() {}' > src/main.rs
ENV CARGO_TARGET_DIR=/tmp/target-deps
RUN cargo build --release --locked

COPY src ./src
ENV CARGO_TARGET_DIR=/tmp/target-app
RUN cargo build --release --locked

RUN set -eux; \
    mkdir -p /out/app /out/etc/ssl/certs; \
    cp "/tmp/target-app/release/${APP_NAME}" "/out/app/${APP_NAME}"; \
    cp /etc/ssl/certs/ca-certificates.crt /out/etc/ssl/certs/ca-certificates.crt; \
    \
    ldd "/out/app/${APP_NAME}" \
      | awk '{for (i=1;i<=NF;i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | xargs -r -I '{}' cp -v --parents '{}' /out/; \
    \
    LIBCURL="$(ldconfig -p | awk '/libcurl\.so\.4/{print $NF; exit}')" ; \
    test -n "$LIBCURL" ; \
    cp -av --parents "$LIBCURL" /out/; \
    ldd "$LIBCURL" \
      | awk '{for (i=1;i<=NF;i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | xargs -r -I '{}' cp -v --parents '{}' /out/; \
    \
    chown 65532:65532 "/out/app/${APP_NAME}"

############################
# Builder (llama-server)
############################
FROM debian:bookworm AS llama_builder
WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    cmake \
    build-essential \
    pkg-config \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /src/llama.cpp
WORKDIR /src/llama.cpp

RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_EXAMPLES=ON \
 && cmake --build build --config Release --target llama-server -j "$(nproc)" \
 || cmake --build build --config Release --target server -j "$(nproc)"

RUN set -eux; \
    mkdir -p /out/usr/local/bin; \
    LLAMA_SERVER="$(find build -type f -name 'llama-server' -perm -111 | head -n 1)"; \
    test -n "$LLAMA_SERVER"; \
    cp -v "$LLAMA_SERVER" /out/usr/local/bin/llama-server; \
    strip /out/usr/local/bin/llama-server || true; \
    ldd /out/usr/local/bin/llama-server \
      | awk '{for (i=1;i<=NF;i++) if ($i ~ /^\//) print $i}' \
      | sort -u \
      | xargs -r -I '{}' cp -v --parents '{}' /out/; \
    chown 65532:65532 /out/usr/local/bin/llama-server

############################
# Runtime (distroless)
############################
FROM gcr.io/distroless/cc-debian12:nonroot
ARG APP_NAME
ARG MODEL_FILE

WORKDIR /app

COPY --from=builder /out/ /
COPY --from=llama_builder /out/ /

# Bake the model into the image (no bind mounts required at runtime)
COPY models/${MODEL_FILE} /models/${MODEL_FILE}

ENV PATH=/usr/local/bin:/app
ENV BIND_HOST=0.0.0.0
ENV BIND_PORT=3000
ENV MODEL_PATH=/models/${MODEL_FILE}
ENV LLAMA_SERVER_PATH=/usr/local/bin/llama-server

EXPOSE 3000

ENTRYPOINT ["/app/granite-4-tiny"]
