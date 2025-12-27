# syntax=docker/dockerfile:1.6
ARG RUST_VERSION=1.88.0
ARG APP_NAME=granite-4-tiny

# Builder
FROM rust:${RUST_VERSION}-bookworm AS builder
ARG APP_NAME
WORKDIR /app

# Include libcurl runtime so we can copy libcurl.so.4 into distroless
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Build deps using a dummy main.rs into a dedicated target dir
RUN mkdir -p src && printf '%s\n' 'fn main() {}' > src/main.rs
ENV CARGO_TARGET_DIR=/tmp/target-deps
RUN cargo build --release --locked

# Now copy real sources and build into a different target dir
COPY src ./src
ENV CARGO_TARGET_DIR=/tmp/target-app
RUN cargo build --release --locked

# Collect runtime artifacts (app + TLS CA certs + app shared libs + libcurl shared libs)
RUN set -eux; \
    mkdir -p /out/app /out/etc/ssl/certs; \
    cp "/tmp/target-app/release/${APP_NAME}" "/out/app/${APP_NAME}"; \
    cp /etc/ssl/certs/ca-certificates.crt /out/etc/ssl/certs/ca-certificates.crt; \
    \
    # Copy shared libs required by the Rust binary
    ldd "/out/app/${APP_NAME}" \
      | awk '{print $3}' \
      | grep -E '^/' \
      | xargs -r -I '{}' cp -v --parents '{}' /out/; \
    \
    # Copy libcurl.so.4 (needed by your mounted llama-server) and its dependencies
    LIBCURL="$(ldconfig -p | awk '/libcurl\.so\.4/{print $NF; exit}')" ; \
    test -n "$LIBCURL" ; \
    cp -av --parents "$LIBCURL" /out/; \
    ldd "$LIBCURL" \
      | awk '{print $3}' \
      | grep -E '^/' \
      | xargs -r -I '{}' cp -v --parents '{}' /out/; \
    \
    chown 65532:65532 "/out/app/${APP_NAME}"

# Runtime (distroless)
FROM gcr.io/distroless/cc-debian12:nonroot
WORKDIR /app
COPY --from=builder /out/ /

ENV BIND_HOST=0.0.0.0
ENV BIND_PORT=3000

EXPOSE 3000

ENTRYPOINT ["/app/granite-4-tiny"]
