# Build stage
FROM rust:1.82-slim AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo build --release -p ai-gateway

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/ai-gateway /usr/local/bin/ai-gateway
COPY crates/ai-gateway/config.example.yaml /app/config.yaml

WORKDIR /app

# Persist the encryption master key (and encrypted secrets) outside the
# container layer. Mount a volume here so keys survive restarts/rebuilds.
ENV AI_GATEWAY_DATA_DIR=/data
RUN mkdir -p /data
VOLUME ["/data"]

# Bind the OAuth callback server to all interfaces so the host browser
# redirect to localhost:1455 can reach the container.
ENV OAUTH_CALLBACK_BIND_HOST=0.0.0.0

EXPOSE 8080 1455

ENTRYPOINT ["ai-gateway"]
CMD ["--config", "/app/config.yaml"]
