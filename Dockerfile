# syntax=docker/dockerfile:1

# ---- build stage ----
FROM rust:1-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY tests ./tests
RUN cargo build --release

# ---- runtime stage ----
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/crayon-demo /usr/local/bin/
COPY --from=builder /app/target/release/crayon-multi /usr/local/bin/
ENV RUST_LOG=info
CMD ["crayon-demo"]
