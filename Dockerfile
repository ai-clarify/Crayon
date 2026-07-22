# syntax=docker/dockerfile:1
FROM rust:1-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --bin crayon-cluster

FROM debian:bookworm-slim
RUN useradd --system --create-home crayon
COPY --from=builder /app/target/release/crayon-cluster /usr/local/bin/crayon-cluster
USER crayon
ENTRYPOINT ["crayon-cluster"]
