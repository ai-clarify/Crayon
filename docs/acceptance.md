# Acceptance

## Pull-request gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo check --all-targets
cargo test
```

`tests/cluster_process.rs` allocates dynamic loopback ports, starts independent
coordinator and worker processes, submits a task through the public binary, and
requires the remote result `42`. Child processes are killed and reaped on every
exit path.

## Required invariants

Unit tests cover:

- exact operation descriptor and codec validation;
- fixed-point resource validation and capped release;
- idempotent worker registration;
- unschedulable resource rejection;
- attempt/lease stale-result rejection;
- dependency failure propagation;
- cancellation resource retention until worker acknowledgement;
- content-derived object IDs and immutable object bytes;
- protocol cluster/deadline validation.

## Manual smoke

```bash
cargo build --bin crayon-cluster
target/debug/crayon-cluster coordinator 127.0.0.1:7000 &
target/debug/crayon-cluster worker 127.0.0.1:7000 127.0.0.1:7001 &
target/debug/crayon-cluster submit 127.0.0.1:7000 20 22
```

Expected output: `42`.

## Deferred acceptance

Before claiming production readiness, add repeated worker-kill/retry tests,
large chunked object transfer, process-level cooperative cancellation, and
coordinator restart/re-registration tests. These are not claimed by the current
release.
