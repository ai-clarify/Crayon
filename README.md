# Crayon

Crayon is a small distributed task runtime for Rust workloads.

It uses one authoritative coordinator, worker processes, registered versioned
operations, fixed-point resources, fenced task attempts, and a worker-local
immutable object data plane.

## Guarantees

- Tasks execute in worker processes; clients never execute them locally.
- Operations use `(namespace, name, version)`; closures are not shipped.
- Worker sessions and task attempts reject stale results.
- Execution is at-least-once; only one fenced result publication is accepted.
- Objects are verified by exact `ObjectId`, size, and BLAKE3 digest.
- Coordinator restart ends the current in-memory cluster session.

## Run

```bash
cargo build --bin crayon-cluster

# terminal 1
target/debug/crayon-cluster coordinator 127.0.0.1:7000

# terminal 2
target/debug/crayon-cluster worker 127.0.0.1:7000 127.0.0.1:7001

# terminal 3
target/debug/crayon-cluster submit 127.0.0.1:7000 20 22
# 42
```

## Verify

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

See [architecture](docs/architecture.md) and [acceptance](docs/acceptance.md).

## Non-goals

The current release does not provide coordinator HA, actors, arbitrary code
shipping, distributed reference counting, lineage replay, hard preemption,
autoscaling, or exactly-once external side effects.
