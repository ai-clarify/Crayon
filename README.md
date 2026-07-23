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

## Install

```bash
cargo install crayon-rs --version 0.2.0
```

The package name is `crayon-rs`, the library import name is `crayon`, and the
binary is `crayon-cluster`.

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

## Benchmark

`cluster-benchmark` measures the current coordinator/worker runtime with real
processes on loopback. See [benchmark methodology](docs/benchmark.md).

V100 host (8 logical CPUs, Intel Xeon Platinum 8260, Linux, loopback), release
binaries, 3 warmups + 30 measured samples, 4 KiB DAG payload:

| Scenario | Workers | Concurrency | Median latency | Throughput |
|---|---:|---:|---:|---:|
| Task (`add`) | 1 | 1 | 53.0 ms | 19.2 ops/s |
| DAG/object (`copy`) | 1 | 1 | 102.0 ms | 9.6 ops/s |
| Task (`add`) | 2 | 2 | 53.2 ms | 18.9 ops/s |
| DAG/object (`copy`) | 2 | 2 | 102.0 ms | 9.8 ops/s |

These are absolute control-plane measurements for the 0.2 runtime. The removed
0.1.x Python/V100 RL benchmark tested a different architecture and is not
evidence for this release.

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
