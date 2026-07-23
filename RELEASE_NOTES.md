# Crayon v0.2.0

First coordinator-based Crayon runtime release.

## Compatibility notice

This release replaces the 0.1.x in-process/Python architecture. Python
bindings, actors, and the old RL benchmark are **not** part of 0.2.0. Wire
protocol major version 2; coordinator, worker, and client processes must be
upgraded together.

## Features

- One authoritative coordinator and independent worker processes.
- Registered versioned Rust operations with fixed-point resource routing.
- Fenced task attempts and worker-local immutable object data plane.
- Direct worker-local object fetch with `ObjectId`, size, and BLAKE3 verification.
- `cluster-benchmark` binary for task and DAG/object transfer measurements.

## Reliability

- RPC retries reuse the original request ID; the coordinator replays the first
  response within one live coordinator epoch.
- Typed `TaskFailed` and `TaskCancelled` results.
- Transient heartbeat transport errors do not permanently stop heartbeats.
- Coordinator can bind non-loopback addresses for container deployments.

## Benchmark

V100 host (8 logical CPUs, Intel Xeon Platinum 8260, Linux, loopback), release
binaries, 3 warmups + 30 measured samples, 4 KiB DAG payload:

| Scenario | Workers | Concurrency | Median latency | Throughput |
|---|---:|---:|---:|---:|
| Task (`add`) | 1 | 1 | 53.0 ms | 19.2 ops/s |
| DAG/object (`copy`) | 1 | 1 | 102.0 ms | 9.6 ops/s |

Raw benchmark artifacts are attached to this release. See
[docs/benchmark.md](docs/benchmark.md) for methodology.

## Install

```bash
cargo install crayon-rs --version 0.2.0
```

## Limits

- Task execution is at-least-once; no exactly-once external side effects.
- No coordinator persistence/HA, TLS/authentication, hard preemption, or
  distributed reference counting.
