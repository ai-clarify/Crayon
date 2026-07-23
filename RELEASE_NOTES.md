# Crayon v0.4.0

Reliability and scheduling release: hardens the control plane against the
worker/coordinator failure edges found in review, adds a zero-copy object store,
and implements the deferred M3–M5 roadmap items. See [CHANGELOG.md](CHANGELOG.md)
for the full list.

## Compatibility notice

Wire protocol **major version 3** (up from 2): `WorkerRequest::Failed` now
carries a typed `FailureClass` instead of a `retryable` bool, a wire-incompatible
change. Coordinator, worker, and client processes must be upgraded together; a
mixed-version cluster is rejected at the envelope check.

## Highlights

- **Retry classification (M3).** A typed `FailureClass` (`Transient` /
  `Permanent`) replaces the worker-supplied `retryable` bool. The coordinator
  owns the retry policy — an operation panic is `Permanent` and terminal even
  with attempts left, so a crashing op no longer burns its retry budget.
- **Graceful drain (M4).** The coordinator handles `SIGTERM`/`SIGINT`: stop
  accepting, drain in-flight handlers, exit cleanly instead of a hard kill.
- **Object-locality scheduling (M5).** `assign_next` prefers a task whose input
  objects the polling worker already owns (bounded look-ahead, FIFO fallback),
  cutting cross-worker object refetches for DAGs.
- **Zero-copy object store.** Payloads are `Arc<[u8]>`, shared across
  completions, replies, and the replay cache instead of cloned.

## Reliability & hardening

- `Release` of a still-`Reserved` output is rejected, and a released output
  reclaims its terminal producer task — closing a coordinator panic cascade and
  the unbounded task-table growth that made `MAX_TASKS` a lifetime cap.
- Dead ephemeral workers are evicted from the registry; unknown-task-id worker
  reports return `TaskNotFound` instead of panicking a connection handler.
- `GetBatch` bounds its cloned inline bytes; the replay cache checks capacity
  before dispatch (no duplicate-on-retry) and clamps entry TTL to a server bound.
- A busy worker survives a `DeleteObject` for one of its outputs; mid-execution
  lease loss or transient transport errors reconnect instead of killing it.

## Benchmark

V100 host (8 logical CPUs, Intel Xeon Platinum 8260, Linux, loopback), release
binaries, 3 warmups + 30 measured samples, 4 KiB DAG payload:

| Scenario | Workers | Concurrency | Median latency | Throughput |
|---|---:|---:|---:|---:|
| Task (`add`) | 1 | 1 | 0.5 ms | 2115 ops/s |
| DAG/object (`copy`) | 1 | 1 | 23.2 ms | 43 ops/s |
| Task (`add`) | 8 | 8 | 1.3 ms | 684 ops/s |
| DAG/object (`copy`) | 8 | 8 | 24.5 ms | 41 ops/s |

Throughput is single-stream (`1 / mean` latency). Task latency is ~100× lower
than the 0.2 polling runtime (53 ms), sustained from 0.3.0's event-driven
dispatch. See [docs/benchmark.md](docs/benchmark.md) for methodology.

## Install

```bash
cargo install crayon-rs --version 0.4.0
```

## Limits

- Task execution is at-least-once; no exactly-once external side effects.
- No coordinator persistence/HA (a `durability_spike` proves it is a cheap
  add-on, but it is deliberately unbuilt — see `docs/feature-prescreen.md`),
  TLS/authentication, hard preemption, or distributed reference counting.
