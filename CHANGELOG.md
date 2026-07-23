# Changelog

## 0.3.0

Performance release: the control plane is now event-driven and batched. On an
identical RL rollout workload (byte-for-byte the same per-task compute, verified
by a matching learning curve), throughput improved ~15× versus 0.2.1 and now
runs 2.5× faster than Ray 2.56.1 on the same host.

### Features

- Batch submit and fetch: `ClusterClient::submit_batch` sends N tasks in one RPC
  and `ClusterClient::results` awaits N outputs in one blocking RPC — the batch
  analogue of `ray.get([refs])`. Collapses per-iteration round-trips from
  `2 * parallelism` to 2.
- `SubmitBatch` / `GetBatch` protocol operations with per-item results, so a
  partial rejection or failure does not sink the whole batch.

### Improvements

- Event-driven long-poll replaces fixed-interval polling. Worker `Poll` and
  client `Get` carry a `wait_ms`; the coordinator parks the request on a `Notify`
  that fires on any state change, so a newly-runnable task or newly-available
  output wakes waiters immediately. A bounded re-check slice covers any missed
  wakeup. This removes the 50 ms worker idle poll and the 25 ms client status
  poll that dominated wall-clock.
- `TaskHandle::result` blocks on a single `Get` instead of a status-poll loop,
  consulting task status only on the rare failure path for a precise error.
- Task completions ship small outputs (≤64 KiB) inline, so the coordinator
  answers a result fetch in one hop instead of redirecting to the worker. Larger
  outputs stay worker-local and are fetched on demand.
- O(1) scheduling: the coordinator keeps a runnable queue and a waiting index, so
  assignment and dependency reconciliation no longer scan the full task table.
  Throughput stays flat as completed tasks accumulate; run length is unbounded.
- Replay cache tracks each entry's serialized size once at insert; the byte-budget
  check sums stored sizes instead of re-serializing every entry per request.
- `TCP_NODELAY` on all control-plane and object connections removes Nagle /
  delayed-ACK latency on small frames.

### Benchmark

- `benchmarks/README.md`: reproducible Crayon-vs-Ray comparison with commands,
  results, and the control-plane efficiency reasons behind the gap.

## 0.2.1

Reliability and verification improvements on top of 0.2.0.

### Fixes

- Replay cache no longer fills under load: only client mutations are cached;
  worker reports are idempotent at the coordinator state machine.
- Worker completions can ship small output bytes inline; the coordinator
  validates size and BLAKE3 checksum before storing them.
- Removed the unimplemented `Await` RPC variant that broke the build.
- Integration test timeouts and sleeps increased to stop flaky failures on
  slower CI runners.

### Features

- `rl_benchmark` binary: long-running distributed RL rollout simulation that
  exercises multi-stage task graphs, object dependencies, retries, and
  cancellation on a real cluster.
- `benchmarks/ray_rl_benchmark.py`: Ray equivalent for apples-to-apples
  comparison.
- Explicit `Release` RPC and worker reconnect state machine.
- `rl_benchmark` supports `-h`/`--help`.

## 0.2.0

First coordinator-based Crayon runtime release. Replaces the 0.1.x
in-process/Python architecture.

### Breaking changes

- Python bindings, actors, and the old RL benchmark are not part of 0.2.0.
- Operations are registered versioned Rust operations; closures and Python
  callables are not shipped.
- Wire protocol major version 2; coordinator, worker, and client processes must
  be upgraded together.

### Features

- One authoritative coordinator and independent worker processes.
- Registered versioned operations with fixed-point resource routing.
- Fenced task attempts and worker-local immutable object data plane.
- Direct worker-local object fetch with `ObjectId`, size, and BLAKE3 verification.
- `cluster-benchmark` binary for task and DAG/object transfer measurements.

### Reliability

- RPC retries reuse the original request ID; the coordinator replays the first
  response within one live coordinator epoch.
- Typed `TaskFailed` and `TaskCancelled` results; failed/cancelled tasks no
  longer time out.
- Transient heartbeat transport errors do not permanently stop heartbeats.

### Limits

- Task execution is at-least-once; no exactly-once external side effects.
- No coordinator persistence/HA, TLS/authentication, hard preemption, or
  distributed reference counting.
