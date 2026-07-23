# Changelog

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
