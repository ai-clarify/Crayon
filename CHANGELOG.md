# Changelog

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
