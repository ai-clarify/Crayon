# Crayon Roadmap — Post-P0 Hardening

This document tracks the remaining work after the single-session runtime hardening
(epoch fencing, immutable operation catalog, capacity budgets, panic isolation,
loopback enforcement). It is the source of truth for the next implementation
pass.

## Status

- P0 hardening: **done** (commit `b232aef`).
- This pass: explicit `Release`, worker reconnect state machine.
- Deferred: coordinator durability, mTLS, lineage GC, streaming objects, actors.

## Milestone 1 — Explicit Object Release

### Problem

Task outputs and client `put` objects live forever in the coordinator metadata
and the owning worker's `LocalObjectStore`. Under a fixed capacity budget the
cluster eventually rejects all new work. Distributed reference counting is
deferred; the minimal correct fix is an explicit, coordinator-mediated release.

### Design

1. Add `ClientRequest::Release(ObjectId)` and `ClientReply::Released`.
2. Coordinator tracks an internal reference count per object:
   - `put` starts at 1 (client-owned).
   - task output starts at 1 (producer-owned).
   - each task that lists the object as a `TaskArg::Object` increments the count
     at submit time; the count is decremented when the task reaches a terminal
     state (`Succeeded`/`Failed`/`Cancelled`).
3. `Release` decrements the client/producer reference. An object whose count
   reaches 0 is deleted:
   - coordinator removes the metadata entry;
   - if the object is worker-owned, the coordinator sends a delete instruction
     to the owning worker on the worker's next poll (reuse the existing
     `WorkerReply` channel, e.g. `WorkerReply::DeleteObject(ObjectId)`);
   - coordinator-owned inline objects are dropped immediately.
4. Releasing an object still referenced by a non-terminal task returns
   `Error::ObjectInUse(ObjectId)`. Releasing an unknown object is idempotent
   (`Released`).
5. `ObjectRef<T>` gains `release(&self)` on the client.

### Invariants

- An object is readable until its count hits 0.
- A task can never observe a dependency disappear before it finishes.
- Release is a mutation: it requires the coordinator epoch and is replay-cached.

### Cost

~2–3 days. Touches `protocol.rs`, `coordinator.rs`, `cluster.rs`, `client.rs`,
`worker.rs`, and one real-process test.

## Milestone 2 — Worker Reconnect State Machine

### Problem

A transient network error or coordinator restart currently exits the worker
process. Recovery depends entirely on an external supervisor. The worker should
treat loss of the coordinator session as a recoverable state, not a fatal one.

### Design

Worker lifecycle becomes an explicit state machine:

```text
Disconnected -> Registering -> Active -> Fenced/Disconnected -> Backoff -> Registering
```

1. `run_worker` wraps the poll loop in a loop that re-registers on failure.
2. On `StaleEpoch`, `StaleFence`, transport error, or unexpected coordinator
   reply:
   - stop accepting new assignments;
   - drop the current `WorkerIdentity`;
   - generate a new `WorkerEpoch`;
   - sleep with capped exponential backoff + jitter;
   - re-register.
3. In-flight attempts from the old session are abandoned. Their leases expire at
   the coordinator and the tasks retry per `max_attempts`. The worker must not
   report completions from an old session to a new one.
4. Heartbeat loop shares the same backoff/exit signal; it stops when the session
   is dropped.
5. A hard fatal error (invalid config, bad operation registration) still exits
   the process.

### Backoff

- initial 100 ms, factor 2, cap 5 s, full jitter.
- reset to initial after a successful registration that holds for one lease
  period.

### Invariants

- A worker never reports under a stale `WorkerIdentity`.
- A worker never holds two live sessions simultaneously.
- Transient coordinator restarts do not require an external process manager.

### Cost

~1 week. Touches `crayon_cluster.rs` worker loop and `coordinator.rs` expiry
logic (no coordinator changes needed beyond what already exists).

## Milestone 3 — Fail-closed Retry Classification (P1, after M1/M2)

### Problem

`retryable: bool` is set to `true` for every worker failure. Deterministic
errors (bad input, codec mismatch, `ObjectLost`) waste capacity and amplify
side effects under `max_attempts > 1`.

### Design

- Default non-retryable. Retry only a whitelist: transport errors, coordinator
  disconnect, worker lease expiry.
- `Protocol`, `Serialization`, `ObjectLost`, `ObjectConflict`,
  `OperationUnavailable`, `IllegalTransition` are terminal.
- Task terminal state stores a stable error category, not just a string.

### Cost

~2–3 days for centralized classification; ~1 week for a structured
`Retryable`/`Terminal` operation error API.

## Milestone 4 — Graceful Drain (P1)

### Design

- Worker catches `SIGTERM`/`Ctrl-C`.
- Enters `Draining`: stops polling for new work, keeps the object endpoint
  alive, gives current attempts a bounded grace period.
- Sends an explicit unregister on completion or timeout.
- Coordinator reuses the worker-expiry transition but triggers it immediately
  instead of waiting for the lease.

### Cost

~1 week.

## Milestone 5 — Object Locality (P1)

### Design

- Worker fetch path checks the local `LocalObjectStore` before going over the
  network.
- Scheduler scores polled workers by total local input bytes; resource and
  capability checks remain hard constraints.

### Cost

~3–5 days.

## Definition of Done for This Pass

- `Release` deletes coordinator metadata and worker-local bytes; a subsequent
  `Get` returns `ObjectNotFound`.
- Releasing an object with a live dependent task returns `ObjectInUse`.
- Killing the coordinator process and restarting it causes workers to
  re-register without an external supervisor; new tasks run after reconnect.
- A worker that loses the network for 10 s and recovers resumes serving tasks.
- All existing 6 real-process tests still pass; new tests cover release and
  reconnect.
- `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test` clean.
- Full task simulation passes on `v100`.
