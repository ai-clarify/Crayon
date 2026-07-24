# Crayon Evolution Plan — Current Gaps (post-0.5.0)

This document supersedes `evolution-plan-legacy-0.3.0.md`. That plan was
written against the 0.3.0 codebase; M3 (retry classification), M5 (locality
scheduling), and the coordinator side of M4 (graceful drain) all shipped in
0.4.0 (see `CHANGELOG.md`). This plan covers only what is still missing or
broken in the current checkout.

---

## What already exists (do not redo)

| Item | Status | Location |
|---|---|---|
| Failure classification (M3) | ✅ Done | `FailureClass` enum `protocol.rs:291`; `fail()` retry decision `coordinator.rs:626` |
| Object-locality scheduling (M5) | ✅ Done | 16-entry look-ahead in `assign_next` `coordinator.rs:421-454` |
| Coordinator SIGTERM drain (M4 coord side) | ✅ Done | `cluster.rs:108-130` — stop accept, drain handlers, 10s timeout |
| Zero-copy object store | ✅ Done | `Arc<[u8]>` payloads, `data_plane.rs` |
| Shared-memory arena (0.5.0) | ✅ Done | large objects bypass 8 MiB frame cap |

---

## Gap 1 — Worker-side failure classification is too coarse (P1)

### Problem
`crayon_cluster.rs:639` marks *every* input-fetch error as
`FailureClass::Transient`. `crayon_cluster.rs:659` marks *every* operation
error as `FailureClass::Transient`. Only a panic (`:662`) is `Permanent`.

But `fetch_inputs` can return `ObjectLost` (the owning worker died and the
object is gone forever — retries will never recover it in Crayon's
no-lineage model). And an operation can fail on deterministic errors: bad
input shape, codec mismatch, invalid argument. Retrying these burns
`max_attempts` and masks the real bug.

### Fix
Classify by error variant, not by source:

- **Transient** (retry): transport error, deadline exceeded, worker loss,
  coordinator disconnect, `ObjectNotFound` for a still-`Reserved` output
  (producer not done yet).
- **Permanent** (terminal): `ObjectLost`, `CodecMismatch`, serialization
  failure, `InvalidArgument`, `OperationUnavailable`, any `Protocol` error
  from the operation itself.

Add `impl Error { pub fn failure_class(&self) -> FailureClass }` in
`error.rs`. Replace the hardcoded `FailureClass::Transient` at
`crayon_cluster.rs:639` and `:659` with `error.failure_class()`.

### Test
- Unit: `error_failure_class_maps_lost_to_permanent`,
  `error_failure_class_maps_transport_to_transient`.
- Integration: submit a task whose op always returns `Error::InvalidArgument`
  → assert exactly 1 attempt, task `Failed`.

### Effort
~0.5 day.

---

## Gap 2 — Worker has no SIGTERM / graceful drain (P0)

### Problem
The coordinator drains on SIGTERM (`cluster.rs:108`), but the worker
(`crayon_cluster.rs`, 1000 lines) has **no signal handling at all**. When K8s
sends SIGTERM to a worker pod:
1. The process dies immediately.
2. Its in-flight task is killed mid-execution.
3. The coordinator only finds out when the lease expires (reaper,
   `cluster.rs:89-100`).
4. The task retries from scratch — all partial work lost.

For RL rollouts this means a partially-computed trajectory is discarded and
re-run.

### Design — worker drain state machine

```
Alive → Draining → Gone
```

**Worker side:**
1. Install `tokio::signal::unix::signal(SIGTERM)` in `run_worker`
   (`crayon_cluster.rs:284`). On signal, set `Arc<AtomicBool>` draining.
2. **Immediately** send `WorkerRequest::Drain(identity)` to the coordinator
   — do this from the signal branch, not from the poll loop top, so the
   coordinator learns about the drain even if a long task is executing.
3. Stop polling for *new* work. If a task is in flight, let it finish — but
   start a **signal-relative** grace timer (not a fixed 25s from task start).
   If the task doesn't finish before the grace budget, abort it.
4. Exit.

**Coordinator side:**
1. `WorkerState` (`coordinator.rs:20`) gains `Draining`.
2. New `WorkerRequest::Drain(WorkerIdentity)` variant in `protocol.rs`.
   Handler: set worker state to `Draining`, wake long-polls.
3. `assign_next` (`coordinator.rs:413`) already returns `None` for
   non-`Alive` workers — `Draining` is covered, no new assignment logic.
4. **Critical fix:** `check_identity` (`coordinator.rs:855`) currently
   requires `worker.state == WorkerState::Alive`. This would reject
   completions/failures from a `Draining` worker. Change to
   `worker.state != WorkerState::Dead` so a draining worker can still report
   its in-flight task's outcome.
5. A `Draining` worker that exits without completing → lease reaper
   (`expire_workers`, `coordinator.rs:642`) re-queues the task as today.

### Abandon vs Failed on grace timeout

Do **not** report `Failed { class: Permanent }` on drain timeout — that
marks the task terminal and the lease reaper won't retry it
(`coordinator.rs:651` only re-queues tasks that still have an assignment).

Instead, the worker should just exit without reporting. The lease reaper
will then re-queue the task normally. This is the lowest-entropy path: no
new coordinator transition, no protocol change for the timeout case.

If the worker *can* report (task finished just before timeout), it sends
`Completed`/`Failed` as normal — the `check_identity` fix above lets it
through.

### Protocol change

Adding `WorkerRequest::Drain` is a wire-shape change. Per
`protocol.rs:21` and the project's own convention (0.4.0 bumped major 2→3
for the `FailureClass` change), this requires **`PROTOCOL_MAJOR` 4→5**.
Deploy all components together.

### Test plan
- **Unit (coordinator):**
  - `draining_worker_gets_no_assignments`: set `Draining`, `assign_next`
    returns `None`.
  - `draining_worker_can_complete`: `check_identity` accepts `Draining`;
    `complete()` on a draining worker's fence succeeds.
  - `draining_worker_expiry_requeues`: worker `Draining` with in-flight
    task, lease expires → task re-queued.
- **Integration:** start worker, assign a long `sleep` task, send SIGTERM →
  assert (a) no new task assigned after drain, (b) in-flight task either
  completes (if < grace) or is re-dispatched after worker exit.

### Effort
~1.5 days. Signal handler + `Drain` variant + `check_identity` fix +
integration test with process signalling.

---

## Gap 3 — Locality look-ahead is unmeasured (P2, do nothing until measured)

### Current state
`assign_next` (`coordinator.rs:421-454`) scans the first 16 runnable tasks
and picks one whose input objects the polling worker already owns. Falls
back to FIFO.

### Concern (from review)
- The 100MB-input example in the legacy plan is wrong: worker outputs are
  frame-bounded; large inputs come from the client arena, whose objects have
  `owner: None` (`coordinator.rs:277`). Locality only helps ≤8 MiB
  worker-produced intermediates.
- The Zaharia delay-scheduling numbers (95% locality, <10% delayed) come
  from HDFS replication + thousands of slots; they don't transfer to
  slots=1, no-replication Crayon.

### Action
**Do not rewrite the scheduler.** First measure on `v100`: what fraction of
RL rollout tasks have a worker-owned intermediate input, and what fraction
of those hit the local worker vs refetch. If locality hit rate is already
>80% with the 16-entry look-ahead, leave it. If it's low and refetch cost
is measurable, then consider a skip-counter — but only with data.

### Effort
~1 day for measurement; 0 for code unless data demands it.

---

## What is explicitly not in this plan

- **Coordinator durability / HA** — ephemeral-worker RL workload; durability
  is near-worthless (killed workers' outputs are lost regardless). Pursue
  availability (fast restart + M2 reconnect, already done), not durability.
- **mTLS** — only when cross-host deployment demand appears.
- **Streaming objects** — 0.5.0's shared-memory arena already lifts the
  8 MiB cap for same-host transfers; network streaming only matters for
  cross-host large objects, which isn't the current bottleneck.
- **Actors** — need durability; deprioritized with it.
- **Replacing the locality scheduler** — unmeasured; do nothing until data.

---

## Summary

| Gap | Priority | Effort | Status |
|---|---|---|---|
| 1. Worker failure classification | P1 | 0.5 day | Ready to implement |
| 2. Worker SIGTERM drain | P0 | 1.5 days | Ready to implement |
| 3. Locality measurement | P2 | 1 day | Measure first, code only if needed |

Total: ~2 days of implementation + 1 day of measurement.
