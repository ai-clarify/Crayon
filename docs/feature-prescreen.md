# Crayon New-Feature Pre-Research

Portfolio pre-research of the features explicitly deferred in `roadmap.md:12`
("coordinator durability, mTLS, lineage GC, streaming objects, actors"). Each
was assessed against the actual source — not the roadmap prose — and scored on
value, feasibility, cost, and risk. The top pick is de-risked by a runnable
spike (`src/bin/durability_spike.rs`).

Scope note: the roadmap's M1 (explicit `Release`) and M2 (worker reconnect) are
**done** (CHANGELOG 0.2.1); M3–M5 (retry classification, graceful drain, object
locality) already have concrete designs and are implementation-ready, not
pre-research. This document covers only the five items where the design has a
real architectural fork.

## Operational reframe — ephemeral-container RL (authoritative)

The generic ranking below anchored on the roadmap's "deferred" list and a
generic "in-memory coordinator is bad, persist it" prior. Corrected against the
actual target workload — **distributed RL rollouts where workers are ephemeral
containers, spawned per task and killed on completion, with at-least-once
fresh-container retry by design** — the priorities change materially:

- **The real bottleneck is dispatch/result latency, not durability.** A single
  `add` task's 53 ms median (README benchmark) is almost entirely two fixed
  sleeps: the worker idle-poll `sleep(50 ms)` (`crayon_cluster.rs:331`, when
  `Poll` returns no work) plus the client status-poll `sleep(≤25 ms)`
  (`client.rs:277`). Compute and loopback RTT are negligible against them.
  **Event-driven dispatch — a long-poll `Poll{wait_ms}` the coordinator holds
  until work is ready, and a blocking `Get{wait_ms}` that waits on the reserved
  output — collapses both to sub-millisecond RTT: ~50× the single-task
  throughput ceiling.** This is the #1 lever and was absent from the five-item
  list; that omission is the report's main error.
- **Durability drops to the bottom.** Ephemerality is the design point, not a
  gap: killed workers' outputs are lost regardless, so a coordinator restart
  already `Lost`s worker-owned objects and re-runs dependents. Persisting
  coordinator state saves only the task-graph structure and completed-but-
  unconsumed results — and in a rollout loop the learner consumes results
  immediately, so there is almost no standing state to save. What matters for
  this workload is coordinator **availability** (fast restart + worker
  auto-reconnect, already delivered by M2), not coordinator **state
  durability**. The spike below still holds — durability is genuinely cheap (7
  derives) — but cheap ≠ needed.

**Workload-specific ranking:**

| # | Feature | Why |
|---|---|---|
| 1 | **Event-driven dispatch** (`Poll{wait_ms}` + blocking `Get{wait_ms}`) | Kills the 50 ms + 25 ms poll latency that *is* the 53 ms benchmark; ~50× per-task throughput. |
| 2 | Streaming objects | Only if RL trajectories/observations are large enough to hit the 8 MiB frame cap. |
| 3 | mTLS / authentication | Situational — ephemeral containers span nodes, but RL clusters are often a trusted VPC. |
| 4 | Actors (resettable) | Fits ephemerality, but the RL-valuable form (parameter server) needs persistence it cannot have here. |
| 5 | Lineage GC | Low; M1 already solved the P0 leak. |
| — | Coordinator durability | Near-worthless for ephemeral RL; pursue coordinator *availability*, not *durability*. |

The generic assessment below remains technically accurate per feature; only the
prioritization is superseded by this section.

## Ranking (generic / workload-agnostic — superseded above)

| # | Feature | Composite | Value | Feasibility | Risk | Cost |
|---|---|--:|:--:|:--:|:--:|---|
| 1 | **Coordinator durability / HA** | 7.4 | 3 | 4 | 3 | 2–3 wk (snapshot); Raft HA +6–10 wk |
| 2 | mTLS / authentication | 6.3 | 4 | 3 | 3 | 2–3 wk (coarse authz) |
| 3 | Streaming objects | 6.2 | 3 | 3 | 4 | 1–1.5 wk (A) / 4–6 wk (B) |
| 4 | Lineage GC / distributed refcount | 5.2 | 3 | 3 | 4 | 1 wk (P1) / 3–4 wk (P2) |
| 5 | Actors (stateful operators) | 5.0 | 3 | 3 | 4 | 3–4 wk |

Feasibility: 5 = easy on the current architecture. Risk: 5 = most uncertain.

**Why durability is #1 despite value 3:** it wins on *leverage*, not delivered
value. It is a dependency magnet — actors-with-durable-state, real result
durability, and lineage's value all gate on it — sitting on a genuinely clean
seam: a single-mutex `apply(command)` surface (`cluster.rs:40`), a monotonic
`revision` counter (`coordinator.rs:117`), and a state made entirely of
already-`Serialize` parts. Its hardest risk is reducible by a zero-dependency,
sub-day spike (below).

## Recommended sequence

1. **Run the durability spike first** (done — see below). De-risk the dependency
   magnet before any dependent commits engineering.
2. **In parallel, ship the two cheap, independent, invariant-preserving wins that
   do not block on durability:** Streaming *Option A* (chunked immutable
   transfer, removes the 8 MiB frame cap at `protocol.rs:22`, no new dep) and
   Lineage GC *Phase 1* (auto-sweep task-internal intermediates, reusing the
   idempotent `object_in_use` scan + bounded task retention).
3. **Build durability proper** (Option A: periodic full snapshot + effect log,
   epoch preserved, `expire_workers` reused as the post-load reconcile). The
   one-time structural investment that unblocks the rest. Scope it honestly as
   *single-node* durability; document that task **output** bytes stay non-durable
   (worker `LocalObjectStore`, `data_plane.rs:7`).
4. **mTLS Option A** whenever cross-host deployment demand materializes — it is
   additive, needs no protocol bump, and gates the already-exposed
   `allow_remote_bind` path (`crayon_cluster.rs:41`). Pull ahead of step 3 if
   remote deployment is imminent.
5. **After durability lands:** Actors *Option A* (resettable, incarnation-fenced)
   then *Option B* (durable survival, reusing the new sink); Lineage GC *Phase 2*
   (reconstruction-on-loss) once a per-operation determinism contract exists.
   Reach for Raft HA (durability Option C) or live producer→consumer streams
   (streaming Option B) only if HA / live-stream demand is proven.

## Per-feature assessment

### 1. Coordinator durability / HA — value 3, feasibility 4, risk 3

Make the single in-memory coordinator survive a restart (or fail over) without
losing task state, terminal outcomes, resource reservations, or fence identity,
so in-flight attempts are neither double-counted nor resurrected.

- **Recommended:** periodic full-state snapshot + effect log, **preserving the
  persisted `CoordinatorEpoch`** across restart, reusing `expire_workers`
  (`coordinator.rs:485`) as the post-load reconcile sweep. Lowest entropy: reuses
  `bincode` (already a dep), zero new crates, one clean mutation boundary.
- **Highest-risk point:** whether a rehydrated coordinator that *preserves* its
  epoch recovers in-flight attempts without double-counting reservations or
  resurrecting stale attempts — the interaction between a snapshot that still
  holds a reservation and a worker that either re-registers under the same epoch
  or is reaped by lease expiry. **The spike targets exactly this.**
- **Why not a command-replay WAL:** `submit`/`assign_next` mint random ids
  (`ids.rs:12`, `coordinator.rs:281`). Replaying commands regenerates *different*
  ids than clients/workers already hold → divergent state. This forces
  effect/state logging, which the snapshot seam already is.
- **Honest caveat:** this is single-node snapshot durability — a dead host is
  still downtime, not HA — and it does **not** make task output bytes durable
  (worker `LocalObjectStore` is in-memory). The "don't lose my results" promise
  is only partly delivered without also persisting the object plane
  (coupled dependency, tracked separately).
- **Depends on:** M2 reconnect (done); `Serialize` derives on the coordinator
  records; a replay-cache decision under epoch preservation (persist it, or
  accept at-least-once `Submit` across restart — `cluster.rs:43`).

### 2. mTLS / authentication — value 4, feasibility 3, risk 3

A `rustls`-based mutually-authenticated transport giving coordinator, workers,
and clients a cryptographic identity, gating the currently loopback-only,
self-asserted planes.

- **Recommended:** a `tokio-rustls` mTLS wrapper with coarse cert-SAN role
  authorization, required whenever the coordinator binds non-loopback, honoring
  the existing `require_loopback`/`allow_remote_bind` seam (`cluster.rs:54`).
- **Single highest value:** the one gate that unlocks every non-loopback
  deployment, layered additively over an unchanged `Envelope`/fence with **no
  protocol bump**. Note an unauthenticated, fully-impersonable remote path
  already ships today (`crayon_cluster.rs:41` calls `allow_remote_bind`
  unconditionally).
- **Highest-risk point:** not the identity model (confirmed additive) but
  whether the one-RPC-per-TCP-connection model (~20 Hz poll + heartbeats +
  per-object fetch, each a fresh `TcpStream` via `request()`, `cluster.rs:388`)
  makes per-connection TLS handshakes affordable, or forces a
  connection-reuse/pooling refactor. **This is the mTLS spike to run next.**
- **It is a leaf:** nobody depends on it, so it can land any time remote demand
  appears.

### 3. Streaming objects — value 3, feasibility 3, risk 4

Break the whole-blob object model into chunked transfer for large outputs and
(later) live producer→consumer streams, keeping BLAKE3 verification and the
immutable/refcount/fence guarantees.

- **Recommended:** phase it. *Option A* (chunked immutable transfer + streaming
  BLAKE3 verify) removes the real limiter — the 8 MiB frame cap (`protocol.rs:22`)
  and 2×-RAM materialization — in ~1 week, preserves every invariant, needs no
  new dep (a sealed stream is just today's `Object`, reusing M1 refcount/release).
  Defer live streams (*Option B*, 4–6 wk) behind their own milestone.
- **Highest-risk point:** whether a consumer can verify an N-chunk object against
  the pre-committed BLAKE3 root using only `blake3::Hasher` incremental update —
  no whole-object buffer, no new dep — and whether that incremental root is
  bit-identical to today's `checksum`.
- **Depends on:** M1 (done); coordinate with M5 object locality (same fetch
  path); a spill-to-disk store before "large" is meaningful; wire protocol
  major bump 2→3.

### 4. Lineage GC / distributed reference counting — value 3, feasibility 3, risk 4

Automatic object reclamation — coordinator-tracked refcounting and/or
Spark-style reconstruction-on-loss — replacing M1's manual `Release`.

- **Recommended:** phase it; do **not** jump to distributed refcounting — it
  fights the single-authority architecture (`architecture.md:13`), and the
  coordinator is already the sole ref holder. *Phase 1*: auto-sweep
  task-internal intermediates reusing the idempotent `object_in_use` scan
  (`coordinator.rs:606`) + bounded task retention (~1 wk). *Phase 2*:
  reconstruction-on-loss (3–4 wk) once a determinism contract exists.
- **Highest-risk point:** whether automatic reclamation preserves, under worker
  loss and at-least-once retries, the invariant the M1 scan gives for free —
  never remove an object a retry or unscheduled dependent still needs, never leak.
- **Lowest urgency:** M1's manual `Release` already solved the P0 capacity leak;
  Phase 1 is a comfort gain, not a capability gap.

### 5. Actors (stateful long-lived operators) — value 3, feasibility 3, risk 4

Re-introduce long-lived, addressable, stateful operator instances (pinned to a
worker, serving serialized calls) on top of today's stateless
`(namespace,name,version)` runtime.

- **Recommended:** *Option A* (ephemeral pinned operator, explicitly
  **resettable**). The only design consistent with the current non-durable,
  immutable-object, at-least-once architecture without inventing durability. It
  answers the framing question honestly: a stateful actor correctly does **not**
  survive worker loss; epoch+incarnation fencing makes in-memory state *safe*
  (fenced, reset), never *recovered*, until durability lands.
- **Highest-risk point:** whether an actor can be fenced so that across worker
  loss at most one incarnation is ever live and no call/effect from a superseded
  incarnation is accepted — without durable coordinator state.
- **Gated:** Option B (durable survival) is a hard dependent of feature #1. Value
  is capped to resettable roles until durability lands — hence it sequences last.

## The spike — `src/bin/durability_spike.rs`

A throwaway binary (run: `cargo run --bin durability_spike`) that drives the
**real** `CoordinatorState` through an in-flight lifecycle, simulates a crash
with the only durability primitive (bincode serialize → deserialize), and
asserts the recovery model. It required exactly one production change to exist:
`#[derive(Serialize, Deserialize)]` on the seven coordinator record types
(`coordinator.rs`) — every leaf type already derived it — which is itself the
central finding.

What it proves, with runnable asserts:

1. **Exact reconstruction.** After a crash at `revision 7`, the recovered state
   is byte-identical in epoch, revision, task/object ids, terminal states, and
   resource reservations — reservations are *not* reset to full.
2. **Fencing survives (worker reconnects within lease).** A worker still holding
   its pre-crash `WorkerIdentity` + `TaskFence` completes an in-flight task on
   the recovered coordinator; exactly one slot is released (no double-count).
   This is precisely what today's design forbids ("restart ends the session").
3. **Clean reconcile (worker lost across the outage).** From the same snapshot,
   with the worker's lease expired, `expire_workers` returns the retryable
   in-flight attempt to `Runnable`, marks the exhausted one terminally `Failed`
   (neither duplicated), clears every stale assignment, and a completion carrying
   the pre-crash fence is rejected with `StaleFence`.
4. **Snapshot, not command-replay.** Replaying `submit` on a fresh state mints a
   different task id, confirming a naive command-log WAL diverges from
   client-observed ids — the snapshot seam is the correct one.

**Conclusion:** coordinator durability is a persistence-layer add-on on the
existing `revision`-tagged serializable state and the existing `expire_workers`
reconcile — **not a coordinator redesign.** The remaining engineering is the
snapshot/effect-log I/O layer, group-commit latency, and the replay-cache
decision — all outside the state machine's correctness core, which the spike
shows is already recovery-safe.

## Also considered (adjacent, ranked lower)

- **Observability / metrics / tracing** — high value for operating any of the
  five, but cross-cutting tooling, not a runtime capability; unblocks nothing.
  Follow, don't precede.
- **Speculative execution / straggler mitigation** — deliberately racing
  duplicates amplifies at-least-once side effects; premature before M3 retry
  classification + a determinism contract.
- **Control-plane sharding / coordinator scale-out** — fights the deliberate
  single-authority invariant; bounded state (16k tasks / 64k objects) fits in
  memory. Raft HA (durability Option C) is the principled fault-tolerance path.
- **Autoscaling / dynamic worker pool** — workers already register/expire
  dynamically; elastic sizing is an orchestration-layer (k8s) concern.
- **Durable / spill-to-disk worker object store** — genuinely needed for real
  result durability and coupled to *both* coordinator durability and large
  streaming; tracked as a coupled dependency, not a standalone feature.
</content>
