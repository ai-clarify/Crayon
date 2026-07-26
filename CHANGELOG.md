# Changelog

## Unreleased

### Fixes

- **Same-host put 5× regression.** 0.6.1's release-grace (a 5 s quarantine before
  a freed arena slot returned to the free list) meant any put/read/release burst
  shorter than 5 s — the storage benchmark and every ephemeral RL worker — never
  recycled a slot, so each put's memcpy hit a cold, never-faulted arena page.
  Measured on v100, 1 MiB: put 855→171 µs, get 190→113 µs. Released slots now
  recycle immediately (LIFO), reusing warm pages. The reader/writer race the
  quarantine masked is an application use-after-free (a client releasing an
  object another still reads) — out of scope, as the arena does no reader
  refcounting. The unrelated reservation-leak fix (TTL reap) stays.

## 0.6.1

Ergonomics: one-call local cluster, the `ray.init()` analogue. No wire change
(protocol stays 7).

### Features

- **`crayon-cluster local [workers] [ops]`** and **`crayon.local_cluster(workers=N)`**
  spawn a coordinator + N workers on this host and (Python) return a connected
  client; the child processes are killed when the handle is dropped. Removes the
  three-terminal, position-argument dance for local development. `local_cluster`
  resolves the `crayon-cluster` binary from `$CRAYON_CLUSTER_BIN` or `PATH`.
  Both share one `LocalCluster` spawner in the library (`crayon::LocalCluster`).

## 0.6.0

Cross-host and hot-path release: large objects now cross host boundaries via
chunked streaming, the dispatch hot path sheds redundant copies and an O(n) scan,
and a first-K-ready fetch primitive lands. Wire protocol major 4 → 7 (chunk
variants, `GetBatch.min_ready`, and the failure-class/drain work from the 0.5.x
line all shift request/reply encodings); deploy all components at the same major.

### Features

- **Cross-host chunked streaming.** A `>8 MiB` object crossing a host boundary
  was a hard failure (the RPC frame caps a single object at ~8 MiB, and only the
  same-host arena could exceed it). New `PutChunk`/`GetChunk` stream such objects
  as frame-sized chunks the coordinator writes into / reads out of its arena, so
  a cross-host client can now put and get multi-hundred-MiB objects. Measured
  cross-host (RTT 0.66 ms): 256 MiB round-trips at ~200 MB/s put, ~247 MB/s get.
  Chunk ranges are bounds-checked server-side; the whole object is checksum-
  verified after reassembly (no per-chunk hashing). Same-host puts still use the
  zero-copy arena path unchanged.
- **First-K-ready fetch (`ray.wait`).** `ClusterClient::wait` (and Python
  `Client.wait`) return as soon as at least `min_ready` of N objects resolve,
  instead of blocking on the slowest. The enabling primitive for pipelined /
  off-policy RL loops; `results` / `get_many` keep the all-or-nothing semantics.
- **Server-side terminal-task reclaim.** A fire-and-forget or crashed client that
  never calls `Release` can no longer wedge new submits: terminal task records are
  reclaimed after a TTL backstop, bounded independently of client behavior.

### Performance

- **~2× cross-host put throughput.** The TCP put path dropped three redundant
  full-payload passes — a re-hash, a re-clone into the arena, and echoing the
  whole payload back in the reply the caller discarded. Measured cross-host:
  ~80 → ~170 MB/s put. Same-host arena put also gains (~970 → ~1440 MB/s) from
  the dropped re-hash.
- **O(1) replay-cache dispatch.** Every mutation ran two O(n ≤ 16384) scans under
  the coordinator lock (a full expiry sweep and a byte-total sum); both are gone —
  a running byte counter and lazy expiry make the common path a single map op.

### Fixes

- Two arena slot-lifecycle bugs on the same-host large-object path (a crashed
  writer's reservation could leak; a released slot could be recycled under an
  in-flight reader) — now reaped after a TTL / held through a release grace.
- Retry backoff (exponential, capped) so a transient fault outlasting the instant
  retry burst does not exhaust `max_attempts`.

### Internal

- `coordinator.rs` / `cluster.rs` carry a contract-first `//!` header and moved
  their test modules out; the state machine stays one aggregate (measured: the
  global lock is not the dispatch bottleneck — avg hold 2.2 µs).
- `storage-benchmark --coordinator <addr>` benchmarks an external coordinator for
  real cross-host measurement.

## 0.5.0

Data-plane and RL-pipeline release: a shared-memory object arena lifts the object
size cap and makes same-host transfer zero-copy, a Python client lands, and a
real-LLM RL pipeline ships end to end. Wire protocol major 3 → 4 (arena request
variants + connection pooling shift request encodings); deploy all components at
the same major.

### Features

- **Shared-memory object arena.** A plasma-style host-local mmap arena: a client
  reserves a slot, writes its bytes straight into the mapping, and commits, so a
  same-host `put`/`get` never crosses the 8 MiB RPC frame cap — objects scale to
  gigabytes. Readers map the arena once and read any object as a slice; cross-host
  peers that cannot map it fall back to the frame-bounded network path. Large
  transfers use parallel BLAKE3 and parallel memcpy, and content hashing is
  skipped for large same-host puts.
- **Python client (`crayon` on PyPI).** pyo3 bindings over `ClusterClient`.
- **Real-LLM RL pipeline.** An `llm-actor` rollout op backed by a resident Python
  LLM sidecar (policy weights synced from the arena once per version, not per
  task), a Rust `llm-judge`, a Python driver, and a micro-batched REINFORCE
  learner that fits on a shared V100.
- **Connection pooling.** The coordinator serves multiple frames per TCP
  connection and clients pool connections, cutting per-RPC connect cost.

### Reliability

- Orphaned `/dev/shm` arenas left by a SIGKILLed coordinator are reaped on the
  next coordinator start (pid-tagged tokens; only ever removes a dead-process
  file), so the arena backing does not leak across restarts.

### Misc

- Ray multi-actor multi-role RL e2e baseline and an `rl-benchmark` bin for
  apples-to-apples comparison; agent contract (`AGENTS.md`).

## 0.4.0

Reliability and scheduling release: hardens the control plane against the
worker/coordinator failure edges found in review, adds a zero-copy object store,
and implements the deferred M3–M5 roadmap items. Wire protocol major 2 → 3 (the
`Failed` report gained a typed failure class); deploy all components together.

### Features

- **Retry classification (M3).** Task failures carry a typed `FailureClass`
  (`Transient` / `Permanent`) instead of a worker-supplied `retryable` bool. The
  coordinator owns the retry policy: a `Permanent` failure — e.g. an operation
  panic — is terminal even with attempts remaining, so a crashing operation no
  longer burns its whole retry budget.
- **Graceful drain (M4).** The coordinator handles `SIGTERM`/`SIGINT`: it stops
  accepting connections, drains in-flight request handlers, and exits cleanly
  instead of being killed mid-request.
- **Object-locality scheduling (M5).** `assign_next` prefers a runnable task
  whose input objects the polling worker already owns (bounded look-ahead),
  cutting cross-worker refetches for DAGs, with a FIFO fallback and no hot-path
  regression.
- **Zero-copy object store.** Object payloads are `Arc<[u8]>`, so completions,
  replies, and the replay cache share bytes instead of cloning them.

### Reliability & hardening

- `Release` of a still-`Reserved` output is rejected, and a released output
  reclaims its terminal producer task — closing a coordinator panic cascade and
  the unbounded task-table growth that made `MAX_TASKS` a lifetime cap.
- Dead workers are evicted from the registry (not left as `Dead` records) and
  their pending deletes dropped — bounding the worker map for ephemeral,
  per-task workers.
- Worker reports for an unknown task id return `TaskNotFound` instead of
  panicking a connection handler; `GetBatch` bounds its cloned inline bytes so a
  crafted batch cannot exhaust coordinator memory under the state lock.
- The mutation replay cache checks capacity before dispatching (no more
  committed-but-uncached mutations that duplicate on retry) and clamps entry TTL
  to a server bound instead of the client-controlled deadline.
- A busy worker survives a `DeleteObject` for one of its outputs; mid-execution
  lease loss or transient transport errors reconnect instead of killing it.

### Performance

- Shorter coordinator critical sections and a hoisted lock-free epoch on the hot
  path; `cancellation_for` and dependency reconciliation scan bounded indexes
  instead of the whole task table. V100 (Xeon 8260, loopback, 3 warmups + 30
  samples, 4 KiB DAG payload): single `add` task 0.5 ms median / 2115 ops/s,
  DAG/object `copy` 23 ms / 43 ops/s — ~100× / 4.4× lower control-plane latency
  than 0.2, sustained from 0.3.0's event-driven rewrite.

### Docs

- `durability_spike` is a runnable binary that proves the four coordinator-
  recovery claims and measures snapshot cost (a full 16 383-task control-plane
  snapshot is 3.4 MiB, ~7 ms to serialize on V100); `feature-prescreen.md`
  corrected to match.

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
