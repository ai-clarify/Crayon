# Crayon Evolution Plan: M3 → M5 → M4

Target workload: distributed RL rollouts on ephemeral worker containers.
Grounded in the current checkout; line anchors are exact.

---

## M3 — Retry Classification (priority 1)

### 1. Problem statement
`WorkerRequest::Failed` carries `retryable: bool` (`protocol.rs:283`), but the
worker hardcodes `retryable: true` for *every* failure — both input-fetch
errors (`crayon_cluster.rs:421`) and application errors
(`crayon_cluster.rs:519`). `coordinator.rs:561` trusts the flag:
`retry = retryable && attempt < max_attempts`. Consequence: a deterministic
application bug (e.g. "add expects two arguments") retries up to `max_attempts`
times on different workers, burning cluster time and masking the real error.
There is no way to declare which error *classes* are transient.

### 2. Industry pattern adopted
**Two-axis failure model + opt-in retryable exception allow-list**
(Ray `retry_exceptions`, Temporal `non_retryable_error_types`,
Celery `autoretry_for`). Default = application errors are terminal; user names
the retryable classes. Paired with **exponential backoff + cap + jitter**
(Temporal/Celery/K8s/Sidekiq all use this). The sliding-window budget
(Nomad `attempts`+`interval`) is rejected as over-engineering for an
ephemeral-worker runtime — `max_attempts` already bounds retries, and worker
churn is the common transient, not a sustained failure storm.

### 3. Design

**Failure taxonomy (coordinator decides, not the worker):**
- `System` failure → always retryable: worker lease expiry (`expire_workers`,
  `coordinator.rs:600`), input fetch failure, dispatch timeout,
  coordinator-driven cancel. These are infrastructure, not the task's fault.
- `Application` failure → terminal *unless* the task's operation declares the
  error class retryable. The worker reports an `error_kind: String`
  (e.g. `"TimeoutError"`, `"ConnectionError"`, `"ValueError"`); the
  coordinator matches against the operation's `retry_on: Vec<String>`
  allow-list.

**Data structures:**
- `OperationDescriptor` (`operation.rs:52`) gains `retry_on: Vec<String>`
  (default empty = app errors terminal). Validate in `validate()`
  (`operation.rs:59`): cap length (e.g. ≤16) and each string ≤128 bytes.
- `TaskRecord` (`coordinator.rs:70`) gains `next_retry_at_ms: Option<u64>` and
  `retry_on: Vec<String>` (snapshot from descriptor at submit, so a later
  descriptor change can't resurrect a terminal task).
- `WorkerRequest::Failed` (`protocol.rs:279`) replaces `retryable: bool` with
  `error_kind: String`. Bump `PROTOCOL_MINOR` to 1 (`protocol.rs:22`) —
  additive field; single-version cluster, so a minor bump is the clean signal.

**Backoff:** in `fail()` (`coordinator.rs:567`), on retry set
`next_retry_at_ms = now + min(max_delay, base * 2^(attempt-1)) + jitter`.
Constants: `base=100ms`, `max_delay=5_000ms`, jitter
`rand(0, base*2^(attempt-1))`. Re-use `rand_jitter()`
(`crayon_cluster.rs:236`) — move it to a shared spot or replicate.

**Scheduler gating:** `mark_runnable` (`coordinator.rs:133`) stays, but
`assign_next` (`coordinator.rs:390`) must skip tasks whose
`next_retry_at_ms > now`. Add a `now: u64` param to `assign_next`. Tasks past
their deadline are assigned normally. This is the only place `now` flows into
scheduling; the reaper already passes `now` to `expire_workers`.

**State machine change:** a failed-retryable task goes
`Running/Assigned → Runnable` (re-queued) with a future `next_retry_at_ms`,
exactly as today, just delayed. A terminal failure goes `→ Failed(String)` as
today. No new terminal state needed. (Dead-letter queue explicitly deferred:
ephemeral workers + no durability make a replay API pointless; the client
already observes `Failed` via `Status`.)

### 4. File-by-file breakdown
- **`operation.rs:52`** — add `retry_on: Vec<String>` field; `:59` validate
  length/size. Update `builtin_descriptor`/`rl_descriptor`
  (`crayon_cluster.rs:84,97`) callers to pass `vec![]`.
- **`protocol.rs:22`** — `PROTOCOL_MINOR = 1`. `:279`
  `WorkerRequest::Failed { identity, fence, message, error_kind: String }`
  (drop `retryable`).
- **`coordinator.rs:70`** `TaskRecord` — add `next_retry_at_ms: Option<u64>`,
  `retry_on: Vec<String>`. `:284` `submit()` — snapshot `retry_on` from
  descriptor, init `next_retry_at_ms: None`.
- **`coordinator.rs:379`** `assign_next(&mut self, node_id, now: u64)` — in the
  pop loop (`coordinator.rs:390`), skip if
  `task.next_retry_at_ms.is_some_and(|t| t > now)` (re-queue, don't assign).
  Thread `now` from `cluster.rs:311`
  (`state.assign_next(identity.node_id)` → pass `now_ms()`).
- **`coordinator.rs:531`** `fail()` — replace `retryable: bool` param with
  `error_kind: &str`. Compute
  `retry = (is_system || retry_on.contains(error_kind)) && attempt < max_attempts`.
  On retry, set `next_retry_at_ms`. `is_system` is true for the
  coordinator-injected path (lease expiry) — that path already calls
  `mark_runnable` directly (`coordinator.rs:601`), so it bypasses `fail()`;
  only the worker-reported path needs the allow-list check.
- **`cluster.rs:337-345`** — dispatch `WorkerRequest::Failed` passes
  `error_kind` to `fail()`.
- **`crayon_cluster.rs:414-427`** (fetch failure) — report
  `error_kind: "InputFetch"`. `:512-524` (app error) — report
  `error_kind: error.classify()`; add a `fn classify(&self) -> &str` on
  `Error` (`error.rs`) mapping known variants to kind strings, default
  `"Application"`.
- **`client.rs:79,109`** — `submit`/`submit_batch` signatures unchanged;
  `retry_on` lives on the `Operation` descriptor, which the client already
  holds. No client API change.

### 5. Test plan
- **Unit (coordinator.rs tests module):**
  - `app_error_terminal_by_default`: fail with `error_kind="ValueError"`,
    `retry_on=[]` → task `Failed`, not re-queued, `assign_next` returns `None`.
  - `app_error_retry_when_listed`: `retry_on=["TimeoutError"]`, fail with that
    kind → task `Runnable`, attempt incremented, `next_retry_at_ms` set.
  - `retry_backoff_grows_and_caps`: fail 5 times, assert `next_retry_at_ms`
    increases and plateaus at `max_delay`.
  - `scheduler_skips_task_before_retry_deadline`: set
    `next_retry_at_ms = now+1000`, `assign_next(now)` returns `None`;
    `assign_next(now+1000)` returns it.
  - `system_failure_always_retries`: `expire_workers` on a worker with an
    in-flight task → task re-queued regardless of `retry_on`.
- **Integration (tests/cluster_process.rs):** submit a task whose op always
  errors with a non-listed kind → assert exactly 1 attempt. Submit with a
  listed kind + `max_attempts=3` → assert 3 attempts then terminal.

### 6. Definition of done
- A deterministic app error with empty `retry_on` executes exactly once
  (measured: 1 `Started` report, task `Failed`).
- A listed transient error retries with monotonically non-decreasing
  `next_retry_at_ms`, capped at 5s, ≤ `max_attempts` times.
- `cargo test` green; no protocol version skew (minor bump accepted by
  `envelope.validate`).

### 7. Risks / mitigations
- **Retry storms** if a whole op class is listed and a dependency is down →
  bounded by `max_attempts` + per-task backoff; jitter spreads retries.
  Acceptable for ephemeral workload.
- **Worker still sends old `retryable` field** → minor version bump rejects
  old workers cleanly (`Protocol("incompatible protocol")`), they reconnect
  with new code. Single-version cluster, so no mixed-version window.
- **`error_kind` strings are free-form** → typo'd kinds silently fail to
  retry. Mitigation: log the kind on terminal failure so operators see it;
  keep the allow-list on the descriptor so it's versioned with the op.

### 8. Effort
~1 day. Mostly mechanical (field plumbing) + the `assign_next` now-gating +
one `Error::classify` helper.

---

## M5 — Object Locality Scheduling (priority 2)

### 1. Problem statement
`assign_next` (`coordinator.rs:379-435`) is pure FIFO: it pops the front of
`runnable`, checks if *this* polling worker can serve the op + resources, and
assigns. It never considers where the task's input objects live. Yet
`ObjectRecord` already tracks `owner: Option<NodeId>` (`coordinator.rs:90`) —
the coordinator knows exactly which worker holds each large input. Result: a
task whose 100MB input is on worker A is just as likely to run on worker B,
which then fetches the object over the network (data_plane). For RL rollouts
this is the dominant cost; dispatch latency is already ~1ms, so data movement
is the new lever.

### 2. Industry pattern adopted
**Preferred-locations list per task + bounded delay scheduling**
(Spark `TaskLocation` + `spark.locality.wait`, YARN delay scheduling, Ray
node-local preference). The Zaharia result: ~95% locality for <10% of tasks
delayed, with waits of just 1–2 slot-frees. **No replication** (Ray model —
objects immutable, low-value, reconstruct via lineage; replicating RL rollouts
is wasteful). **No multi-level rack relaxation** (speculative generality — no
rack topology exists; per "delete before you add").

Rejected: Ray's full top-k utilization scoring. Crayon workers have
`slots=1` and a single `Mutex`; utilization scoring adds complexity for no
gain when each worker runs one task. Locality preference is the whole story.

### 3. Design

**Preferred workers per task:** when a task enters `Runnable` (in `submit`
`coordinator.rs:363-367`, `mark_runnable` `coordinator.rs:133`, and the retry
path `coordinator.rs:568`), compute `preferred: Vec<NodeId>` = distinct owners
of the task's `Object` args that are still `Alive`. Inline args have no
location. Cheap: args are already on `TaskRecord.args`.

**Bounded delay via skip-round counter:** `TaskRecord` gains
`locality_skips: u32`. In `assign_next`, when the popped task's preferred set
is non-empty and the polling worker is *not* in it *and* at least one
preferred worker is alive with a free slot, increment `locality_skips`,
re-queue the task to the back, and continue scanning. When
`locality_skips >= LOCALITY_MAX_SKIPS` (default 4, ≈ Spark's "wait a few
slot-frees"), or no preferred worker is alive/free, assign to whoever is
polling.

Why a counter and not wall-clock: the long-poll model has no per-task clock;
each `Poll` is one scheduling opportunity. A skip counter bounds delay in
*scheduling rounds*, which maps directly to "wait for N slots to free" —
exactly the Zaharia mechanism. Wall-clock would require a deadline field +
reaper scans; the counter is lower-entropy.

**Tie-breaking when multiple preferred workers free up:** FIFO already picks
the first assignable; if the polling worker is preferred, it gets the task.
No need to score among preferred workers (slots=1, one task per worker).

**No new protocol messages.** Locality is purely a coordinator-internal
scheduling decision. Workers are unaffected.

### 4. File-by-file breakdown
- **`coordinator.rs:70`** `TaskRecord` — add `preferred: Vec<NodeId>`,
  `locality_skips: u32`.
- **`coordinator.rs:133`** `mark_runnable` — after setting state, call a new
  `fn compute_preferred(&self, task) -> Vec<NodeId>` that scans `task.args`
  for `TaskArg::Object`, looks up `self.objects[id].owner`, filters to
  `Alive` workers, dedupes. Store on the task. Reset `locality_skips = 0`.
- **`coordinator.rs:363-367`** `submit` — when setting `Runnable` directly
  (no deps), also compute `preferred` (or route through `mark_runnable` for
  uniformity — refactor the `else` branch at `coordinator.rs:374` to call
  `mark_runnable` instead of `runnable.push_back`).
- **`coordinator.rs:390-405`** `assign_next` pop loop — after the existing
  `worker.operations.contains_key && can_fit` check (`coordinator.rs:395`),
  add the locality gate:
  ```rust
  if !task.preferred.is_empty()
      && !task.preferred.contains(&node_id)
      && task.locality_skips < LOCALITY_MAX_SKIPS
      && self.preferred_worker_free(&task.preferred)
  {
      task.locality_skips += 1;
      requeue.push(id);
      continue;
  }
  ```
  `preferred_worker_free` scans `preferred` for an `Alive` worker with
  `free_slots > 0` that can serve the op (so we don't skip for a preferred
  worker that can't run this op anyway). On assign, `locality_skips` is
  irrelevant (task leaves Runnable).
- **`coordinator.rs`** — add `const LOCALITY_MAX_SKIPS: u32 = 4;` near
  `MAX_TASKS`.
- **`coordinator.rs:607-612`** `expire_workers` — when a worker dies, its
  objects go `Lost` (`coordinator.rs:608`). Tasks that preferred that worker
  should drop it from `preferred` (or recompute). Simplest: recompute
  `preferred` lazily in `assign_next` if any preferred node is no longer in
  `self.workers`. Cheaper than a sweep.

### 5. Test plan
- **Unit:**
  - `prefers_local_worker`: two workers A,B; put object on A (complete a task
    on A so `owner=A`); submit task depending on that object; poll B first →
    B gets `None` (task skipped), poll A → A gets it.
  - `relaxes_after_max_skips`: same setup, only B polls 5 times → 5th poll
    assigns to B (locality exhausted).
  - `no_preferred_runs_anywhere`: task with only inline args → assigned to
    first polling worker, no skipping.
  - `preferred_worker_dead_runs_elsewhere`: owner worker expired → task
    assigned to survivor immediately (no skips).
- **Integration:** two workers, large object produced on A; chain of tasks
  consuming it; assert ≥90% run on A (locality hit rate) under load, with
  <10% of tasks delayed beyond 1 skip-round.

### 6. Definition of done
- A task with inputs on worker A is assigned to A when A has a free slot,
  before any non-preferred worker.
- Locality never deadlocks: after `LOCALITY_MAX_SKIPS` rounds the task runs
  anywhere.
- No regression in dispatch latency for inline-only tasks (still FIFO, ~1ms).

### 7. Risks / mitigations
- **Skip counter causes head-of-line blocking** for the non-preferred worker:
  it keeps getting `None` while preferred tasks skip. Mitigation: the
  non-preferred worker still gets *other* tasks (the loop continues past the
  skipped one). Only tasks with a live preferred worker are skipped;
  inline-only tasks and tasks whose preferred worker is busy-but-alive...
  if the preferred worker is alive but its slot is full, `preferred_worker_free`
  returns false and we don't skip. Good, no blocking on a busy preferred
  worker.
- **Stale `preferred` after owner dies** → lazy recompute in `assign_next`;
  worst case one extra skip round, then it runs elsewhere.
- **`preferred` stored per task adds memory** → `Vec<NodeId>` is small (few
  owners); acceptable.

### 8. Effort
~1 day. The core is ~20 lines in `assign_next` + a `compute_preferred`
helper. Tests are the bulk.

---

## M4 — Graceful Drain (priority 3)

### 1. Problem statement
No SIGTERM handling anywhere. When the orchestrator (K8s) kills a worker
container, the in-flight task is killed mid-execution. The coordinator only
discovers this when the lease expires (`expire_workers`, `cluster.rs:89-100`
reaper, default lease/3 tick). Until then the task sits in `Running`, and on
expiry it retries — but the work done so far is lost, and there's no
propagation delay so the coordinator may assign a *new* task to the dying
worker right up to the kill. For RL rollouts this means a partially-computed
trajectory is discarded and re-run from scratch.

### 2. Industry pattern adopted
**Two-phase shutdown: "unready first, then drain"**
(K8s `preStop`+readiness, Nomad `shutdown_delay`, Envoy
health-fail-first). The long-poll model simplifies this: "stop routing to me"
= "stop issuing new polls / tell coordinator I'm draining." Then finish the
in-flight task (or checkpoint). The coordinator must stop assigning to a
draining worker the instant it declares drain, and re-dispatch the task if
the worker misses the grace deadline.

Rejected: checkpoint/resume. RL rollout checkpointing requires the *task
code* to expose a resumable state, which Crayon's `OperationFn`
(`worker.rs:11`) doesn't support — it's `fn(args) -> Result<bytes>`. Adding
checkpointing is a task-API change, out of scope. **Abandon + re-dispatch**
(Ray's default on actor death) is the pragmatic choice, made less wasteful by
finishing the current task if it fits the grace window.

### 3. Design

**Worker side:**
- Install a `tokio::signal::unix::signal(SIGTERM)` handler in `run_worker`
  (`crayon_cluster.rs:147`). On signal, set an `Arc<AtomicBool>` draining
  flag.
- The poll loop (`crayon_cluster.rs:314`) checks the flag *before* each
  `Poll`: if draining and no task in flight, break out of the loop and exit
  cleanly. If a task is in flight, stop polling for *new* work but let
  `execute_assignment` finish (it already polls only for cancellation with
  `wait_ms:0`, `crayon_cluster.rs:448` — that's fine, it doesn't pull new
  assignments).
- Send a new `WorkerRequest::Drain(identity)` to the coordinator *before*
  stopping polls, so the coordinator stops assigning immediately (the
  "unready" step). This is the propagation-delay substitute — in Crayon the
  coordinator IS the service discovery, so telling it directly replaces the
  K8s endpoint-removal wait.
- Self-terminate the in-flight task at `GRACE_BUDGET_MS` (default 25s, under
  K8s's 30s `terminationGracePeriodSeconds`): wrap `execute_assignment` in a
  `tokio::time::timeout`; on timeout, abort the execution task and report
  `Failed { error_kind: "DrainTimeout" }` (terminal — the worker is dying,
  don't retry on the same worker) then exit. The coordinator will re-dispatch
  on lease expiry if it was retryable, but the worker explicitly releases the
  slot.

**Coordinator side:**
- `WorkerState` (`coordinator.rs:20`) gains `Draining`. `WorkerRecord`
  already has `state`.
- New `WorkerRequest::Drain(WorkerIdentity)` (`protocol.rs`). Handler: set
  worker state to `Draining`, free its slot if idle, wake long-polls.
  `assign_next` (`coordinator.rs:381`) already checks
  `worker.state != WorkerState::Alive` → a `Draining` worker gets `None` from
  `assign_next`. Perfect — no new assignment logic needed, just the state
  check already there.
- A `Draining` worker's in-flight task: if it completes normally, fine. If
  the worker exits without completing (SIGKILL after grace), the lease reaper
  (`expire_workers`) re-queues it as today (`coordinator.rs:600-605`). No
  change needed there — lease expiry already handles "worker gone."
- Heartbeat: a `Draining` worker should still heartbeat so its lease doesn't
  expire mid-drain (which would prematurely re-queue its task). `heartbeat`
  (`coordinator.rs:215`) accepts any `Alive` worker; extend to accept
  `Draining` too (or just check `state != Dead`).

**No protocol version bump needed** if `Drain` is additive to `WorkerRequest`
— but bincode enum variants are positional; adding a variant at the end is
backward-compatible *within the same major*. Since M3 already bumps minor to
1, M4 rides that bump.

### 4. File-by-file breakdown
- **`protocol.rs:257`** `WorkerRequest` — add variant `Drain(WorkerIdentity)`
  at the end.
- **`coordinator.rs:20`** `WorkerState` — add `Draining`. `:215`
  `heartbeat` — accept `Alive | Draining` (change `!= WorkerState::Alive` to
  `== WorkerState::Dead`). `:381` `assign_next` — already returns `None` for
  non-`Alive`; `Draining` is covered. Add
  `pub fn drain_worker(&mut self, identity) -> Result<(), Error>`: check
  identity, set state `Draining`, if `free_slots == slots` (idle) it's
  already not getting work; wake handled by caller. Return `Accepted`.
- **`cluster.rs:295`** Poll dispatch — no change (Draining worker gets
  `None`). Add dispatch for `WorkerRequest::Drain` →
  `state.drain_worker(...)`.
- **`crayon_cluster.rs:147`** `run_worker` — spawn SIGTERM listener setting
  `Arc<AtomicBool>`. Pass the flag into `run_worker_session`. `:314` poll
  loop — at loop top,
  `if draining.load() && !in_flight { send Drain; break; }`. Track
  `in_flight` with a flag set around `execute_assignment`
  (`crayon_cluster.rs:338`).
- **`crayon_cluster.rs:393`** `execute_assignment` — wrap the whole body in
  `tokio::time::timeout(Duration::from_millis(GRACE_BUDGET_MS), ...)`; on
  timeout, abort execution, report
  `Failed { error_kind: "DrainTimeout", .. }`, return. Add
  `const GRACE_BUDGET_MS: u64 = 25_000;`.
- **`crayon_cluster.rs`** — send `WorkerRequest::Drain` via the existing
  `report`/`rpc` path before breaking the poll loop.

### 5. Test plan
- **Unit (coordinator):**
  - `draining_worker_gets_no_assignments`: register worker, set `Draining`,
    `assign_next` returns `None` even with runnable tasks.
  - `draining_worker_heartbeat_accepted`: heartbeat on a `Draining` worker
    refreshes lease (doesn't go `Dead`).
  - `draining_worker_task_requeued_on_expiry`: worker `Draining` with
    in-flight task, then lease expires → task re-queued (or failed per
    `max_attempts`), same as `Alive` expiry path.
- **Integration (cluster_process.rs):** start worker, assign a long `sleep`
  task, send SIGTERM to the worker process → assert (a) no *new* task is
  assigned to it after drain, (b) the in-flight `sleep` task either completes
  (if < grace) or is re-dispatched to another worker after the worker exits.

### 6. Definition of done
- A SIGTERM'd worker stops receiving new assignments within one poll cycle
  (< `WORKER_POLL_WAIT_MS`).
- Its in-flight task completes if it finishes within `GRACE_BUDGET_MS`;
  otherwise the slot is released and the task is re-dispatched on lease
  expiry.
- No task is silently lost — every task either completes or is retried
  elsewhere.

### 7. Risks / mitigations
- **SIGTERM handler not wired in the binary entry point** → `run_worker` is
  the right place; ensure it's called from `main` (`crayon_cluster.rs:33`)
  for the `worker` subcommand.
- **`Drain` RPC itself fails (coordinator unreachable)** → worker should still
  stop polling and exit; the lease reaper will clean up. The `Drain` RPC is
  best-effort; log and continue if it errors.
- **In-flight task ignores abort** (CPU-bound, no `.await`) → cooperative
  abort already has this limitation (`crayon_cluster.rs:469-471` comment).
  The `GRACE_BUDGET_MS` timeout at least lets the worker *exit* (aborting the
  tokio task, which may not stop a tight loop, but the process exits on
  return from `run_worker_session`). Document: hard CPU-bound ops may run to
  SIGKILL; that's the existing contract.
- **`Draining` workers accumulate if they never exit** → they still
  heartbeat, so they don't expire; but a `Draining` worker that stops
  heartbeating will expire normally. Acceptable.

### 8. Effort
~1.5 days. Signal handling + the new `Drain` variant + the grace timeout
wrap + integration test with process signalling.

---

## Cross-cutting notes
- **Order matters:** M3 changes `WorkerRequest::Failed` (drops `retryable`);
  M4's `DrainTimeout` failure uses the new `error_kind` field. Do M3 first.
- **Protocol minor bump to 1** ships M3 + M4's `Drain` variant together. M5
  needs no protocol change.
- **No durability/HA work** — all three milestones respect the
  single-coordinator, ephemeral-worker design point. Exactly-once is
  explicitly out of scope (at-least-once + idempotent tasks, per every
  compared system).
- **Total effort:** ~3.5 days for M3+M5+M4, plus test hardening.
