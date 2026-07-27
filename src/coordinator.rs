//! The cluster control-plane state machine.
//!
//! # Contract
//!
//! `CoordinatorState` is a **single aggregate root**, not three tables. Tasks,
//! objects, and workers share invariants — `complete()` touches all of them at
//! once, `task.output` *is* an `ObjectId`, `task.assigned` references a worker —
//! so they are owned together and mutated only through the methods below.
//! Splitting them into separate stores would scatter the state⇔index invariant
//! across module boundaries; keeping one aggregate is deliberate.
//!
//! Invariants the methods preserve, and callers may rely on:
//! - **Pure and IO-free.** No sockets, no clock beyond an injected `now: u64`,
//!   no logging. Concurrency lives entirely in the caller's single `Mutex`
//!   (`cluster.rs`); this file never blocks, so a transition is always O(bounded).
//! - **State ⇔ index consistency.** `task.state` and its membership in the
//!   `runnable`/`waiting`/`cancel_requested` indices move together, only via the
//!   `mark_*` helpers. Stale index entries are tolerated on read, never written.
//! - **Fencing is authoritative.** Every worker report is checked against
//!   `(node, worker_epoch, session, lease, attempt)`; a stale report cannot
//!   mutate terminal state. Reports are idempotent by fence.
//! - **Bounded.** `MAX_TASKS`/`MAX_OBJECTS` cap the tables; terminal records are
//!   reclaimed on Release or the TTL backstop, so the caps are not lifetime caps.
//!
//! All IO, networking, long-poll wakeups, and persistence live in `cluster.rs`,
//! which wraps this whole struct in one lock. Tests live in `coordinator_tests.rs`.

use crate::{
    error::Error,
    ids::{
        Attempt, CoordinatorEpoch, LeaseId, NodeId, ObjectId, Revision, TaskId, WorkerEpoch,
        WorkerSessionId,
    },
    operation::{Codec, OperationDescriptor, OperationKey, TaskArg},
    protocol::{
        FailureClass, RegisterWorker, TaskAssignment, TaskCompletion, TaskFence, TaskStatus,
        WorkerIdentity,
    },
    resources::ResourceSet,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub const MAX_TASKS: usize = 16_384;
pub const MAX_OBJECTS: usize = 65_536;

/// Exponential retry backoff bounds. A retried task waits `base * 2^(attempt-1)`
/// ms (capped) before it may be re-dispatched, so a transient fault outlasting
/// the instant retry burst does not exhaust `max_attempts`. Cap 30s ≈ 6× the
/// default 5s lease — a backed-off task can intentionally outlive several lease
/// windows.
const RETRY_BACKOFF_BASE_MS: u64 = 100;
const RETRY_BACKOFF_CAP_MS: u64 = 30_000;

/// How long a terminal task's record and output are retained before the server
/// reclaims them, if the client never calls Release. Client-driven Release stays
/// the fast path; this is the backstop so a fire-and-forget or crashed client
/// cannot grow the task table to MAX_TASKS and wedge new submits. Well above any
/// reasonable result-fetch window — a client that wants its output must Get it
/// within this window. NOT distributed refcounting (a non-goal): a single-node
/// retention timer, tick-driven like the lease reaper.
const TERMINAL_TASK_TTL_MS: u64 = 600_000; // 10 min

/// Backoff for the `attempt`-th failure (attempts already made): 1→100ms,
/// 2→200ms, 3→400ms, … capped. The shift is guarded so a runaway attempt count
/// cannot overflow.
fn retry_backoff_ms(attempt: u32) -> u64 {
    RETRY_BACKOFF_BASE_MS
        .saturating_mul(1u64 << attempt.saturating_sub(1).min(20))
        .min(RETRY_BACKOFF_CAP_MS)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum WorkerState {
    Alive,
    /// SIGTERM received: no new assignments, but in-flight work may still
    /// report and the lease keeps renewing until the worker exits.
    Draining,
    Dead,
}
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum TaskState {
    Waiting,
    Runnable,
    Assigned,
    Running,
    CancelRequested,
    Succeeded,
    Failed(String),
    Cancelled,
}
impl From<&TaskState> for TaskStatus {
    fn from(state: &TaskState) -> Self {
        match state {
            TaskState::Waiting => Self::Waiting,
            TaskState::Runnable => Self::Runnable,
            TaskState::Assigned => Self::Assigned,
            TaskState::Running => Self::Running,
            TaskState::CancelRequested => Self::CancelRequested,
            TaskState::Succeeded => Self::Succeeded,
            TaskState::Failed(message) => Self::Failed(message.clone()),
            TaskState::Cancelled => Self::Cancelled,
        }
    }
}
impl TaskState {
    /// Terminal states never transition again; their output is releasable and
    /// the record is reclaimable once no live task references the output.
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed(_) | Self::Cancelled)
    }
}

/// Per-worker tally from a lease-expiry sweep, so the reaper can log a
/// `crayon.worker_dead` line without re-scanning the task/object tables.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct WorkerExpiry {
    pub node: NodeId,
    pub retried: u32,
    pub failed: u32,
    pub cancelled: u32,
    pub objects_lost: u32,
}

/// Result of admitting a client-put inline object. Carries the stored `Arc` and
/// content checksum so the caller reuses them instead of re-hashing/re-cloning.
pub struct PutOutcome {
    pub id: ObjectId,
    pub checksum: [u8; 32],
    pub stored: Arc<[u8]>,
}

/// Aggregate cluster counts for the periodic `crayon.health` log line.
#[derive(Debug, Clone, Copy, Default)]
pub struct HealthCounts {
    pub tasks: usize,
    pub runnable: usize,
    pub running: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub objects: usize,
    pub available: usize,
    pub reserved: usize,
    pub lost: usize,
    pub workers: usize,
    pub retries_total: u64,
    pub failures_total: u64,
}
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum ObjectState {
    Reserved,
    Available,
    Failed,
    Cancelled,
    Lost,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRecord {
    pub identity: WorkerIdentity,
    pub state: WorkerState,
    pub advertise_addr: String,
    pub total: ResourceSet,
    pub available: ResourceSet,
    pub slots: u32,
    pub free_slots: u32,
    pub operations: HashMap<OperationKey, OperationDescriptor>,
    pub lease_deadline_ms: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    pub id: TaskId,
    pub operation: OperationKey,
    pub args: Vec<TaskArg>,
    pub output: ObjectId,
    pub resources: ResourceSet,
    pub max_attempts: u32,
    pub attempt: Attempt,
    pub state: TaskState,
    pub assigned: Option<(NodeId, WorkerEpoch, WorkerSessionId, LeaseId)>,
    /// Earliest wall-clock (ms since epoch) this task may be re-dispatched;
    /// set to `now + backoff` on retry so a transient fault is not hammered.
    /// 0 = immediately runnable. Real state (a retried task keeps its deadline
    /// across a durability snapshot), so not `#[serde(skip)]`.
    pub not_before_ms: u64,
    /// Wall-clock (ms) the task first became terminal, stamped lazily by the
    /// reaper; 0 while non-terminal. Drives server-side reclaim after
    /// `TERMINAL_TASK_TTL_MS` when the client never Releases.
    pub terminal_since_ms: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectRecord {
    pub id: ObjectId,
    pub state: ObjectState,
    pub codec: Option<Codec>,
    // A durability snapshot persists the control-plane graph, not the object
    // payloads (worker-local / re-derivable via lineage), so bytes are skipped —
    // which also keeps the state serde-clean without serde's `rc` feature.
    #[serde(skip)]
    pub bytes: Option<Arc<[u8]>>,
    pub size_bytes: Option<u64>,
    pub checksum: Option<[u8; 32]>,
    pub location: Option<String>,
    pub owner: Option<NodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorState {
    pub epoch: CoordinatorEpoch,
    pub revision: Revision,
    pub workers: HashMap<NodeId, WorkerRecord>,
    pub tasks: HashMap<TaskId, TaskRecord>,
    pub objects: HashMap<ObjectId, ObjectRecord>,
    pub operations: HashMap<OperationKey, OperationDescriptor>,
    pub pending_deletes: HashMap<NodeId, Vec<ObjectId>>,
    /// FIFO of tasks in `Runnable` state, so scheduling picks the next candidate
    /// without scanning every task. May hold stale ids (a task left Runnable via
    /// cancel/retry); `assign_next` skips any entry no longer runnable.
    runnable: VecDeque<TaskId>,
    /// Tasks in `Waiting` state, so dependency reconciliation visits only blocked
    /// tasks instead of the whole table.
    waiting: HashSet<TaskId>,
    /// Tasks in `CancelRequested`, so `cancellation_for` visits only tasks
    /// awaiting a worker ack instead of scanning the whole table.
    cancel_requested: HashSet<TaskId>,
    /// Measurement only (evolution-plan gap 3): how often the locality
    /// look-ahead actually matters. The server's reaper task logs these
    /// periodically; transient, so they never enter serialized state.
    /// ponytail: log-line metrics; promote to a stats RPC if data warrants it.
    #[serde(skip)]
    pub sched_assigns: u64,
    #[serde(skip)]
    pub sched_owned_input: u64,
    #[serde(skip)]
    pub sched_local_hits: u64,
    /// Cumulative retries and permanent failures, for the periodic health log.
    /// Transient (log-line metrics), same as `sched_*`.
    #[serde(skip)]
    pub retries_total: u64,
    #[serde(skip)]
    pub failures_total: u64,
}
impl CoordinatorState {
    pub fn new(epoch: CoordinatorEpoch) -> Self {
        Self {
            epoch,
            revision: Revision(0),
            workers: HashMap::new(),
            tasks: HashMap::new(),
            objects: HashMap::new(),
            operations: HashMap::new(),
            pending_deletes: HashMap::new(),
            runnable: VecDeque::new(),
            waiting: HashSet::new(),
            cancel_requested: HashSet::new(),
            sched_assigns: 0,
            sched_owned_input: 0,
            sched_local_hits: 0,
            retries_total: 0,
            failures_total: 0,
        }
    }
    fn changed(&mut self) {
        self.revision.0 += 1;
    }
    /// Sets `Runnable` and enqueues; the queue may hold stale ids, `assign_next`
    /// skips them on pop.
    fn mark_runnable(&mut self, id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&id) {
            task.state = TaskState::Runnable;
        }
        self.waiting.remove(&id);
        self.runnable.push_back(id);
    }
    /// Sets `CancelRequested` and indexes it for `cancellation_for`.
    fn mark_cancel_requested(&mut self, id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&id) {
            task.state = TaskState::CancelRequested;
        }
        self.cancel_requested.insert(id);
    }
    pub fn register_worker(
        &mut self,
        request: RegisterWorker,
        now: u64,
        lease_ms: u64,
    ) -> Result<WorkerIdentity, Error> {
        crate::protocol::validate_advertise_addr(&request.advertise_addr)?;
        if request.slots == 0 {
            return Err(Error::InvalidResource(
                "worker slots must be positive".into(),
            ));
        }
        let mut operations = HashMap::new();
        for descriptor in &request.operations {
            descriptor.validate()?;
            if let Some(previous) = operations.insert(descriptor.key.clone(), descriptor.clone()) {
                if previous != *descriptor {
                    return Err(Error::OperationConflict(descriptor.key.to_string()));
                }
            }
            if let Some(existing) = self.operations.get(&descriptor.key) {
                if existing != descriptor {
                    return Err(Error::OperationConflict(descriptor.key.to_string()));
                }
            }
        }
        if let Some(existing) = self.workers.get_mut(&request.node_id) {
            // Draining counts as active: overwriting it would orphan its
            // in-flight task's assignment. Let the lease reaper clear it first.
            if existing.state != WorkerState::Dead {
                let same = existing.identity.worker_epoch == request.worker_epoch
                    && existing.advertise_addr == request.advertise_addr
                    && existing.total == request.resources
                    && existing.slots == request.slots
                    && existing.operations == operations;
                if !same {
                    return Err(Error::Protocol("duplicate active node id".into()));
                }
                existing.lease_deadline_ms = now.saturating_add(lease_ms);
                let identity = existing.identity.clone();
                self.changed();
                return Ok(identity);
            }
        }
        let identity = WorkerIdentity {
            node_id: request.node_id,
            worker_epoch: request.worker_epoch,
            session_id: WorkerSessionId::new(),
            coordinator_epoch: self.epoch,
        };
        for (key, descriptor) in &operations {
            self.operations.insert(key.clone(), descriptor.clone());
        }
        self.workers.insert(
            request.node_id,
            WorkerRecord {
                identity: identity.clone(),
                state: WorkerState::Alive,
                advertise_addr: request.advertise_addr,
                total: request.resources.clone(),
                available: request.resources,
                slots: request.slots,
                free_slots: request.slots,
                operations,
                lease_deadline_ms: now.saturating_add(lease_ms),
            },
        );
        self.changed();
        Ok(identity)
    }
    pub fn heartbeat(
        &mut self,
        identity: &WorkerIdentity,
        now: u64,
        lease_ms: u64,
    ) -> Result<(), Error> {
        self.check_identity(identity)?;
        self.workers
            .get_mut(&identity.node_id)
            .unwrap()
            .lease_deadline_ms = now.saturating_add(lease_ms);
        Ok(())
    }
    /// Admits a client-put inline object, returning its id, content checksum,
    /// and the stored `Arc` so the caller can reuse both without re-hashing or
    /// re-cloning the payload.
    pub fn put(&mut self, codec: Codec, bytes: Vec<u8>) -> Result<PutOutcome, Error> {
        let checksum = crate::cluster::checksum(&bytes);
        let id = ObjectId::from_checksum(checksum);
        let size_bytes = bytes.len() as u64;
        let stored: Arc<[u8]> = bytes.into();
        self.admit_object(id, codec, size_bytes, checksum, Some(stored.clone()))?;
        Ok(PutOutcome {
            id,
            checksum,
            stored,
        })
    }
    /// Registers a client-put object whose bytes live in the shared-memory
    /// arena: the record carries metadata only, and readers are handed an
    /// arena offset instead of inline bytes.
    pub fn put_meta(
        &mut self,
        id: ObjectId,
        codec: Codec,
        size_bytes: u64,
        checksum: [u8; 32],
    ) -> Result<ObjectId, Error> {
        self.admit_object(id, codec, size_bytes, checksum, None)
    }
    fn admit_object(
        &mut self,
        id: ObjectId,
        codec: Codec,
        size_bytes: u64,
        checksum: [u8; 32],
        bytes: Option<Arc<[u8]>>,
    ) -> Result<ObjectId, Error> {
        if let Some(existing) = self.objects.get(&id) {
            if existing.state != ObjectState::Available
                || existing.codec.as_ref() != Some(&codec)
                || existing.size_bytes != Some(size_bytes)
                || existing.checksum != Some(checksum)
            {
                // A content-derived id already proves matching content; random
                // (unhashed) ids never repeat. Either way metadata equality is
                // the whole check -- no full-payload compare.
                return Err(Error::ObjectConflict(id));
            }
            return Ok(id);
        }
        if self.objects.len() >= MAX_OBJECTS {
            return Err(Error::CapacityExceeded("object store limit reached".into()));
        }
        self.objects.insert(
            id,
            ObjectRecord {
                id,
                state: ObjectState::Available,
                codec: Some(codec),
                size_bytes: Some(size_bytes),
                checksum: Some(checksum),
                bytes,
                location: Some("coordinator".into()),
                owner: None,
            },
        );
        self.changed();
        Ok(id)
    }
    /// Resolves an object to its payload or the error it currently maps to.
    /// A reserved output of a still-running task is `ObjectPending` so a blocking
    /// `Get`/`GetBatch` parks; a `Lost` object is `ObjectLost`; anything else
    /// unavailable is a protocol error. Shared by single and batch fetch.
    pub fn resolve_object(&self, id: ObjectId) -> Result<crate::protocol::ObjectPayload, Error> {
        match self.objects.get(&id) {
            Some(object) if object.state == ObjectState::Available => {
                Ok(crate::protocol::ObjectPayload {
                    id,
                    codec: object.codec.clone().unwrap(),
                    size_bytes: object.size_bytes.unwrap(),
                    checksum: object.checksum.unwrap(),
                    location: object.location.clone().unwrap(),
                    bytes: object.bytes.clone(),
                    arena: None,
                })
            }
            Some(object) if object.state == ObjectState::Lost => Err(Error::ObjectLost(id)),
            Some(object) if object.state == ObjectState::Reserved => Err(Error::ObjectPending(id)),
            _ => Err(Error::Protocol("object unavailable".into())),
        }
    }
    pub fn submit(
        &mut self,
        operation: OperationKey,
        args: Vec<TaskArg>,
        resources: ResourceSet,
        max_attempts: u32,
    ) -> Result<(TaskId, ObjectId), Error> {
        if max_attempts == 0 {
            return Err(Error::IllegalTransition(
                "max_attempts must be positive".into(),
            ));
        }
        if self.tasks.len() >= MAX_TASKS {
            return Err(Error::CapacityExceeded("task limit reached".into()));
        }
        // Admission gate mirroring `put`: submitting reserves an output object, so
        // it must respect the same object-store cap or the store can exceed it.
        if self.objects.len() >= MAX_OBJECTS {
            return Err(Error::CapacityExceeded("object store limit reached".into()));
        }
        let descriptor = self
            .operations
            .get(&operation)
            .cloned()
            .ok_or_else(|| Error::OperationUnavailable(operation.to_string()))?;
        descriptor.validate_args(&args)?;
        if !self.workers.values().any(|worker| {
            worker.state == WorkerState::Alive
                && worker.operations.contains_key(&operation)
                && worker.total.can_fit(&resources)
        }) {
            return Err(Error::InvalidResource(
                "no registered worker can satisfy task resources".into(),
            ));
        }
        let mut waiting = false;
        for arg in &args {
            if let TaskArg::Object(id) = arg {
                let object = self.objects.get(id).ok_or(Error::ObjectNotFound(*id))?;
                if object
                    .codec
                    .as_ref()
                    .is_some_and(|codec| codec != &descriptor.input_codec)
                {
                    return Err(Error::Protocol("object argument codec mismatch".into()));
                }
                match object.state {
                    ObjectState::Available => {}
                    ObjectState::Reserved => waiting = true,
                    _ => return Err(Error::DependencyFailed(*id)),
                }
            }
        }
        let id = TaskId::new();
        let output = ObjectId::new();
        self.objects.insert(
            output,
            ObjectRecord {
                id: output,
                state: ObjectState::Reserved,
                codec: Some(descriptor.output_codec.clone()),
                bytes: None,
                size_bytes: None,
                checksum: None,
                location: None,
                owner: None,
            },
        );
        self.tasks.insert(
            id,
            TaskRecord {
                id,
                operation,
                args,
                output,
                resources,
                max_attempts,
                attempt: Attempt(0),
                state: if waiting {
                    TaskState::Waiting
                } else {
                    TaskState::Runnable
                },
                assigned: None,
                not_before_ms: 0,
                terminal_since_ms: 0,
            },
        );
        if waiting {
            self.waiting.insert(id);
        } else {
            self.runnable.push_back(id);
        }
        self.changed();
        Ok((id, output))
    }
    pub fn assigned_unstarted(
        &self,
        identity: &WorkerIdentity,
    ) -> Result<Option<TaskAssignment>, Error> {
        self.check_identity(identity)?;
        Ok(self.tasks.values().find_map(|task| {
            let (node, epoch, session, lease) = task.assigned?;
            (task.state == TaskState::Assigned
                && node == identity.node_id
                && epoch == identity.worker_epoch
                && session == identity.session_id)
                .then(|| TaskAssignment {
                    fence: TaskFence {
                        task_id: task.id,
                        attempt: task.attempt,
                        lease_id: lease,
                    },
                    operation: task.operation.clone(),
                    args: task.args.clone(),
                    output_id: task.output,
                    resources: task.resources.clone(),
                })
        }))
    }
    pub fn assign_next(
        &mut self,
        node_id: NodeId,
        now: u64,
    ) -> Result<Option<TaskAssignment>, Error> {
        let worker = self.workers.get(&node_id).ok_or(Error::StaleFence)?;
        if worker.state != WorkerState::Alive || worker.free_slots == 0 {
            return Ok(None);
        }
        // Pop runnable ids, skipping stale entries and re-queuing tasks this
        // worker can't serve. Bounded by the runnable count, not the task table.
        let (workers, runnable, tasks, objects) = (
            &self.workers,
            &mut self.runnable,
            &self.tasks,
            &self.objects,
        );
        let worker = &workers[&node_id];
        // Prefer a runnable task whose input objects this worker already owns, so a
        // downstream task runs where its inputs are local instead of refetching them
        // across the network. Bounded look-ahead keeps the hot path cheap; with no
        // local match it falls back to the first servable task in FIFO order.
        const LOCALITY_LOOKAHEAD: usize = 16;
        let mut requeue: Vec<TaskId> = Vec::new(); // not servable by this worker
        let mut passed: Vec<TaskId> = Vec::new(); // servable, not local; keep FIFO priority
        let mut chosen: Option<(TaskId, bool)> = None; // (task, picked via local branch)
        let mut looked = 0;
        while let Some(id) = runnable.pop_front() {
            let Some(task) = tasks.get(&id) else { continue };
            if task.state != TaskState::Runnable {
                continue;
            }
            // Retried task not yet due: keep it queued (back) and skip. Pickup is
            // bounded by the worker Poll long-poll's 250ms re-dispatch slice.
            // ponytail: no dedicated backoff timer; 250ms slice bounds the wait.
            if task.not_before_ms > now {
                requeue.push(id);
                continue;
            }
            if !(worker.operations.contains_key(&task.operation)
                && worker.available.can_fit(&task.resources))
            {
                requeue.push(id);
                continue;
            }
            let local = task.args.iter().any(|arg| {
                matches!(arg, TaskArg::Object(oid)
                    if objects.get(oid).and_then(|object| object.owner) == Some(node_id))
            });
            if local {
                chosen = Some((id, true));
                break;
            }
            passed.push(id);
            looked += 1;
            if looked >= LOCALITY_LOOKAHEAD {
                break;
            }
        }
        if chosen.is_none() {
            chosen = passed.first().map(|&id| (id, false));
        }
        // Servable-but-passed-over tasks go back to the front (FIFO priority kept);
        // tasks this worker cannot serve go to the back, as before.
        for id in passed.into_iter().rev() {
            if chosen.map(|(picked, _)| picked) != Some(id) {
                self.runnable.push_front(id);
            }
        }
        for id in requeue {
            self.runnable.push_back(id);
        }
        let Some((id, local_hit)) = chosen else {
            return Ok(None);
        };
        // Locality measurement: local_hit is the selection loop's own verdict;
        // owned_input asks whether locality was even possible for this task.
        let owned_input = local_hit
            || self.tasks[&id].args.iter().any(|arg| {
                matches!(arg, TaskArg::Object(oid)
                    if self.objects.get(oid).and_then(|object| object.owner).is_some())
            });
        self.sched_assigns += 1;
        self.sched_owned_input += owned_input as u64;
        self.sched_local_hits += local_hit as u64;
        let attempt = Attempt(self.tasks[&id].attempt.0 + 1);
        let lease = LeaseId::new();
        let resources = self.tasks[&id].resources.clone();
        let worker = self.workers.get_mut(&node_id).unwrap();
        worker.available.subtract(&resources)?;
        worker.free_slots -= 1;
        let task = self.tasks.get_mut(&id).unwrap();
        task.attempt = attempt;
        task.state = TaskState::Assigned;
        task.assigned = Some((
            node_id,
            worker.identity.worker_epoch,
            worker.identity.session_id,
            lease,
        ));
        let assignment = TaskAssignment {
            fence: TaskFence {
                task_id: id,
                attempt,
                lease_id: lease,
            },
            operation: task.operation.clone(),
            args: task.args.clone(),
            output_id: task.output,
            resources,
        };
        self.changed();
        Ok(Some(assignment))
    }
    pub fn started(&mut self, identity: &WorkerIdentity, fence: TaskFence) -> Result<(), Error> {
        // Idempotent: a duplicate started report for an already-running task is
        // accepted so workers can safely retry reports. Look up first so an
        // unknown task_id yields TaskNotFound instead of panicking on index.
        let task = self
            .tasks
            .get(&fence.task_id)
            .ok_or(Error::TaskNotFound(fence.task_id))?;
        if task.state == TaskState::Running {
            return Ok(());
        }
        self.check_fence(identity, fence)?;
        let task = self.tasks.get_mut(&fence.task_id).unwrap();
        if task.state != TaskState::Assigned {
            return Err(Error::IllegalTransition("task not assigned".into()));
        }
        task.state = TaskState::Running;
        self.changed();
        Ok(())
    }
    pub fn complete(
        &mut self,
        identity: &WorkerIdentity,
        mut report: TaskCompletion,
    ) -> Result<(), Error> {
        // Look up first so an unknown task_id yields TaskNotFound instead of
        // panicking on index.
        let task = self
            .tasks
            .get(&report.fence.task_id)
            .ok_or(Error::TaskNotFound(report.fence.task_id))?;
        // Idempotent: a duplicate completion for an already-succeeded task with
        // a matching output is accepted so workers can safely retry reports. The
        // output may have been released (GC) after success, so a retried report
        // must not panic when the object is gone.
        if task.state == TaskState::Succeeded {
            return match self.objects.get(&task.output) {
                Some(object)
                    if object.state == ObjectState::Available
                        && object.checksum == Some(report.checksum)
                        && object.size_bytes == Some(report.size_bytes) =>
                {
                    Ok(())
                }
                None => Ok(()),
                _ => Err(Error::ObjectConflict(task.output)),
            };
        }
        self.check_fence(identity, report.fence)?;
        let task = &self.tasks[&report.fence.task_id];
        if !matches!(task.state, TaskState::Assigned | TaskState::Running) {
            return Err(Error::IllegalTransition(
                "task cannot complete in current state".into(),
            ));
        }
        let worker = &self.workers[&identity.node_id];
        let descriptor = self
            .operations
            .get(&task.operation)
            .ok_or_else(|| Error::OperationUnavailable(task.operation.to_string()))?;
        // Inline bytes, when shipped, must match the reported size and checksum;
        // a mismatch means the worker sent inconsistent data.
        if let Some(bytes) = &report.bytes {
            if bytes.len() as u64 != report.size_bytes
                || crate::cluster::checksum(bytes) != report.checksum
            {
                return Err(Error::ObjectConflict(task.output));
            }
        }
        if task.output != report.output_id
            || self.objects[&task.output].state != ObjectState::Reserved
            || report.codec != descriptor.output_codec
            || report.location != worker.advertise_addr
            || report.size_bytes > crate::protocol::MAX_OBJECT_BYTES as u64
        {
            return Err(Error::ObjectConflict(task.output));
        }
        let output = task.output;
        let resources = task.resources.clone();
        self.release(identity.node_id, &resources)?;
        let task = self.tasks.get_mut(&report.fence.task_id).unwrap();
        task.state = TaskState::Succeeded;
        task.assigned = None;
        let object = self.objects.get_mut(&output).unwrap();
        object.state = ObjectState::Available;
        object.codec = Some(report.codec);
        object.size_bytes = Some(report.size_bytes);
        object.checksum = Some(report.checksum);
        object.location = Some(report.location);
        object.owner = Some(identity.node_id);
        object.bytes = report.bytes.take();
        self.reconcile_dependencies();
        self.changed();
        Ok(())
    }
    /// Records a worker's failure report. Returns `true` if this made the task
    /// permanently `Failed` (the caller logs a `crayon.task_failed`); `false` if
    /// it was retried, or was a duplicate report for an already-terminal task.
    pub fn fail(
        &mut self,
        identity: &WorkerIdentity,
        fence: TaskFence,
        message: String,
        class: FailureClass,
        now: u64,
    ) -> Result<bool, Error> {
        // Idempotent: a duplicate failure report for an already-terminal task
        // is accepted so workers can safely retry reports. Look up first so an
        // unknown task_id yields TaskNotFound instead of panicking on index.
        let task = self
            .tasks
            .get(&fence.task_id)
            .ok_or(Error::TaskNotFound(fence.task_id))?;
        if task.state.is_terminal() {
            return Ok(false);
        }
        self.check_fence(identity, fence)?;
        if !matches!(
            self.tasks[&fence.task_id].state,
            TaskState::Assigned | TaskState::Running
        ) {
            return Err(Error::IllegalTransition(
                "task cannot fail in current state".into(),
            ));
        }
        let resources = self.tasks[&fence.task_id].resources.clone();
        // Coordinator-owned retry policy: only Transient failures retry, and only
        // while attempts remain. A Permanent failure (e.g. an operation panic) is
        // terminal even if attempts are left, so a crashing op is not retried.
        let attempt = self.tasks[&fence.task_id].attempt.0;
        let retry = matches!(class, FailureClass::Transient)
            && attempt < self.tasks[&fence.task_id].max_attempts;
        let output = self.tasks[&fence.task_id].output;
        self.release(identity.node_id, &resources)?;
        self.tasks.get_mut(&fence.task_id).unwrap().assigned = None;
        let permanently_failed = if retry {
            self.mark_runnable(fence.task_id);
            self.tasks.get_mut(&fence.task_id).unwrap().not_before_ms =
                now.saturating_add(retry_backoff_ms(attempt));
            self.retries_total += 1;
            false
        } else {
            self.tasks.get_mut(&fence.task_id).unwrap().state = TaskState::Failed(message);
            self.objects.get_mut(&output).unwrap().state = ObjectState::Failed;
            self.reconcile_dependencies();
            self.failures_total += 1;
            true
        };
        self.changed();
        Ok(permanently_failed)
    }
    pub fn expire_workers(&mut self, now: u64) -> Result<Vec<WorkerExpiry>, Error> {
        let expired: Vec<_> = self
            .workers
            .values()
            .filter(|worker| worker.state != WorkerState::Dead && worker.lease_deadline_ms <= now)
            .map(|worker| worker.identity.node_id)
            .collect();
        let mut report = Vec::with_capacity(expired.len());
        for node in &expired {
            self.workers.get_mut(node).unwrap().state = WorkerState::Dead;
            let mut tally = WorkerExpiry {
                node: *node,
                retried: 0,
                failed: 0,
                cancelled: 0,
                objects_lost: 0,
            };
            let tasks: Vec<_> = self
                .tasks
                .values()
                .filter(|task| task.assigned.is_some_and(|assigned| assigned.0 == *node))
                .map(|task| task.id)
                .collect();
            for id in tasks {
                let task = self.tasks.get_mut(&id).unwrap();
                task.assigned = None;
                let output = task.output;
                if task.state == TaskState::CancelRequested {
                    task.state = TaskState::Cancelled;
                    self.cancel_requested.remove(&id);
                    self.objects.get_mut(&output).unwrap().state = ObjectState::Cancelled;
                    tally.cancelled += 1;
                } else if task.attempt.0 < task.max_attempts {
                    let attempt = task.attempt.0;
                    self.mark_runnable(id);
                    self.tasks.get_mut(&id).unwrap().not_before_ms =
                        now.saturating_add(retry_backoff_ms(attempt));
                    self.retries_total += 1;
                    tally.retried += 1;
                } else {
                    task.state = TaskState::Failed("worker lease expired".into());
                    self.objects.get_mut(&output).unwrap().state = ObjectState::Failed;
                    // A lease-expiry permanent failure is reported as part of this
                    // worker-death event (crayon.worker_dead failed=N), not a
                    // separate per-task crayon.task_failed line.
                    // ponytail: rollup, not per-task; expire_workers returns a batch,
                    // threading a per-task outcome out of it would be higher-entropy.
                    self.failures_total += 1;
                    tally.failed += 1;
                }
            }
            for object in self.objects.values_mut() {
                if object.owner == Some(*node) && object.state == ObjectState::Available {
                    object.owner = None;
                    object.location = object.bytes.is_some().then(|| "coordinator".into());
                    if object.bytes.is_none() {
                        object.state = ObjectState::Lost;
                        tally.objects_lost += 1;
                    }
                }
            }
            // Ephemeral workers die on completion and never return, so drop the
            // dead record and its pending deletes instead of leaking them forever.
            // Fencing-safe: a late request from this node hits check_identity ->
            // workers.get -> None -> StaleFence, exactly as when it was left Dead.
            self.workers.remove(node);
            self.pending_deletes.remove(node);
            report.push(tally);
        }
        if !expired.is_empty() {
            self.reconcile_dependencies();
            self.changed();
        }
        Ok(report)
    }
    /// Server-side backstop for terminal tasks the client never Releases: stamp
    /// each terminal task's first-seen time lazily, then reclaim (via the same
    /// `release_object` path a client Release uses) once it has been terminal for
    /// `TERMINAL_TASK_TTL_MS` and its output is no longer referenced by a live
    /// task. Returns how many records were reclaimed. Driven from the reaper tick;
    /// O(tasks) — one pass to stamp, one to collect the still-referenced outputs,
    /// one to collect the due ones.
    pub fn reap_terminal_tasks(&mut self, now: u64) -> usize {
        // Stamp newly-terminal tasks and, in the same pass, gather the outputs any
        // live (non-terminal) task still depends on — a referenced output blocks
        // reclaim of its producer, exactly the client-Release precondition.
        let mut referenced: HashSet<ObjectId> = HashSet::new();
        for task in self.tasks.values_mut() {
            if task.state.is_terminal() {
                if task.terminal_since_ms == 0 {
                    task.terminal_since_ms = now;
                }
            } else {
                for arg in &task.args {
                    if let TaskArg::Object(oid) = arg {
                        referenced.insert(*oid);
                    }
                }
            }
        }
        let due: Vec<ObjectId> = self
            .tasks
            .values()
            .filter(|task| {
                task.state.is_terminal()
                    && now.saturating_sub(task.terminal_since_ms) >= TERMINAL_TASK_TTL_MS
                    && !referenced.contains(&task.output)
            })
            .map(|task| task.output)
            .collect();
        // release_object removes the object and retains-out its producing task.
        let mut reclaimed = 0;
        for output in due {
            if self.release_object(output).is_ok() {
                reclaimed += 1;
            }
        }
        reclaimed
    }
    /// One pass over the task/object tables for the periodic health log. Bounded
    /// by MAX_TASKS/MAX_OBJECTS; called at the reaper's ~30s cadence, not a hot
    /// path — cheaper than maintaining per-state live counters.
    pub fn health_counts(&self) -> HealthCounts {
        let mut h = HealthCounts {
            tasks: self.tasks.len(),
            objects: self.objects.len(),
            workers: self.workers.len(),
            retries_total: self.retries_total,
            failures_total: self.failures_total,
            ..Default::default()
        };
        for task in self.tasks.values() {
            match task.state {
                TaskState::Runnable => h.runnable += 1,
                TaskState::Running | TaskState::Assigned => h.running += 1,
                TaskState::Succeeded => h.succeeded += 1,
                TaskState::Failed(_) => h.failed += 1,
                _ => {}
            }
        }
        for object in self.objects.values() {
            match object.state {
                ObjectState::Available => h.available += 1,
                ObjectState::Reserved => h.reserved += 1,
                ObjectState::Lost => h.lost += 1,
                _ => {}
            }
        }
        h
    }
    pub fn drain(&mut self, identity: &WorkerIdentity) -> Result<(), Error> {
        self.check_identity(identity)?;
        self.workers.get_mut(&identity.node_id).unwrap().state = WorkerState::Draining;
        self.changed();
        Ok(())
    }
    pub fn cancellation_for(&self, identity: &WorkerIdentity) -> Result<Option<TaskFence>, Error> {
        self.check_identity(identity)?;
        Ok(self.cancel_requested.iter().find_map(|id| {
            let task = self.tasks.get(id)?;
            let (node, epoch, session, lease_id) = task.assigned?;
            (node == identity.node_id
                && epoch == identity.worker_epoch
                && session == identity.session_id)
                .then_some(TaskFence {
                    task_id: task.id,
                    attempt: task.attempt,
                    lease_id,
                })
        }))
    }
    pub fn acknowledge_cancel(
        &mut self,
        identity: &WorkerIdentity,
        fence: TaskFence,
    ) -> Result<(), Error> {
        // Idempotent: a duplicate cancel ack for an already-cancelled task is
        // accepted so workers can safely retry reports. Look up first so an
        // unknown task_id yields TaskNotFound instead of panicking on index.
        let task = self
            .tasks
            .get(&fence.task_id)
            .ok_or(Error::TaskNotFound(fence.task_id))?;
        if task.state == TaskState::Cancelled {
            return Ok(());
        }
        self.check_fence(identity, fence)?;
        if self.tasks[&fence.task_id].state != TaskState::CancelRequested {
            return Err(Error::IllegalTransition(
                "task is not awaiting cancellation".into(),
            ));
        }
        let resources = self.tasks[&fence.task_id].resources.clone();
        let output = self.tasks[&fence.task_id].output;
        self.release(identity.node_id, &resources)?;
        self.cancel_requested.remove(&fence.task_id);
        let task = self.tasks.get_mut(&fence.task_id).unwrap();
        task.state = TaskState::Cancelled;
        task.assigned = None;
        self.objects.get_mut(&output).unwrap().state = ObjectState::Cancelled;
        self.reconcile_dependencies();
        self.changed();
        Ok(())
    }
    pub fn cancel(&mut self, id: TaskId) -> Result<(), Error> {
        let task = self.tasks.get(&id).ok_or(Error::TaskNotFound(id))?;
        if task.state.is_terminal() {
            return Err(Error::IllegalTransition("task is already terminal".into()));
        }
        if task.state == TaskState::CancelRequested {
            return Ok(());
        }
        let output = task.output;
        if task.assigned.is_some() {
            self.mark_cancel_requested(id);
        } else {
            self.tasks.get_mut(&id).unwrap().state = TaskState::Cancelled;
            self.waiting.remove(&id);
            self.objects.get_mut(&output).unwrap().state = ObjectState::Cancelled;
            self.reconcile_dependencies();
        }
        self.changed();
        Ok(())
    }
    pub fn release_object(&mut self, id: ObjectId) -> Result<(), Error> {
        let object = match self.objects.get(&id) {
            Some(object) => object.clone(),
            None => return Ok(()),
        };
        // A Reserved object is always the pending output of a still-live task;
        // deleting it would make every terminal transition of that task panic on
        // a missing object. Reject until the producer is terminal, at which point
        // the output is Available/Failed/Cancelled/Lost and release proceeds.
        if object.state == ObjectState::Reserved {
            return Err(Error::ObjectInUse(id));
        }
        if self.object_in_use(&id) {
            return Err(Error::ObjectInUse(id));
        }
        // Only enqueue a delete for an owner that still exists, so a released
        // object owned by an already-evicted worker does not re-leak.
        if let Some(owner) = object.owner {
            if self.workers.contains_key(&owner) {
                self.pending_deletes.entry(owner).or_default().push(id);
            }
        }
        self.objects.remove(&id);
        // A releasable (non-Reserved) output implies its producer is terminal, and
        // object_in_use guarantees no live consumer references it, so drop the
        // producing task too — otherwise the task table is never reclaimed and
        // MAX_TASKS becomes a lifetime cap.
        // ponytail: O(tasks) retain, release is client-driven cleanup not a hot
        // path; add an output->task_id index only if release ever gets hot.
        self.tasks.retain(|_, task| task.output != id);
        self.changed();
        Ok(())
    }
    pub fn take_pending_deletes(&mut self, node: NodeId) -> Vec<ObjectId> {
        self.pending_deletes.remove(&node).unwrap_or_default()
    }
    fn object_in_use(&self, id: &ObjectId) -> bool {
        self.tasks.values().any(|task| {
            !task.state.is_terminal()
                && task.args.iter().any(|arg| match arg {
                    TaskArg::Object(dep) => dep == id,
                    _ => false,
                })
        })
    }
    fn reconcile_dependencies(&mut self) {
        loop {
            let mut changed = false;
            // Visit only blocked tasks, not the whole table.
            let waiting: Vec<_> = self.waiting.iter().copied().collect();
            for id in waiting {
                let dependencies: Vec<_> = self.tasks[&id]
                    .args
                    .iter()
                    .filter_map(|arg| match arg {
                        TaskArg::Object(id) => Some(*id),
                        _ => None,
                    })
                    .collect();
                let failed = dependencies.iter().any(|object| {
                    self.objects.get(object).is_none_or(|record| {
                        !matches!(record.state, ObjectState::Reserved | ObjectState::Available)
                    })
                });
                if failed {
                    let output = self.tasks[&id].output;
                    self.tasks.get_mut(&id).unwrap().state =
                        TaskState::Failed("dependency failed".into());
                    self.waiting.remove(&id);
                    self.objects.get_mut(&output).unwrap().state = ObjectState::Failed;
                    changed = true;
                } else if dependencies.iter().all(|object| {
                    self.objects
                        .get(object)
                        .is_some_and(|record| record.state == ObjectState::Available)
                }) {
                    self.mark_runnable(id);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
    fn check_identity(&self, identity: &WorkerIdentity) -> Result<(), Error> {
        if identity.coordinator_epoch != self.epoch {
            return Err(Error::StaleFence);
        }
        let worker = self
            .workers
            .get(&identity.node_id)
            .ok_or(Error::StaleFence)?;
        // Draining workers must still pass: their in-flight task reports
        // Completed/Failed after the drain request. Only Dead is fenced out.
        if worker.state == WorkerState::Dead
            || worker.identity.worker_epoch != identity.worker_epoch
            || worker.identity.session_id != identity.session_id
        {
            return Err(Error::StaleFence);
        }
        Ok(())
    }
    fn check_fence(&self, identity: &WorkerIdentity, fence: TaskFence) -> Result<(), Error> {
        self.check_identity(identity)?;
        let task = self
            .tasks
            .get(&fence.task_id)
            .ok_or(Error::TaskNotFound(fence.task_id))?;
        if task.assigned
            != Some((
                identity.node_id,
                identity.worker_epoch,
                identity.session_id,
                fence.lease_id,
            ))
            || task.attempt != fence.attempt
        {
            return Err(Error::StaleFence);
        }
        Ok(())
    }
    fn release(&mut self, node: NodeId, resources: &ResourceSet) -> Result<(), Error> {
        let worker = self.workers.get_mut(&node).ok_or(Error::StaleFence)?;
        if worker.free_slots >= worker.slots {
            return Err(Error::InvalidResource(
                "resource slot release exceeds worker total".into(),
            ));
        }
        let mut available = worker.available.clone();
        available.add_capped(resources, &worker.total)?;
        worker.available = available;
        worker.free_slots += 1;
        Ok(())
    }
}

#[cfg(test)]
#[path = "coordinator_tests.rs"]
mod tests;
