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

/// What `fail()` did with a worker's failure report, so the IO-capable caller
/// can log it without re-inspecting state.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum FailOutcome {
    /// Transient failure with attempts left: re-queued with backoff.
    Retried,
    /// Permanent, or attempts exhausted: terminal Failed.
    Failed,
    /// Duplicate report for an already-terminal task; nothing changed.
    Duplicate,
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
    pub fn put(&mut self, codec: Codec, bytes: Vec<u8>) -> Result<ObjectId, Error> {
        let checksum = crate::cluster::checksum(&bytes);
        let id = ObjectId::from_checksum(checksum);
        let size_bytes = bytes.len() as u64;
        self.admit_object(id, codec, size_bytes, checksum, Some(bytes.into()))
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
    pub fn fail(
        &mut self,
        identity: &WorkerIdentity,
        fence: TaskFence,
        message: String,
        class: FailureClass,
        now: u64,
    ) -> Result<FailOutcome, Error> {
        // Idempotent: a duplicate failure report for an already-terminal task
        // is accepted so workers can safely retry reports. Look up first so an
        // unknown task_id yields TaskNotFound instead of panicking on index.
        let task = self
            .tasks
            .get(&fence.task_id)
            .ok_or(Error::TaskNotFound(fence.task_id))?;
        if matches!(
            task.state,
            TaskState::Failed(_) | TaskState::Cancelled | TaskState::Succeeded
        ) {
            return Ok(FailOutcome::Duplicate);
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
        let outcome = if retry {
            self.mark_runnable(fence.task_id);
            self.tasks.get_mut(&fence.task_id).unwrap().not_before_ms =
                now.saturating_add(retry_backoff_ms(attempt));
            self.retries_total += 1;
            FailOutcome::Retried
        } else {
            self.tasks.get_mut(&fence.task_id).unwrap().state = TaskState::Failed(message);
            self.objects.get_mut(&output).unwrap().state = ObjectState::Failed;
            self.reconcile_dependencies();
            self.failures_total += 1;
            FailOutcome::Failed
        };
        self.changed();
        Ok(outcome)
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
                    object.state = ObjectState::Lost;
                    object.location = None;
                    tally.objects_lost += 1;
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
        match task.state {
            TaskState::Succeeded | TaskState::Failed(_) | TaskState::Cancelled => {
                return Err(Error::IllegalTransition("task is already terminal".into()))
            }
            TaskState::CancelRequested => return Ok(()),
            _ => {}
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
            !matches!(
                task.state,
                TaskState::Succeeded | TaskState::Failed(_) | TaskState::Cancelled
            ) && task.args.iter().any(|arg| match arg {
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
mod tests {
    use super::*;

    fn descriptor() -> OperationDescriptor {
        OperationDescriptor {
            key: OperationKey::new("test", "copy", 1),
            input_codec: Codec::RawBytes,
            output_codec: Codec::RawBytes,
            max_inline_arg_bytes: 8,
        }
    }
    fn register(state: &mut CoordinatorState, node: NodeId) -> WorkerIdentity {
        state
            .register_worker(
                RegisterWorker {
                    node_id: node,
                    worker_epoch: WorkerEpoch::new(),
                    advertise_addr: "127.0.0.1:9001".into(),
                    resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                    slots: 1,
                    operations: vec![descriptor()],
                },
                0,
                100,
            )
            .unwrap()
    }

    #[test]
    fn duplicate_registration_is_idempotent_only_when_exact() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let node = NodeId::new();
        let epoch = WorkerEpoch::new();
        let request = RegisterWorker {
            node_id: node,
            worker_epoch: epoch,
            advertise_addr: "127.0.0.1:9001".into(),
            resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
            slots: 1,
            operations: vec![descriptor()],
        };
        let first = state.register_worker(request.clone(), 0, 100).unwrap();
        let second = state.register_worker(request, 1, 100).unwrap();
        assert_eq!(first.session_id, second.session_id);
        let mut conflict = descriptor();
        conflict.output_codec = Codec::JsonV1;
        assert!(state
            .register_worker(
                RegisterWorker {
                    node_id: node,
                    worker_epoch: epoch,
                    advertise_addr: "127.0.0.1:9001".into(),
                    resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                    slots: 1,
                    operations: vec![conflict]
                },
                2,
                100
            )
            .is_err());
    }

    #[test]
    fn failure_propagates_through_waiting_chain() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (_, first_output) = state
            .submit(
                descriptor().key.clone(),
                vec![TaskArg::Inline {
                    codec: Codec::RawBytes,
                    bytes: vec![1],
                }],
                ResourceSet::default(),
                1,
            )
            .unwrap();
        let (second, second_output) = state
            .submit(
                descriptor().key.clone(),
                vec![TaskArg::Object(first_output)],
                ResourceSet::default(),
                1,
            )
            .unwrap();
        let (third, _) = state
            .submit(
                descriptor().key.clone(),
                vec![TaskArg::Object(second_output)],
                ResourceSet::default(),
                1,
            )
            .unwrap();
        let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state
            .fail(
                &identity,
                assignment.fence,
                "boom".into(),
                FailureClass::Permanent,
                0,
            )
            .unwrap();
        assert!(matches!(state.tasks[&second].state, TaskState::Failed(_)));
        assert!(matches!(state.tasks[&third].state, TaskState::Failed(_)));
    }

    #[test]
    fn cancel_keeps_resources_until_worker_ack() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (task, _) = state
            .submit(
                descriptor().key.clone(),
                vec![],
                ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                1,
            )
            .unwrap();
        let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state.cancel(task).unwrap();
        assert_eq!(state.workers[&identity.node_id].free_slots, 0);
        state
            .acknowledge_cancel(&identity, assignment.fence)
            .unwrap();
        assert_eq!(state.workers[&identity.node_id].free_slots, 1);
    }

    #[test]
    fn stale_attempt_cannot_complete_retry() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 2)
            .unwrap();
        let first = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state
            .fail(
                &identity,
                first.fence,
                "retry".into(),
                FailureClass::Transient,
                0,
            )
            .unwrap();
        // Transient retry backs off ~100ms; dispatch with now past the deadline.
        let second = state.assign_next(identity.node_id, 1_000).unwrap().unwrap();
        assert_ne!(first.fence.attempt, second.fence.attempt);
        assert_eq!(
            state.started(&identity, first.fence),
            Err(Error::StaleFence)
        );
    }

    #[test]
    fn rejects_unschedulable_resources() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        register(&mut state, NodeId::new());
        let result = state.submit(
            descriptor().key,
            vec![],
            ResourceSet::cpu_gpu(2.0, 0.0).unwrap(),
            1,
        );
        assert!(matches!(result, Err(Error::InvalidResource(_))));
    }

    #[test]
    fn scheduler_drains_queue_and_leaves_no_stale_index_entries() {
        // Submit -> assign -> complete many single-slot tasks in sequence and
        // confirm the runnable queue and waiting index return to empty. A leak
        // here is exactly the O(n) scan regression the queue was added to kill.
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        for i in 0..50u8 {
            let (_, output) = state
                .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
                .unwrap();
            let assignment = state
                .assign_next(identity.node_id, 0)
                .unwrap()
                .expect("a freshly submitted task must be assignable");
            state.started(&identity, assignment.fence).unwrap();
            state
                .complete(
                    &identity,
                    crate::protocol::TaskCompletion {
                        fence: assignment.fence,
                        output_id: output,
                        codec: Codec::RawBytes,
                        size_bytes: 1,
                        checksum: crate::cluster::checksum(&[i]),
                        location: "127.0.0.1:9001".into(),
                        bytes: Some(vec![i].into()),
                    },
                )
                .unwrap();
        }
        assert!(state.runnable.is_empty(), "runnable queue leaked entries");
        assert!(state.waiting.is_empty(), "waiting index leaked entries");
        assert!(state.assign_next(identity.node_id, 0).unwrap().is_none());
    }

    #[test]
    fn cancelled_waiting_task_is_dropped_from_index() {
        // A task blocked on a dependency, then cancelled, must leave the waiting
        // index so reconciliation never revisits a terminal task.
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let _ = register(&mut state, NodeId::new());
        let (_, dep_output) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
            .unwrap();
        let (blocked, _) = state
            .submit(
                descriptor().key.clone(),
                vec![TaskArg::Object(dep_output)],
                ResourceSet::default(),
                1,
            )
            .unwrap();
        assert!(state.waiting.contains(&blocked));
        state.cancel(blocked).unwrap();
        assert!(!state.waiting.contains(&blocked));
        assert_eq!(state.tasks[&blocked].state, TaskState::Cancelled);
    }

    #[test]
    fn locality_counters_track_owned_input_placement() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (_, first_output) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
            .unwrap();
        let first = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state
            .complete(&identity, completion(first.fence, first_output))
            .unwrap();
        state
            .submit(
                descriptor().key.clone(),
                vec![TaskArg::Object(first_output)],
                ResourceSet::default(),
                1,
            )
            .unwrap();
        state.assign_next(identity.node_id, 0).unwrap().unwrap();
        assert_eq!(state.sched_assigns, 2);
        assert_eq!(state.sched_owned_input, 1);
        assert_eq!(state.sched_local_hits, 1);
    }

    #[test]
    fn draining_worker_gets_no_assignments() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
            .unwrap();
        state.drain(&identity).unwrap();
        assert!(state.assign_next(identity.node_id, 0).unwrap().is_none());
    }

    #[test]
    fn draining_worker_can_complete_in_flight_task() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (task, output) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
            .unwrap();
        let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state.drain(&identity).unwrap();
        state
            .complete(&identity, completion(assignment.fence, output))
            .unwrap();
        assert_eq!(state.tasks[&task].state, TaskState::Succeeded);
    }

    #[test]
    fn draining_worker_expiry_requeues_task() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new()); // lease deadline 100
        let (task, _) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 2)
            .unwrap();
        state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state.drain(&identity).unwrap();
        // Draining worker exits without reporting; the reaper re-queues.
        let expired = state.expire_workers(1_000).unwrap();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].node, identity.node_id);
        assert_eq!(expired[0].retried, 1);
        assert_eq!(state.tasks[&task].state, TaskState::Runnable);
    }

    fn completion(fence: TaskFence, output: ObjectId) -> TaskCompletion {
        TaskCompletion {
            fence,
            output_id: output,
            codec: Codec::RawBytes,
            size_bytes: 1,
            checksum: crate::cluster::checksum(&[7]),
            location: "127.0.0.1:9001".into(),
            bytes: Some(vec![7].into()),
        }
    }

    #[test]
    fn release_reserved_output_is_rejected_then_task_completes() {
        // Releasing a live task's still-Reserved output must be refused; otherwise
        // the object is deleted and the task's completion panics on a missing key.
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (_, output) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
            .unwrap();
        let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        assert_eq!(
            state.release_object(output),
            Err(Error::ObjectInUse(output))
        );
        state.started(&identity, assignment.fence).unwrap();
        state
            .complete(&identity, completion(assignment.fence, output))
            .unwrap();
        assert_eq!(
            state.tasks[&assignment.fence.task_id].state,
            TaskState::Succeeded
        );
    }

    #[test]
    fn release_reclaims_terminal_task_and_object() {
        // A released output must drop both the object and its now-terminal producer
        // task, so the task table is not a lifetime-capped leak.
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (task, output) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
            .unwrap();
        let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state.started(&identity, assignment.fence).unwrap();
        state
            .complete(&identity, completion(assignment.fence, output))
            .unwrap();
        state.release_object(output).unwrap();
        assert!(!state.tasks.contains_key(&task));
        assert!(!state.objects.contains_key(&output));
    }

    #[test]
    fn expired_worker_is_evicted() {
        // Dead ephemeral workers must be removed, not left as Dead records that
        // grow the map and inflate every submit's schedulability scan.
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let node = NodeId::new();
        let _ = register(&mut state, node); // lease_deadline_ms = 0 + 100
        let expired = state.expire_workers(200).unwrap();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].node, node);
        assert!(!state.workers.contains_key(&node));
        assert!(!state.pending_deletes.contains_key(&node));
    }

    #[test]
    fn cancellation_index_tracks_only_pending() {
        // cancellation_for scans this index instead of the whole task table, so it
        // must hold exactly the tasks awaiting a worker cancel ack.
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (task, _) = state
            .submit(
                descriptor().key.clone(),
                vec![],
                ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                1,
            )
            .unwrap();
        state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state.cancel(task).unwrap();
        assert!(state.cancel_requested.contains(&task));
        let fence = state.cancellation_for(&identity).unwrap().unwrap();
        assert_eq!(fence.task_id, task);
        state.acknowledge_cancel(&identity, fence).unwrap();
        assert!(state.cancel_requested.is_empty());
        assert_eq!(state.tasks[&task].state, TaskState::Cancelled);
    }

    #[test]
    fn unknown_task_id_report_does_not_panic() {
        // Worker reports for a task the coordinator never had must map to
        // TaskNotFound, not a HashMap-index panic in the idempotent short-circuit.
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let bogus = TaskFence {
            task_id: TaskId::new(),
            attempt: Attempt(1),
            lease_id: LeaseId::new(),
        };
        assert!(matches!(
            state.started(&identity, bogus),
            Err(Error::TaskNotFound(_))
        ));
        assert!(matches!(
            state.fail(&identity, bogus, "x".into(), FailureClass::Transient, 0),
            Err(Error::TaskNotFound(_))
        ));
        assert!(matches!(
            state.acknowledge_cancel(&identity, bogus),
            Err(Error::TaskNotFound(_))
        ));
        assert!(matches!(
            state.complete(&identity, completion(bogus, ObjectId::new())),
            Err(Error::TaskNotFound(_))
        ));
    }

    #[test]
    fn permanent_failure_is_not_retried_despite_remaining_attempts() {
        // A Permanent failure (e.g. an operation panic) is terminal even with
        // attempts left; only Transient failures retry.
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (task, _) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
            .unwrap();
        let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state
            .fail(
                &identity,
                a.fence,
                "boom".into(),
                FailureClass::Permanent,
                0,
            )
            .unwrap();
        assert!(matches!(state.tasks[&task].state, TaskState::Failed(_)));
        // A Transient failure with attempts left retries instead.
        let (task2, _) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
            .unwrap();
        let b = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state
            .fail(
                &identity,
                b.fence,
                "flaky".into(),
                FailureClass::Transient,
                0,
            )
            .unwrap();
        assert_eq!(state.tasks[&task2].state, TaskState::Runnable);
    }

    #[test]
    fn transient_retry_defers_dispatch_until_backoff_elapses() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (task, _) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
            .unwrap();
        let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        let outcome = state
            .fail(
                &identity,
                a.fence,
                "flaky".into(),
                FailureClass::Transient,
                1_000,
            )
            .unwrap();
        // First retry (attempt 1) backs off base = 100ms → due at 1100.
        assert_eq!(outcome, FailOutcome::Retried);
        assert_eq!(state.tasks[&task].state, TaskState::Runnable);
        assert_eq!(state.tasks[&task].not_before_ms, 1_100);
        // Not yet due: assign_next skips it and returns nothing.
        assert!(state
            .assign_next(identity.node_id, 1_050)
            .unwrap()
            .is_none());
        assert_eq!(state.tasks[&task].state, TaskState::Runnable);
        // Due: dispatched.
        assert!(state
            .assign_next(identity.node_id, 1_100)
            .unwrap()
            .is_some());
    }

    #[test]
    fn backoff_grows_exponentially_across_attempts() {
        assert_eq!(retry_backoff_ms(1), 100);
        assert_eq!(retry_backoff_ms(2), 200);
        assert_eq!(retry_backoff_ms(3), 400);
        // Capped at RETRY_BACKOFF_CAP_MS, and a huge attempt cannot overflow.
        assert_eq!(retry_backoff_ms(30), RETRY_BACKOFF_CAP_MS);
        assert_eq!(retry_backoff_ms(u32::MAX), RETRY_BACKOFF_CAP_MS);
    }

    #[test]
    fn lease_expiry_retry_also_applies_backoff() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        // Worker lease deadline = now(0) + 100.
        let identity = register(&mut state, NodeId::new());
        let (task, _) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
            .unwrap();
        state.assign_next(identity.node_id, 0).unwrap().unwrap();
        // Expire the lease at now = 200; attempt-1 retry → due at 300.
        let expired = state.expire_workers(200).unwrap();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].retried, 1);
        assert_eq!(state.tasks[&task].state, TaskState::Runnable);
        assert_eq!(state.tasks[&task].not_before_ms, 300);
    }

    #[test]
    fn permanent_failure_leaves_no_backoff_deadline() {
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        let (task, _) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
            .unwrap();
        let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        let outcome = state
            .fail(
                &identity,
                a.fence,
                "boom".into(),
                FailureClass::Permanent,
                500,
            )
            .unwrap();
        assert_eq!(outcome, FailOutcome::Failed);
        assert_eq!(state.tasks[&task].not_before_ms, 0);
    }

    #[test]
    fn assign_prefers_a_task_with_local_inputs() {
        // A worker that already owns an object should get the downstream task that
        // consumes it, even when a non-local task sits ahead in the FIFO queue.
        let mut state = CoordinatorState::new(CoordinatorEpoch::new());
        let identity = register(&mut state, NodeId::new());
        // Produce object O on this worker.
        let (_, output) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
            .unwrap();
        let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        state.started(&identity, a.fence).unwrap();
        state
            .complete(&identity, completion(a.fence, output))
            .unwrap();
        // Enqueue a non-local task first, then a local one that consumes O.
        let (non_local, _) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
            .unwrap();
        let (local, _) = state
            .submit(
                descriptor().key.clone(),
                vec![TaskArg::Object(output)],
                ResourceSet::default(),
                1,
            )
            .unwrap();
        let picked = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        assert_eq!(picked.fence.task_id, local);
        assert_ne!(picked.fence.task_id, non_local);
        // The non-local task keeps its place and is served next.
        state.started(&identity, picked.fence).unwrap();
        state
            .complete(
                &identity,
                completion(picked.fence, state.tasks[&local].output),
            )
            .unwrap();
        let next = state.assign_next(identity.node_id, 0).unwrap().unwrap();
        assert_eq!(next.fence.task_id, non_local);
    }
}
