use crate::{
    error::Error,
    ids::{
        Attempt, CoordinatorEpoch, LeaseId, NodeId, ObjectId, Revision, TaskId, WorkerEpoch,
        WorkerSessionId,
    },
    operation::{Codec, OperationDescriptor, OperationKey, TaskArg},
    protocol::{
        RegisterWorker, TaskAssignment, TaskCompletion, TaskFence, TaskStatus, WorkerIdentity,
    },
    resources::ResourceSet,
};
use std::collections::{HashMap, HashSet, VecDeque};

pub const MAX_TASKS: usize = 16_384;
pub const MAX_OBJECTS: usize = 65_536;
pub const MAX_INLINE_OBJECT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WorkerState {
    Alive,
    Dead,
}
#[derive(Debug, Clone, Eq, PartialEq)]
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
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ObjectState {
    Reserved,
    Available,
    Failed,
    Cancelled,
    Lost,
}
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
}
#[derive(Debug, Clone)]
pub struct ObjectRecord {
    pub id: ObjectId,
    pub state: ObjectState,
    pub codec: Option<Codec>,
    pub bytes: Option<Vec<u8>>,
    pub size_bytes: Option<u64>,
    pub checksum: Option<[u8; 32]>,
    pub location: Option<String>,
    pub owner: Option<NodeId>,
    pub ref_count: u32,
}

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
    /// Tasks in `CancelRequested`; lets `cancellation_for` skip its scan at 0.
    cancel_requested: usize,
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
            cancel_requested: 0,
        }
    }
    fn changed(&mut self) {
        self.revision.0 += 1;
    }
    /// Marks a task `Runnable` and enqueues it for scheduling. The queue may end
    /// up with duplicate or stale ids; `assign_next` validates on pop, so the
    /// only invariant is that every genuinely-runnable task is present at least once.
    fn mark_runnable(&mut self, id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&id) {
            task.state = TaskState::Runnable;
        }
        self.waiting.remove(&id);
        self.runnable.push_back(id);
    }
    /// Marks a task `Waiting` on unresolved dependencies and indexes it.
    fn mark_waiting(&mut self, id: TaskId) {
        if let Some(task) = self.tasks.get_mut(&id) {
            task.state = TaskState::Waiting;
        }
        self.waiting.insert(id);
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
            if existing.state == WorkerState::Alive {
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
        if let Some(existing) = self.objects.get(&id) {
            if existing.state != ObjectState::Available
                || existing.codec.as_ref() != Some(&codec)
                || existing.size_bytes != Some(bytes.len() as u64)
                || existing.checksum != Some(checksum)
                || existing.bytes.as_deref() != Some(bytes.as_slice())
            {
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
                size_bytes: Some(bytes.len() as u64),
                checksum: Some(checksum),
                bytes: Some(bytes),
                location: Some("coordinator".into()),
                owner: None,
                ref_count: 1,
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
                ref_count: 1,
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
                // Overwritten immediately by mark_runnable/mark_waiting below,
                // under the same lock, before any observer can see it.
                state: TaskState::Waiting,
                assigned: None,
            },
        );
        if waiting {
            self.mark_waiting(id);
        } else {
            self.mark_runnable(id);
        }
        self.changed();
        Ok((id, output))
    }
    pub fn assign_next(&mut self, node_id: NodeId) -> Result<Option<TaskAssignment>, Error> {
        let worker = self.workers.get(&node_id).ok_or(Error::StaleFence)?;
        if worker.state != WorkerState::Alive || worker.free_slots == 0 {
            return Ok(None);
        }
        // Pop from the runnable queue, skipping stale entries (tasks no longer
        // runnable) and re-queuing runnable tasks this worker cannot serve — for
        // example a task needing an operation or resources it lacks. Bounded by
        // the number of genuinely-runnable tasks, not the whole task table.
        let mut requeue: Vec<TaskId> = Vec::new();
        let mut chosen = None;
        while let Some(id) = self.runnable.pop_front() {
            let Some(task) = self.tasks.get(&id) else {
                continue; // task gone
            };
            if task.state != TaskState::Runnable {
                continue; // stale entry (retried elsewhere, cancelled, ...)
            }
            let worker = self.workers.get(&node_id).unwrap();
            if worker.operations.contains_key(&task.operation)
                && worker.available.can_fit(&task.resources)
            {
                chosen = Some(id);
                break;
            }
            requeue.push(id); // runnable but not for this worker; keep it
        }
        for id in requeue {
            self.runnable.push_back(id);
        }
        let Some(id) = chosen else { return Ok(None) };
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
        // accepted so workers can safely retry reports.
        if self.tasks[&fence.task_id].state == TaskState::Running {
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
        let task = &self.tasks[&report.fence.task_id];
        // Idempotent: a duplicate completion for an already-succeeded task with
        // a matching output is accepted so workers can safely retry reports.
        if task.state == TaskState::Succeeded {
            let object = &self.objects[&task.output];
            if object.state == ObjectState::Available
                && object.checksum == Some(report.checksum)
                && object.size_bytes == Some(report.size_bytes)
            {
                return Ok(());
            }
            return Err(Error::ObjectConflict(task.output));
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
        retryable: bool,
    ) -> Result<(), Error> {
        // Idempotent: a duplicate failure report for an already-terminal task
        // is accepted so workers can safely retry reports.
        if matches!(
            self.tasks[&fence.task_id].state,
            TaskState::Failed(_) | TaskState::Cancelled | TaskState::Succeeded
        ) {
            return Ok(());
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
        let retry = retryable
            && self.tasks[&fence.task_id].attempt.0 < self.tasks[&fence.task_id].max_attempts;
        let output = self.tasks[&fence.task_id].output;
        self.release(identity.node_id, &resources)?;
        let task = self.tasks.get_mut(&fence.task_id).unwrap();
        task.assigned = None;
        if retry {
            self.mark_runnable(fence.task_id);
        } else {
            task.state = TaskState::Failed(message);
            self.objects.get_mut(&output).unwrap().state = ObjectState::Failed;
            self.reconcile_dependencies();
        }
        self.changed();
        Ok(())
    }
    pub fn expire_workers(&mut self, now: u64) -> Result<Vec<NodeId>, Error> {
        let expired: Vec<_> = self
            .workers
            .values()
            .filter(|worker| worker.state == WorkerState::Alive && worker.lease_deadline_ms <= now)
            .map(|worker| worker.identity.node_id)
            .collect();
        for node in &expired {
            self.workers.get_mut(node).unwrap().state = WorkerState::Dead;
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
                    self.cancel_requested = self.cancel_requested.saturating_sub(1);
                    self.objects.get_mut(&output).unwrap().state = ObjectState::Cancelled;
                } else if task.attempt.0 < task.max_attempts {
                    self.mark_runnable(id);
                } else {
                    task.state = TaskState::Failed("worker lease expired".into());
                    self.objects.get_mut(&output).unwrap().state = ObjectState::Failed;
                }
            }
            for object in self.objects.values_mut() {
                if object.owner == Some(*node) && object.state == ObjectState::Available {
                    object.state = ObjectState::Lost;
                    object.location = None;
                }
            }
        }
        if !expired.is_empty() {
            self.reconcile_dependencies();
            self.changed();
        }
        Ok(expired)
    }
    pub fn cancellation_for(&self, identity: &WorkerIdentity) -> Result<Option<TaskFence>, Error> {
        self.check_identity(identity)?;
        if self.cancel_requested == 0 {
            return Ok(None);
        }
        Ok(self.tasks.values().find_map(|task| {
            let (node, epoch, session, lease_id) = task.assigned?;
            (node == identity.node_id
                && epoch == identity.worker_epoch
                && session == identity.session_id
                && task.state == TaskState::CancelRequested)
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
        // accepted so workers can safely retry reports.
        if self.tasks[&fence.task_id].state == TaskState::Cancelled {
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
        let task = self.tasks.get_mut(&fence.task_id).unwrap();
        task.state = TaskState::Cancelled;
        task.assigned = None;
        self.cancel_requested = self.cancel_requested.saturating_sub(1);
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
            self.tasks.get_mut(&id).unwrap().state = TaskState::CancelRequested;
            self.cancel_requested += 1;
        } else {
            self.tasks.get_mut(&id).unwrap().state = TaskState::Cancelled;
            // stale runnable-queue entry is skipped by assign_next on pop
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
        if self.object_in_use(&id) {
            return Err(Error::ObjectInUse(id));
        }
        if let Some(owner) = object.owner {
            self.pending_deletes.entry(owner).or_default().push(id);
        }
        self.objects.remove(&id);
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
        if worker.state != WorkerState::Alive
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
        let assignment = state.assign_next(identity.node_id).unwrap().unwrap();
        state
            .fail(&identity, assignment.fence, "boom".into(), false)
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
        let assignment = state.assign_next(identity.node_id).unwrap().unwrap();
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
        let first = state.assign_next(identity.node_id).unwrap().unwrap();
        state
            .fail(&identity, first.fence, "retry".into(), true)
            .unwrap();
        let second = state.assign_next(identity.node_id).unwrap().unwrap();
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
                .assign_next(identity.node_id)
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
                        bytes: Some(vec![i]),
                    },
                )
                .unwrap();
        }
        assert!(state.runnable.is_empty(), "runnable queue leaked entries");
        assert!(state.waiting.is_empty(), "waiting index leaked entries");
        assert!(state.assign_next(identity.node_id).unwrap().is_none());
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
}
