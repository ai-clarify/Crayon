use crate::{
    error::Error,
    ids::{
        Attempt, ClusterId, CoordinatorEpoch, LeaseId, NodeId, ObjectId, RequestId, Revision,
        TaskId, WorkerEpoch, WorkerSessionId,
    },
    operation::{Codec, OperationDescriptor, OperationKey, TaskArg},
    resources::ResourceSet,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

pub const MAGIC: [u8; 4] = *b"CRYN";
// Arena reservations now carry an owner token and Direct/Streamed mode;
// streamed commit requires complete coverage. Wire-incompatible with 7.
pub const PROTOCOL_MAJOR: u16 = 8;
pub const PROTOCOL_MINOR: u16 = 0;
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_OBJECT_BYTES: usize = MAX_FRAME_BYTES - 64 * 1024;
pub const DEFAULT_RPC_TIMEOUT_MS: u64 = 5_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub magic: [u8; 4],
    pub major: u16,
    pub minor: u16,
    pub cluster_id: ClusterId,
    pub coordinator_epoch: Option<CoordinatorEpoch>,
    pub request_id: RequestId,
    pub deadline_unix_ms: u64,
    pub body: RpcRequest,
}
impl Envelope {
    pub fn new(cluster_id: ClusterId, body: RpcRequest, deadline_unix_ms: u64) -> Self {
        Self {
            magic: MAGIC,
            major: PROTOCOL_MAJOR,
            minor: PROTOCOL_MINOR,
            cluster_id,
            coordinator_epoch: None,
            request_id: RequestId::new(),
            deadline_unix_ms,
            body,
        }
    }
    pub fn validate(&self, expected_cluster: ClusterId, now_ms: u64) -> Result<(), Error> {
        if self.magic != MAGIC || self.major != PROTOCOL_MAJOR || self.minor > PROTOCOL_MINOR {
            return Err(Error::Protocol("incompatible protocol".into()));
        }
        if self.cluster_id != expected_cluster {
            return Err(Error::Protocol("cluster id mismatch".into()));
        }
        if self.deadline_unix_ms <= now_ms {
            return Err(Error::DeadlineExceeded);
        }
        Ok(())
    }
}
pub fn validate_advertise_addr(value: &str) -> Result<(), Error> {
    let Some((host, port)) = value.rsplit_once(':') else {
        return Err(Error::InvalidAddress(value.into()));
    };
    let port: u16 = port
        .parse()
        .map_err(|_| Error::InvalidAddress(value.into()))?;
    if port == 0 {
        return Err(Error::InvalidAddress(value.into()));
    }
    if host.is_empty() {
        return Err(Error::InvalidAddress(value.into()));
    }
    Ok(())
}
pub fn require_loopback_addr(value: &str) -> Result<(), Error> {
    validate_advertise_addr(value)?;
    let Some((host, _)) = value.rsplit_once(':') else {
        return Err(Error::InvalidAddress(value.into()));
    };
    if host != "127.0.0.1" && host != "::1" && host != "localhost" {
        return Err(Error::InvalidAddress(format!(
            "unauthenticated mode requires loopback address: {value}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerIdentity {
    pub node_id: NodeId,
    pub worker_epoch: WorkerEpoch,
    pub session_id: WorkerSessionId,
    pub coordinator_epoch: CoordinatorEpoch,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterWorker {
    pub node_id: NodeId,
    pub worker_epoch: WorkerEpoch,
    pub advertise_addr: String,
    pub resources: ResourceSet,
    pub slots: u32,
    pub operations: Vec<OperationDescriptor>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredWorker {
    pub identity: WorkerIdentity,
    pub revision: Revision,
    pub lease_timeout_ms: u64,
}
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct TaskFence {
    pub task_id: TaskId,
    pub attempt: Attempt,
    pub lease_id: LeaseId,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAssignment {
    pub fence: TaskFence,
    pub operation: OperationKey,
    pub args: Vec<TaskArg>,
    pub output_id: ObjectId,
    pub resources: ResourceSet,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCompletion {
    pub fence: TaskFence,
    pub output_id: ObjectId,
    pub codec: Codec,
    pub size_bytes: u64,
    pub checksum: [u8; 32],
    pub location: String,
    /// Small outputs are shipped inline with the completion so the coordinator
    /// can answer `Get` in one hop instead of redirecting to the worker.
    /// `None` for large outputs, which are fetched from `location` on demand.
    pub bytes: Option<Arc<[u8]>>,
}
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArenaWriteMode {
    Direct,
    Streamed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientRequest {
    Connect,
    Put {
        codec: Codec,
        bytes: Vec<u8>,
    },
    Submit {
        operation: OperationKey,
        args: Vec<TaskArg>,
        resources: ResourceSet,
        max_attempts: u32,
    },
    /// Submit many tasks in one RPC. Each spec is admitted independently; the
    /// reply carries a per-spec result so a partial failure does not sink the
    /// batch. Collapses N submit round-trips into one.
    SubmitBatch(Vec<SubmitSpec>),
    Status(TaskId),
    /// Fetch an object, optionally blocking. The coordinator holds the request
    /// until the object is available or `wait_ms` elapses (`wait_ms: 0` returns
    /// immediately). Blocking replaces the client-side status-polling loop in
    /// `TaskHandle::result` with an event-driven wait for the reserved output.
    Get {
        object: ObjectId,
        wait_ms: u64,
    },
    /// Fetch many objects in one RPC. Blocks until at least `min_ready` of them
    /// have resolved (available or terminal) or `wait_ms` elapses; still-pending
    /// slots return `ObjectPending`. `min_ready == objects.len()` is the
    /// all-or-nothing `ray.get([refs])`; a smaller value is the first-K-ready
    /// `ray.wait`, letting a caller drain finished work before slow slots land.
    GetBatch {
        objects: Vec<ObjectId>,
        wait_ms: u64,
        min_ready: u32,
    },
    GetLocal(ObjectId),
    /// Reserve an arena slot for a client-side shared-memory put. The client
    /// then writes its bytes at the returned offset and sends `ArenaCommit`.
    /// Bypasses the RPC frame cap, so objects scale to gigabytes. An all-zero
    /// checksum means the payload is unhashed (large objects skip the pass;
    /// the id is then random, not content-derived).
    ArenaReserve {
        id: ObjectId,
        codec: Codec,
        size_bytes: u64,
        checksum: [u8; 32],
        mode: ArenaWriteMode,
    },
    ArenaCommit {
        id: ObjectId,
        reservation: RequestId,
    },
    /// Write a byte range of a reserved arena object over TCP, for a cross-host
    /// client that cannot mmap the arena. Sits between `ArenaReserve` and
    /// `ArenaCommit`: the coordinator copies `bytes` into the slot at `offset`.
    /// Each chunk is one frame, so a multi-GiB object crosses as many chunks.
    PutChunk {
        id: ObjectId,
        reservation: RequestId,
        offset: u64,
        bytes: Vec<u8>,
    },
    /// Read a byte range of a committed object, for a cross-host client that
    /// cannot mmap the arena. The whole-object checksum is verified by the
    /// client after reassembly; per-chunk fetches are stateless (id+offset+len).
    GetChunk {
        id: ObjectId,
        offset: u64,
        len: u64,
    },
    Workers,
    Cancel(TaskId),
    Release(ObjectId),
}
/// One task in a `SubmitBatch`, carrying the same fields as a single `Submit`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitSpec {
    pub operation: OperationKey,
    pub args: Vec<TaskArg>,
    pub resources: ResourceSet,
    pub max_attempts: u32,
}
/// A successfully admitted task: its id and reserved output id.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SubmittedTask {
    pub task_id: TaskId,
    pub output_id: ObjectId,
}
/// A resolved object's payload. Small outputs arrive with `bytes` set; large
/// outputs carry `location` for a worker-local fetch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectPayload {
    pub id: ObjectId,
    pub codec: Codec,
    pub size_bytes: u64,
    pub checksum: [u8; 32],
    pub location: String,
    pub bytes: Option<Arc<[u8]>>,
    /// Set when the payload lives in the host-local shared-memory arena: a
    /// same-host reader maps the arena and reads it zero-copy at `arena.offset`
    /// instead of receiving `bytes`.
    pub arena: Option<crate::arena::ArenaRef>,
}
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum TaskStatus {
    Waiting,
    Runnable,
    Assigned,
    Running,
    CancelRequested,
    Succeeded,
    Failed(String),
    Cancelled,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskView {
    pub task_id: TaskId,
    pub output_id: ObjectId,
    pub state: TaskStatus,
    pub attempt: Attempt,
    pub worker: Option<NodeId>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerView {
    pub identity: WorkerIdentity,
    pub advertise_addr: String,
    pub alive: bool,
    pub slots: u32,
    pub free_slots: u32,
    pub resources: ResourceSet,
    pub available: ResourceSet,
    pub operations: Vec<OperationDescriptor>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientReply {
    Connected {
        coordinator_epoch: CoordinatorEpoch,
        /// Token of the coordinator's shared-memory arena. A client that can
        /// map it is co-located and puts/gets payloads through the arena.
        arena_token: String,
    },
    Object(ObjectPayload),
    /// Slot granted for an `ArenaReserve`: write the bytes at `offset`, then
    /// send `ArenaCommit` with the reservation token. (A reserve of already-stored
    /// content returns `Object` instead.)
    ArenaReserved {
        id: ObjectId,
        offset: u64,
        reservation: RequestId,
    },
    Submitted {
        task_id: TaskId,
        output_id: ObjectId,
    },
    /// Per-spec admission results, in request order: `Ok` for admitted tasks,
    /// `Err` for specs the coordinator rejected.
    SubmittedBatch(Vec<Result<SubmittedTask, Error>>),
    /// Per-object fetch results, in request order. Each is the resolved payload
    /// or the error that object resolved to (failed/cancelled/lost/pending).
    ObjectBatch(Vec<Result<ObjectPayload, Error>>),
    Status(TaskView),
    Workers(Vec<WorkerView>),
    Cancelled,
    Released,
    /// A byte range of a committed object, answering `GetChunk`.
    Chunk(Vec<u8>),
    /// Acks a `PutChunk`: the range was written into the reserved slot.
    ChunkWritten,
    Error(Error),
}
/// Why a task attempt failed, so the coordinator owns the retry decision instead
/// of trusting a bare worker-supplied bool. `Permanent` failures (e.g. an
/// operation panic) are never retried even if attempts remain; `Transient`
/// failures (infra, missing input) retry up to `max_attempts`.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum FailureClass {
    Transient,
    Permanent,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerRequest {
    Register(RegisterWorker),
    Heartbeat(WorkerIdentity),
    /// Long-poll for work. The coordinator holds the request until an assignment,
    /// cancellation, or object deletion is ready, or `wait_ms` elapses. A busy
    /// worker polling only for cancellation passes `wait_ms: 0` for a prompt reply.
    Poll {
        identity: WorkerIdentity,
        wait_ms: u64,
    },
    Cancelled {
        identity: WorkerIdentity,
        fence: TaskFence,
    },
    Started {
        identity: WorkerIdentity,
        fence: TaskFence,
    },
    Completed {
        identity: WorkerIdentity,
        report: TaskCompletion,
    },
    Failed {
        identity: WorkerIdentity,
        fence: TaskFence,
        message: String,
        class: FailureClass,
    },
    /// SIGTERM received: stop assigning me work. An in-flight task may still
    /// report Completed/Failed afterwards; if the worker exits before it can,
    /// the lease reaper re-queues the task as with any silent death.
    Drain(WorkerIdentity),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcRequest {
    Client(ClientRequest),
    Worker(WorkerRequest),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RpcReply {
    Client(ClientReply),
    Worker(WorkerReply),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerReply {
    Registered(RegisteredWorker),
    Assignment(Option<TaskAssignment>),
    Cancel(TaskFence),
    DeleteObject(ObjectId),
    Accepted,
    Error(Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_rejects_wrong_cluster_and_deadline() {
        let cluster = ClusterId::new();
        let envelope = Envelope::new(
            cluster,
            RpcRequest::Client(ClientRequest::Status(TaskId::new())),
            10,
        );
        assert!(envelope.validate(cluster, 9).is_ok());
        assert!(envelope.validate(ClusterId::new(), 9).is_err());
        assert!(matches!(
            envelope.validate(cluster, 10),
            Err(Error::DeadlineExceeded)
        ));
    }
}
