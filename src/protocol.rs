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

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

pub const MAGIC: [u8; 4] = *b"CRYN";
pub const PROTOCOL_MAJOR: u16 = 2;
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
    pub bytes: Option<Vec<u8>>,
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
    Status(TaskId),
    /// Fetch an object, optionally blocking. The coordinator holds the request
    /// until the object is available or `wait_ms` elapses (`wait_ms: 0` returns
    /// immediately). Blocking replaces the client-side status-polling loop in
    /// `TaskHandle::result` with an event-driven wait for the reserved output.
    Get {
        object: ObjectId,
        wait_ms: u64,
    },
    GetLocal(ObjectId),
    Workers,
    Cancel(TaskId),
    Release(ObjectId),
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
    },
    Object {
        id: ObjectId,
        codec: Codec,
        size_bytes: u64,
        checksum: [u8; 32],
        location: String,
        bytes: Option<Vec<u8>>,
    },
    Submitted {
        task_id: TaskId,
        output_id: ObjectId,
    },
    Status(TaskView),
    Workers(Vec<WorkerView>),
    Cancelled,
    Released,
    Error(Error),
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
        retryable: bool,
    },
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
