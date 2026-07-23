use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use parking_lot::Mutex;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Semaphore,
};

use crate::{
    coordinator::CoordinatorState,
    error::Error,
    ids::{ClusterId, CoordinatorEpoch, RequestId},
    protocol::{
        ClientReply, ClientRequest, Envelope, RegisteredWorker, RpcReply, RpcRequest, TaskView,
        WorkerReply, WorkerRequest, WorkerView, DEFAULT_RPC_TIMEOUT_MS, MAX_FRAME_BYTES,
        MAX_OBJECT_BYTES,
    },
};

const MAX_CONNECTIONS: usize = 256;
const RETRY_DELAY: Duration = Duration::from_millis(25);

#[derive(Clone)]
struct ReplayEntry {
    body_hash: [u8; 32],
    reply: RpcReply,
    expires_at_ms: u64,
}

#[derive(Clone)]
pub struct CoordinatorServer {
    pub cluster_id: ClusterId,
    pub state: Arc<Mutex<CoordinatorState>>,
    lease_ms: u64,
    connections: Arc<Semaphore>,
    replay: Arc<Mutex<HashMap<RequestId, ReplayEntry>>>,
}
impl CoordinatorServer {
    pub fn new(cluster_id: ClusterId, lease_ms: u64) -> Self {
        Self {
            cluster_id,
            state: Arc::new(Mutex::new(CoordinatorState::new(CoordinatorEpoch::new()))),
            lease_ms,
            connections: Arc::new(Semaphore::new(MAX_CONNECTIONS)),
            replay: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    pub async fn serve(self, bind: &str) -> Result<(), Error> {
        let listener = TcpListener::bind(bind).await?;
        let reaper = self.clone();
        tokio::spawn(async move {
            let period = Duration::from_millis((reaper.lease_ms / 3).max(10));
            let mut interval = tokio::time::interval(period);
            loop {
                interval.tick().await;
                let _ = reaper.state.lock().expire_workers(now_ms());
            }
        });
        loop {
            let (stream, _) = listener.accept().await?;
            let permit = match self.connections.clone().try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => continue,
            };
            let server = self.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let _ = server.handle(stream).await;
            });
        }
    }
    async fn handle(&self, mut stream: TcpStream) -> Result<(), Error> {
        let envelope = timeout_io(read_frame::<Envelope>(&mut stream))
            .await?
            .ok_or_else(|| Error::Protocol("connection closed before request".into()))?;
        envelope.validate(self.cluster_id, now_ms())?;
        let reply = self.dispatch_once(&envelope)?;
        timeout_io(write_frame(&mut stream, &reply)).await
    }
    fn dispatch_once(&self, envelope: &Envelope) -> Result<RpcReply, Error> {
        let now = now_ms();
        let body_hash = checksum(&bincode::serialize(&envelope.body)?);
        let mut replay = self.replay.lock();
        replay.retain(|_, entry| entry.expires_at_ms > now);
        if let Some(entry) = replay.get(&envelope.request_id) {
            if entry.body_hash != body_hash {
                return Ok(request_error(
                    &envelope.body,
                    Error::Protocol("request id reused with different body".into()),
                ));
            }
            return Ok(entry.reply.clone());
        }
        let reply = self.dispatch(envelope.body.clone());
        replay.insert(
            envelope.request_id,
            ReplayEntry {
                body_hash,
                reply: reply.clone(),
                expires_at_ms: envelope.deadline_unix_ms,
            },
        );
        Ok(reply)
    }
    fn dispatch(&self, request: RpcRequest) -> RpcReply {
        match request {
            RpcRequest::Worker(request) => RpcReply::Worker(match request {
                WorkerRequest::Register(request) => {
                    let registration = {
                        let mut state = self.state.lock();
                        match state.register_worker(request, now_ms(), self.lease_ms) {
                            Ok(identity) => Ok((identity, state.revision)),
                            Err(error) => Err(error),
                        }
                    };
                    match registration {
                        Ok((identity, revision)) => WorkerReply::Registered(RegisteredWorker {
                            identity,
                            revision,
                            lease_timeout_ms: self.lease_ms,
                        }),
                        Err(error) => WorkerReply::Error(error),
                    }
                }
                WorkerRequest::Heartbeat(identity) => {
                    match self
                        .state
                        .lock()
                        .heartbeat(&identity, now_ms(), self.lease_ms)
                    {
                        Ok(()) => WorkerReply::Accepted,
                        Err(error) => WorkerReply::Error(error),
                    }
                }
                WorkerRequest::Poll(identity) => {
                    let mut state = self.state.lock();
                    match state.cancellation_for(&identity) {
                        Ok(Some(fence)) => WorkerReply::Cancel(fence),
                        Ok(None) => match state.assign_next(identity.node_id) {
                            Ok(value) => WorkerReply::Assignment(value),
                            Err(error) => WorkerReply::Error(error),
                        },
                        Err(error) => WorkerReply::Error(error),
                    }
                }
                WorkerRequest::Cancelled { identity, fence } => {
                    match self.state.lock().acknowledge_cancel(&identity, fence) {
                        Ok(()) => WorkerReply::Accepted,
                        Err(error) => WorkerReply::Error(error),
                    }
                }
                WorkerRequest::Started { identity, fence } => {
                    match self.state.lock().started(&identity, fence) {
                        Ok(()) => WorkerReply::Accepted,
                        Err(error) => WorkerReply::Error(error),
                    }
                }
                WorkerRequest::Completed { identity, report } => {
                    match self.state.lock().complete(&identity, report) {
                        Ok(()) => WorkerReply::Accepted,
                        Err(error) => WorkerReply::Error(error),
                    }
                }
                WorkerRequest::Failed {
                    identity,
                    fence,
                    message,
                    retryable,
                } => match self.state.lock().fail(&identity, fence, message, retryable) {
                    Ok(()) => WorkerReply::Accepted,
                    Err(error) => WorkerReply::Error(error),
                },
            }),
            RpcRequest::Client(request) => RpcReply::Client(match request {
                ClientRequest::Put { codec, bytes } => {
                    if bytes.len() > MAX_OBJECT_BYTES {
                        ClientReply::Error(Error::Protocol("object too large".into()))
                    } else {
                        match self.state.lock().put(codec.clone(), bytes.clone()) {
                            Ok(id) => {
                                let checksum = checksum(&bytes);
                                ClientReply::Object {
                                    id,
                                    codec,
                                    size_bytes: bytes.len() as u64,
                                    checksum,
                                    location: "coordinator".into(),
                                    bytes: Some(bytes),
                                }
                            }
                            Err(error) => ClientReply::Error(error),
                        }
                    }
                }
                ClientRequest::Submit {
                    operation,
                    args,
                    resources,
                    max_attempts,
                } => match self
                    .state
                    .lock()
                    .submit(operation, args, resources, max_attempts)
                {
                    Ok((task_id, output_id)) => ClientReply::Submitted { task_id, output_id },
                    Err(error) => ClientReply::Error(error),
                },
                ClientRequest::Status(id) => match self.state.lock().tasks.get(&id) {
                    Some(task) => ClientReply::Status(TaskView {
                        task_id: task.id,
                        output_id: task.output,
                        state: (&task.state).into(),
                        attempt: task.attempt,
                        worker: task.assigned.map(|assigned| assigned.0),
                    }),
                    None => ClientReply::Error(Error::TaskNotFound(id)),
                },
                ClientRequest::Get(id) => match self.state.lock().objects.get(&id) {
                    Some(object) if object.state == crate::coordinator::ObjectState::Available => {
                        ClientReply::Object {
                            id,
                            codec: object.codec.clone().unwrap(),
                            size_bytes: object.size_bytes.unwrap(),
                            checksum: object.checksum.unwrap(),
                            location: object.location.clone().unwrap(),
                            bytes: object.bytes.clone(),
                        }
                    }
                    Some(object) if object.state == crate::coordinator::ObjectState::Lost => {
                        ClientReply::Error(Error::ObjectLost(id))
                    }
                    _ => ClientReply::Error(Error::Protocol("object unavailable".into())),
                },
                ClientRequest::GetLocal(_) => ClientReply::Error(Error::Protocol(
                    "coordinator has no worker-local object endpoint".into(),
                )),
                ClientRequest::Workers => {
                    let state = self.state.lock();
                    ClientReply::Workers(
                        state
                            .workers
                            .values()
                            .map(|worker| WorkerView {
                                identity: worker.identity.clone(),
                                advertise_addr: worker.advertise_addr.clone(),
                                alive: worker.state == crate::coordinator::WorkerState::Alive,
                                slots: worker.slots,
                                free_slots: worker.free_slots,
                                resources: worker.total.clone(),
                                available: worker.available.clone(),
                                operations: worker.operations.values().cloned().collect(),
                            })
                            .collect(),
                    )
                }
                ClientRequest::Cancel(id) => match self.state.lock().cancel(id) {
                    Ok(()) => ClientReply::Cancelled,
                    Err(error) => ClientReply::Error(error),
                },
            }),
        }
    }
}

fn request_error(request: &RpcRequest, error: Error) -> RpcReply {
    match request {
        RpcRequest::Client(_) => RpcReply::Client(ClientReply::Error(error)),
        RpcRequest::Worker(_) => RpcReply::Worker(WorkerReply::Error(error)),
    }
}

pub async fn request(address: &str, envelope: &Envelope) -> Result<RpcReply, Error> {
    for attempt_number in 0..2 {
        let remaining_ms = envelope.deadline_unix_ms.saturating_sub(now_ms());
        if remaining_ms == 0 {
            return Err(Error::DeadlineExceeded);
        }
        let attempt = tokio::time::timeout(Duration::from_millis(remaining_ms), async {
            let mut stream = TcpStream::connect(address).await?;
            write_frame(&mut stream, envelope).await?;
            read_frame(&mut stream)
                .await?
                .ok_or_else(|| Error::Io("connection closed before reply".into()))
        })
        .await;
        match attempt {
            Ok(Ok(reply)) => return Ok(reply),
            Ok(Err(Error::Io(_))) if attempt_number == 0 => {}
            Ok(Err(error)) => return Err(error),
            Err(_) => return Err(Error::DeadlineExceeded),
        }
        let remaining_ms = envelope.deadline_unix_ms.saturating_sub(now_ms());
        if remaining_ms == 0 {
            return Err(Error::DeadlineExceeded);
        }
        tokio::time::sleep(RETRY_DELAY.min(Duration::from_millis(remaining_ms))).await;
    }
    unreachable!()
}

pub fn envelope(cluster_id: ClusterId, body: RpcRequest) -> Envelope {
    Envelope::new(
        cluster_id,
        body,
        now_ms().saturating_add(DEFAULT_RPC_TIMEOUT_MS),
    )
}

pub async fn write_frame<T: serde::Serialize>(
    stream: &mut TcpStream,
    value: &T,
) -> Result<(), Error> {
    let bytes = bincode::serialize(value)?;
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(Error::Protocol("frame too large".into()));
    }
    stream.write_u32(bytes.len() as u32).await?;
    stream.write_all(&bytes).await?;
    Ok(())
}
pub async fn read_frame<T: serde::de::DeserializeOwned>(
    stream: &mut TcpStream,
) -> Result<Option<T>, Error> {
    let len = match stream.read_u32().await {
        Ok(value) => value as usize,
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if len > MAX_FRAME_BYTES {
        return Err(Error::Protocol("frame too large".into()));
    }
    let mut bytes = vec![0; len];
    stream.read_exact(&mut bytes).await?;
    Ok(Some(bincode::deserialize(&bytes)?))
}
async fn timeout_io<T>(
    future: impl std::future::Future<Output = Result<T, Error>>,
) -> Result<T, Error> {
    tokio::time::timeout(Duration::from_millis(DEFAULT_RPC_TIMEOUT_MS), future)
        .await
        .map_err(|_| Error::DeadlineExceeded)?
}
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
pub fn checksum(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ids::{NodeId, WorkerEpoch},
        operation::{Codec, OperationDescriptor, OperationKey, TaskArg},
        protocol::{RegisterWorker, TaskStatus},
        resources::ResourceSet,
    };

    fn descriptor() -> OperationDescriptor {
        OperationDescriptor {
            key: OperationKey::new("test", "copy", 1),
            input_codec: Codec::RawBytes,
            output_codec: Codec::RawBytes,
            max_inline_arg_bytes: 8,
        }
    }

    fn future_envelope(cluster: ClusterId, body: RpcRequest) -> Envelope {
        Envelope::new(cluster, body, now_ms().saturating_add(10_000))
    }

    fn registered_server() -> (CoordinatorServer, crate::protocol::WorkerIdentity) {
        let cluster = ClusterId::new();
        let server = CoordinatorServer::new(cluster, 5_000);
        let reply = server
            .dispatch_once(&future_envelope(
                cluster,
                RpcRequest::Worker(WorkerRequest::Register(RegisterWorker {
                    node_id: NodeId::new(),
                    worker_epoch: WorkerEpoch::new(),
                    advertise_addr: "127.0.0.1:9001".into(),
                    resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                    slots: 1,
                    operations: vec![descriptor()],
                })),
            ))
            .unwrap();
        let RpcReply::Worker(WorkerReply::Registered(registered)) = reply else {
            panic!("registration failed")
        };
        (server, registered.identity)
    }

    #[test]
    fn duplicate_submit_is_dispatched_once() {
        let (server, _) = registered_server();
        let request = future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::Submit {
                operation: descriptor().key,
                args: vec![TaskArg::Inline {
                    codec: Codec::RawBytes,
                    bytes: vec![1],
                }],
                resources: ResourceSet::default(),
                max_attempts: 1,
            }),
        );
        let first = server.dispatch_once(&request).unwrap();
        let second = server.dispatch_once(&request).unwrap();
        let submitted = |reply| match reply {
            RpcReply::Client(ClientReply::Submitted { task_id, output_id }) => (task_id, output_id),
            _ => panic!("submit failed"),
        };
        assert_eq!(submitted(first), submitted(second));
        assert_eq!(server.state.lock().tasks.len(), 1);
    }

    #[test]
    fn request_id_cannot_be_reused_with_another_body() {
        let (server, _) = registered_server();
        let first = future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::Workers),
        );
        server.dispatch_once(&first).unwrap();
        let mut conflicting = future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::Status(crate::ids::TaskId::new())),
        );
        conflicting.request_id = first.request_id;
        assert!(matches!(
            server.dispatch_once(&conflicting),
            Ok(RpcReply::Client(ClientReply::Error(Error::Protocol(_))))
        ));
    }

    #[test]
    fn completion_reply_is_replayed_without_releasing_twice() {
        let (server, identity) = registered_server();
        let submit = server
            .dispatch_once(&future_envelope(
                server.cluster_id,
                RpcRequest::Client(ClientRequest::Submit {
                    operation: descriptor().key,
                    args: vec![],
                    resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                    max_attempts: 1,
                }),
            ))
            .unwrap();
        let RpcReply::Client(ClientReply::Submitted { task_id, output_id }) = submit else {
            panic!("submit failed")
        };
        let assignment = server
            .state
            .lock()
            .assign_next(identity.node_id)
            .unwrap()
            .unwrap();
        server
            .state
            .lock()
            .started(&identity, assignment.fence)
            .unwrap();
        let node_id = identity.node_id;
        let completion = future_envelope(
            server.cluster_id,
            RpcRequest::Worker(WorkerRequest::Completed {
                identity,
                report: crate::protocol::TaskCompletion {
                    fence: assignment.fence,
                    output_id,
                    codec: Codec::RawBytes,
                    size_bytes: 1,
                    checksum: checksum(&[7]),
                    location: "127.0.0.1:9001".into(),
                },
            }),
        );
        assert!(matches!(
            server.dispatch_once(&completion),
            Ok(RpcReply::Worker(WorkerReply::Accepted))
        ));
        assert!(matches!(
            server.dispatch_once(&completion),
            Ok(RpcReply::Worker(WorkerReply::Accepted))
        ));
        let state = server.state.lock();
        assert_eq!(
            crate::protocol::TaskStatus::from(&state.tasks[&task_id].state),
            TaskStatus::Succeeded
        );
        assert_eq!(state.workers[&node_id].free_slots, 1);
    }

    #[tokio::test]
    async fn transport_retry_reuses_the_envelope() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let cluster = ClusterId::new();
        let envelope = future_envelope(cluster, RpcRequest::Client(ClientRequest::Workers));
        let expected = envelope.request_id;
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let received = read_frame::<Envelope>(&mut stream).await.unwrap().unwrap();
                assert_eq!(received.request_id, expected);
                if attempt == 1 {
                    write_frame(&mut stream, &RpcReply::Client(ClientReply::Workers(vec![])))
                        .await
                        .unwrap();
                }
            }
        });
        assert!(matches!(
            request(&address.to_string(), &envelope).await,
            Ok(RpcReply::Client(ClientReply::Workers(_)))
        ));
        server.await.unwrap();
    }
}
