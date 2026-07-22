use std::{
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
    error::{DeadlineContext, Error},
    ids::{ClusterId, CoordinatorEpoch},
    protocol::{
        ClientReply, ClientRequest, Envelope, RegisteredWorker, RpcReply, RpcRequest, TaskView,
        WorkerReply, WorkerRequest, WorkerView, DEFAULT_RPC_TIMEOUT_MS, MAX_FRAME_BYTES,
        MAX_OBJECT_BYTES,
    },
};

const MAX_CONNECTIONS: usize = 256;

#[derive(Clone)]
pub struct CoordinatorServer {
    pub cluster_id: ClusterId,
    pub state: Arc<Mutex<CoordinatorState>>,
    lease_ms: u64,
    connections: Arc<Semaphore>,
}
impl CoordinatorServer {
    pub fn new(cluster_id: ClusterId, lease_ms: u64) -> Self {
        Self {
            cluster_id,
            state: Arc::new(Mutex::new(CoordinatorState::new(CoordinatorEpoch::new()))),
            lease_ms,
            connections: Arc::new(Semaphore::new(MAX_CONNECTIONS)),
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
        let reply = self.dispatch(envelope.body);
        timeout_io(write_frame(&mut stream, &reply)).await
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
                        output_id: task.output,
                        state: task.state.clone(),
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

pub async fn request(address: &str, envelope: &Envelope) -> Result<RpcReply, Error> {
    let deadline = envelope.deadline_unix_ms.saturating_sub(now_ms());
    if deadline == 0 {
        return Err(Error::DeadlineExceeded(DeadlineContext::RpcRequest));
    }
    tokio::time::timeout(Duration::from_millis(deadline), async {
        let mut stream = TcpStream::connect(address).await?;
        write_frame(&mut stream, envelope).await?;
        read_frame(&mut stream)
            .await?
            .ok_or_else(|| Error::Protocol("connection closed before reply".into()))
    })
    .await
    .map_err(|_| Error::DeadlineExceeded(DeadlineContext::RpcRequest))?
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
        .map_err(|_| Error::DeadlineExceeded(DeadlineContext::RpcIo))?
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
