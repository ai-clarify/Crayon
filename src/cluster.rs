use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use parking_lot::Mutex;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{Notify, Semaphore},
};

use crate::{
    coordinator::CoordinatorState,
    error::Error,
    ids::{ClusterId, CoordinatorEpoch, RequestId},
    protocol::{
        ClientReply, ClientRequest, Envelope, ObjectPayload, RegisteredWorker, RpcReply,
        RpcRequest, TaskView, WorkerReply, WorkerRequest, WorkerView, DEFAULT_RPC_TIMEOUT_MS,
        MAX_FRAME_BYTES, MAX_OBJECT_BYTES,
    },
};

const MAX_CONNECTIONS: usize = 256;
const RETRY_DELAY: Duration = Duration::from_millis(25);
const MAX_REPLAY_ENTRIES: usize = 16_384;
const MAX_REPLAY_BYTES: usize = 64 * 1024 * 1024;
/// Per-entry bookkeeping overhead (request-id key + expiry + hash) charged on
/// top of the serialized reply size when accounting the replay byte budget.
const REPLAY_ENTRY_OVERHEAD: usize = 40;
/// Ceiling on how long the coordinator parks a long-poll, kept under the RPC
/// transport deadline so a held request always answers before the client retries.
const MAX_LONG_POLL_MS: u64 = 4_000;
/// Belt-and-suspenders re-check interval while parked. `Notify` wakes a parked
/// poll on any state change; this cap bounds the delay if a wakeup is ever missed.
const LONG_POLL_SLICE_MS: u64 = 250;

#[derive(Clone)]
struct ReplayEntry {
    body_hash: [u8; 32],
    reply: RpcReply,
    expires_at_ms: u64,
    /// Serialized reply size, computed once at insert. Kept on the entry so the
    /// cache byte budget is a running sum instead of re-serializing every entry
    /// on each request (which made cached submits O(n^2) under load).
    bytes: usize,
}

#[derive(Clone)]
pub struct CoordinatorServer {
    pub cluster_id: ClusterId,
    pub state: Arc<Mutex<CoordinatorState>>,
    /// Lock-free copy of the write-once state epoch, for epoch fencing.
    epoch: CoordinatorEpoch,
    lease_ms: u64,
    connections: Arc<Semaphore>,
    replay: Arc<Mutex<HashMap<RequestId, ReplayEntry>>>,
    /// Fires on every state change so parked long-polls (worker `Poll`, client
    /// blocking `Get`) wake immediately instead of spinning on a timer.
    wakeup: Arc<Notify>,
    require_loopback: bool,
}
impl CoordinatorServer {
    pub fn new(cluster_id: ClusterId, lease_ms: u64) -> Self {
        // one epoch for both state and the hoisted copy
        let epoch = CoordinatorEpoch::new();
        Self {
            cluster_id,
            state: Arc::new(Mutex::new(CoordinatorState::new(epoch))),
            epoch,
            lease_ms,
            connections: Arc::new(Semaphore::new(MAX_CONNECTIONS)),
            replay: Arc::new(Mutex::new(HashMap::new())),
            wakeup: Arc::new(Notify::new()),
            require_loopback: true,
        }
    }
    /// Allows the coordinator to bind to non-loopback addresses. Unauthenticated
    /// clusters must stay on loopback; cross-host deployments require mTLS.
    pub fn allow_remote_bind(mut self) -> Self {
        self.require_loopback = false;
        self
    }
    pub async fn serve(self, bind: &str) -> Result<(), Error> {
        if self.require_loopback {
            crate::protocol::require_loopback_addr(bind)?;
        }
        let listener = TcpListener::bind(bind).await?;
        let reaper = self.clone();
        tokio::spawn(async move {
            let period = Duration::from_millis((reaper.lease_ms / 3).max(10));
            let mut interval = tokio::time::interval(period);
            loop {
                interval.tick().await;
                let expired = reaper.state.lock().expire_workers(now_ms());
                if !matches!(expired, Ok(ref list) if list.is_empty()) {
                    // Expiry retries or fails tasks; wake parked polls to react.
                    reaper.wakeup.notify_waiters();
                }
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
                let _ = stream.set_nodelay(true);
                let _ = server.handle(stream).await;
            });
        }
    }
    async fn handle(&self, mut stream: TcpStream) -> Result<(), Error> {
        let envelope = timeout_io(read_frame::<Envelope>(&mut stream))
            .await?
            .ok_or_else(|| Error::Protocol("connection closed before request".into()))?;
        envelope.validate(self.cluster_id, now_ms())?;
        let reply = self.dispatch_blocking(&envelope).await?;
        timeout_io(write_frame(&mut stream, &reply)).await
    }
    /// Handles a request, parking long-poll variants (`Poll`, blocking `Get`)
    /// until work is ready or their wait budget elapses. Everything else answers
    /// immediately. State-changing dispatches wake parked polls.
    async fn dispatch_blocking(&self, envelope: &Envelope) -> Result<RpcReply, Error> {
        match &envelope.body {
            RpcRequest::Worker(WorkerRequest::Poll { wait_ms, .. }) => {
                self.long_poll(envelope, *wait_ms, worker_poll_ready).await
            }
            RpcRequest::Client(ClientRequest::Get { wait_ms, .. }) if *wait_ms > 0 => {
                self.long_poll(envelope, *wait_ms, client_get_ready).await
            }
            RpcRequest::Client(ClientRequest::GetBatch { wait_ms, .. }) if *wait_ms > 0 => {
                self.long_poll(envelope, *wait_ms, client_get_batch_ready)
                    .await
            }
            _ => self.dispatch_and_notify(envelope),
        }
    }
    /// Dispatches once and, if the coordinator state advanced, wakes parked
    /// long-polls so a newly-runnable task or newly-available object is observed
    /// without waiting for the fallback re-check slice.
    fn dispatch_and_notify(&self, envelope: &Envelope) -> Result<RpcReply, Error> {
        let before = self.state.lock().revision;
        let reply = self.dispatch_once(envelope)?;
        if self.state.lock().revision != before {
            self.wakeup.notify_waiters();
        }
        Ok(reply)
    }
    async fn long_poll(
        &self,
        envelope: &Envelope,
        wait_ms: u64,
        ready: fn(&RpcReply) -> bool,
    ) -> Result<RpcReply, Error> {
        let deadline =
            tokio::time::Instant::now() + Duration::from_millis(wait_ms.min(MAX_LONG_POLL_MS));
        loop {
            // Register for wakeup BEFORE dispatching: Notify permits are one-shot
            // and not stored, so enabling first guarantees a state change between
            // our dispatch and our await cannot be lost.
            let notified = self.wakeup.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            let reply = self.dispatch_and_notify(envelope)?;
            if ready(&reply) {
                return Ok(reply);
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Ok(reply);
            }
            let slice = remaining.min(Duration::from_millis(LONG_POLL_SLICE_MS));
            let _ = tokio::time::timeout(slice, notified).await;
        }
    }
    fn dispatch_once(&self, envelope: &Envelope) -> Result<RpcReply, Error> {
        if is_mutation(&envelope.body) {
            self.validate_epoch(envelope)?;
        }
        if !should_cache(&envelope.body) {
            return Ok(self.dispatch(envelope.body.clone()));
        }
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
        let entry_bytes = estimate_reply_bytes(&reply);
        // Sum of stored sizes: one pass, no re-serialization.
        let used: usize = replay
            .values()
            .map(|entry| entry.bytes + REPLAY_ENTRY_OVERHEAD)
            .sum();
        if replay.len() >= MAX_REPLAY_ENTRIES
            || used + entry_bytes + REPLAY_ENTRY_OVERHEAD > MAX_REPLAY_BYTES
        {
            return Ok(request_error(
                &envelope.body,
                Error::CapacityExceeded("replay cache limit reached".into()),
            ));
        }
        replay.insert(
            envelope.request_id,
            ReplayEntry {
                body_hash,
                reply: reply.clone(),
                expires_at_ms: envelope.deadline_unix_ms,
                bytes: entry_bytes,
            },
        );
        Ok(reply)
    }
    fn validate_epoch(&self, envelope: &Envelope) -> Result<(), Error> {
        match envelope.coordinator_epoch {
            Some(epoch) if epoch == self.epoch => Ok(()),
            Some(_) => Err(Error::StaleEpoch),
            None => Err(Error::Protocol(
                "mutation requires coordinator epoch".into(),
            )),
        }
    }
    fn dispatch(&self, request: RpcRequest) -> RpcReply {
        match request {
            RpcRequest::Worker(request) => RpcReply::Worker(match request {
                WorkerRequest::Register(request) => {
                    let registration = {
                        let mut state = self.state.lock();
                        if self.require_loopback {
                            if let Err(error) =
                                crate::protocol::require_loopback_addr(&request.advertise_addr)
                            {
                                Err(error)
                            } else {
                                match state.register_worker(request, now_ms(), self.lease_ms) {
                                    Ok(identity) => Ok((identity, state.revision)),
                                    Err(error) => Err(error),
                                }
                            }
                        } else {
                            match state.register_worker(request, now_ms(), self.lease_ms) {
                                Ok(identity) => Ok((identity, state.revision)),
                                Err(error) => Err(error),
                            }
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
                WorkerRequest::Poll { identity, .. } => {
                    let mut state = self.state.lock();
                    let mut deletes = state.take_pending_deletes(identity.node_id);
                    if let Some(id) = deletes.first().copied() {
                        deletes.remove(0);
                        if !deletes.is_empty() {
                            state
                                .pending_deletes
                                .entry(identity.node_id)
                                .or_default()
                                .extend(deletes);
                        }
                        WorkerReply::DeleteObject(id)
                    } else {
                        match state.cancellation_for(&identity) {
                            Ok(Some(fence)) => WorkerReply::Cancel(fence),
                            Ok(None) => match state.assign_next(identity.node_id) {
                                Ok(value) => WorkerReply::Assignment(value),
                                Err(error) => WorkerReply::Error(error),
                            },
                            Err(error) => WorkerReply::Error(error),
                        }
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
                ClientRequest::Connect => ClientReply::Connected {
                    coordinator_epoch: self.epoch,
                },
                ClientRequest::Put { codec, bytes } => {
                    if bytes.len() > MAX_OBJECT_BYTES {
                        ClientReply::Error(Error::Protocol("object too large".into()))
                    } else {
                        match self.state.lock().put(codec.clone(), bytes.clone()) {
                            Ok(id) => {
                                let checksum = checksum(&bytes);
                                ClientReply::Object(ObjectPayload {
                                    id,
                                    codec,
                                    size_bytes: bytes.len() as u64,
                                    checksum,
                                    location: "coordinator".into(),
                                    bytes: Some(bytes.into()),
                                })
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
                ClientRequest::SubmitBatch(specs) => {
                    let mut state = self.state.lock();
                    let results = specs
                        .into_iter()
                        .map(|spec| {
                            state
                                .submit(
                                    spec.operation,
                                    spec.args,
                                    spec.resources,
                                    spec.max_attempts,
                                )
                                .map(|(task_id, output_id)| crate::protocol::SubmittedTask {
                                    task_id,
                                    output_id,
                                })
                        })
                        .collect();
                    ClientReply::SubmittedBatch(results)
                }
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
                ClientRequest::Get { object: id, .. } => {
                    match self.state.lock().resolve_object(id) {
                        Ok(payload) => ClientReply::Object(payload),
                        Err(error) => ClientReply::Error(error),
                    }
                }
                ClientRequest::GetBatch { objects, .. } => {
                    let state = self.state.lock();
                    ClientReply::ObjectBatch(
                        objects
                            .into_iter()
                            .map(|id| state.resolve_object(id))
                            .collect(),
                    )
                }
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
                ClientRequest::Release(id) => match self.state.lock().release_object(id) {
                    Ok(()) => ClientReply::Released,
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

fn is_mutation(request: &RpcRequest) -> bool {
    match request {
        RpcRequest::Client(client) => matches!(
            client,
            ClientRequest::Put { .. }
                | ClientRequest::Submit { .. }
                | ClientRequest::SubmitBatch(_)
                | ClientRequest::Cancel(_)
                | ClientRequest::Release(_)
        ),
        RpcRequest::Worker(worker) => !matches!(
            worker,
            WorkerRequest::Register(_) | WorkerRequest::Poll { .. }
        ),
    }
}

/// A worker `Poll` is satisfied by anything other than "no work yet": an
/// assignment, a cancellation, an object deletion, or an error worth returning.
fn worker_poll_ready(reply: &RpcReply) -> bool {
    !matches!(reply, RpcReply::Worker(WorkerReply::Assignment(None)))
}

/// A blocking `Get` is satisfied once the object resolves either way; only a
/// still-reserved (pending) output keeps it parked.
fn client_get_ready(reply: &RpcReply) -> bool {
    !matches!(
        reply,
        RpcReply::Client(ClientReply::Error(Error::ObjectPending(_)))
    )
}

/// A blocking `GetBatch` is satisfied only when every object has resolved; a
/// single still-pending object keeps the whole batch parked. Callers that want
/// partial results pass `wait_ms: 0` for an immediate, non-parking reply.
fn client_get_batch_ready(reply: &RpcReply) -> bool {
    match reply {
        RpcReply::Client(ClientReply::ObjectBatch(results)) => !results
            .iter()
            .any(|result| matches!(result, Err(Error::ObjectPending(_)))),
        _ => true,
    }
}

/// Only client mutations are cached. Worker reports are idempotent at the
/// coordinator state machine and Poll returns transient state, so caching them
/// would only fill the cache under load without adding safety.
fn should_cache(request: &RpcRequest) -> bool {
    matches!(
        request,
        RpcRequest::Client(
            ClientRequest::Submit { .. }
                | ClientRequest::SubmitBatch(_)
                | ClientRequest::Cancel(_)
                | ClientRequest::Release(_)
        )
    )
}

fn estimate_reply_bytes(reply: &RpcReply) -> usize {
    bincode::serialize(reply)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
}

pub async fn request(address: &str, envelope: &Envelope) -> Result<RpcReply, Error> {
    for attempt_number in 0..2 {
        let remaining_ms = envelope.deadline_unix_ms.saturating_sub(now_ms());
        if remaining_ms == 0 {
            return Err(Error::DeadlineExceeded);
        }
        let attempt = tokio::time::timeout(Duration::from_millis(remaining_ms), async {
            let mut stream = TcpStream::connect(address).await?;
            let _ = stream.set_nodelay(true);
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

    fn with_epoch(server: &CoordinatorServer, mut envelope: Envelope) -> Envelope {
        envelope.coordinator_epoch = Some(server.state.lock().epoch);
        envelope
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
        let request = with_epoch(
            &server,
            future_envelope(
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
            ),
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
        let first = with_epoch(
            &server,
            future_envelope(
                server.cluster_id,
                RpcRequest::Client(ClientRequest::Submit {
                    operation: descriptor().key,
                    args: vec![],
                    resources: ResourceSet::default(),
                    max_attempts: 1,
                }),
            ),
        );
        server.dispatch_once(&first).unwrap();
        let mut conflicting = with_epoch(
            &server,
            future_envelope(
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
            ),
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
            .dispatch_once(&with_epoch(
                &server,
                future_envelope(
                    server.cluster_id,
                    RpcRequest::Client(ClientRequest::Submit {
                        operation: descriptor().key,
                        args: vec![],
                        resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                        max_attempts: 1,
                    }),
                ),
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
        let completion = with_epoch(
            &server,
            future_envelope(
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
                        bytes: Some(vec![7].into()),
                    },
                }),
            ),
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
