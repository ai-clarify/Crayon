//! The RPC server, dispatch, and wire transport for the coordinator.
//!
//! # Contract
//!
//! This is the only IO layer over [`CoordinatorState`]: it wraps the whole
//! aggregate in one `Mutex` and turns framed RPCs into state transitions. The
//! state machine stays pure; everything that blocks, allocates a socket, or
//! touches the clock lives here.
//!
//! Invariants callers and protocol peers may rely on:
//! - **One frame, one bound.** Every TCP frame is length-prefixed and capped at
//!   `MAX_FRAME_BYTES` (8 MiB); a client-put object is capped at
//!   `MAX_OBJECT_BYTES`. Only the shared-memory arena path may carry a payload
//!   larger than one frame — see [`crate::arena`].
//! - **Mutations are idempotent by request id.** `should_cache` lists every
//!   cached (mutating) request; the replay cache returns the first response for
//!   a request id until its TTL, so a retry cannot double-apply. A pooled
//!   connection retry must dial fresh (`request`).
//! - **Blocking is bounded.** Long-poll variants (`Poll`, blocking `Get`/
//!   `GetBatch`) park on `wakeup` up to `MAX_LONG_POLL_MS`, re-checking on a
//!   slice so a missed notify cannot hang a caller past its deadline.
//! - **The state lock is never held across IO.** Handlers lock, transition,
//!   unlock, then write the reply frame — the reaper likewise snapshots under
//!   the lock and logs outside it.
//!
//! Tests live in `cluster_tests.rs`.

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
/// Per-entry overhead (key + expiry + hash) added to reply size for the byte budget.
const REPLAY_ENTRY_OVERHEAD: usize = 40;
/// Server ceiling on how long a mutation reply is retained for idempotent
/// replay, independent of the client-chosen (unbounded) envelope deadline.
const MAX_REPLAY_TTL_MS: u64 = 30_000;
/// Long-poll park ceiling, kept under the RPC deadline so a held request answers first.
const MAX_LONG_POLL_MS: u64 = 4_000;
/// Fallback re-check slice guarding a missed `Notify` wakeup.
const LONG_POLL_SLICE_MS: u64 = 250;

#[derive(Clone)]
struct ReplayEntry {
    body_hash: [u8; 32],
    reply: RpcReply,
    expires_at_ms: u64,
    /// Reply size, computed once at insert; the byte budget sums these, no re-serialize.
    bytes: usize,
}

/// Idempotency cache with an O(1) hot path: a running byte total replaces the
/// per-call `values().sum()`, and expiry is lazy — the touched entry is checked
/// on read, the full sweep runs only under capacity pressure.
#[derive(Default)]
struct ReplayCache {
    entries: HashMap<RequestId, ReplayEntry>,
    used_bytes: usize,
}
impl ReplayCache {
    fn remove(&mut self, id: &RequestId) {
        if let Some(entry) = self.entries.remove(id) {
            self.used_bytes -= entry.bytes + REPLAY_ENTRY_OVERHEAD;
        }
    }
    fn insert(&mut self, id: RequestId, entry: ReplayEntry) {
        self.used_bytes += entry.bytes + REPLAY_ENTRY_OVERHEAD;
        self.entries.insert(id, entry);
    }
    fn at_capacity(&self) -> bool {
        self.entries.len() >= MAX_REPLAY_ENTRIES || self.used_bytes >= MAX_REPLAY_BYTES
    }
    /// Drop expired entries; called only when `at_capacity`, so the O(n) scan is
    /// amortized away from the common dispatch path.
    fn sweep(&mut self, now: u64) {
        let mut freed = 0;
        self.entries.retain(|_, entry| {
            let keep = entry.expires_at_ms > now;
            if !keep {
                freed += entry.bytes + REPLAY_ENTRY_OVERHEAD;
            }
            keep
        });
        self.used_bytes -= freed;
    }
}

#[derive(Clone)]
pub struct CoordinatorServer {
    pub cluster_id: ClusterId,
    pub state: Arc<Mutex<CoordinatorState>>,
    /// Lock-free copy of the write-once state epoch, for epoch fencing.
    epoch: CoordinatorEpoch,
    lease_ms: u64,
    connections: Arc<Semaphore>,
    replay: Arc<Mutex<ReplayCache>>,
    /// Fires on every state change so parked long-polls (worker `Poll`, client
    /// blocking `Get`) wake immediately instead of spinning on a timer.
    wakeup: Arc<Notify>,
    require_loopback: bool,
    /// Same-host shared-memory arena: client-put payloads are published here so
    /// a co-located `get` maps the arena once and reads them zero-copy instead
    /// of pulling the bytes over TCP.
    arena: Arc<crate::arena::ArenaStore>,
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
            replay: Arc::new(Mutex::new(ReplayCache::default())),
            wakeup: Arc::new(Notify::new()),
            require_loopback: true,
            arena: Arc::new(
                crate::arena::ArenaStore::new().expect("create shared-memory object arena"),
            ),
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
        eprintln!(
            "crayon.start bind={bind} cluster_id={} arena_token={} arena_path={} lease_ms={}",
            self.cluster_id,
            self.arena.token(),
            self.arena.path().display(),
            self.lease_ms
        );
        let reaper = self.clone();
        tokio::spawn(async move {
            let period = Duration::from_millis((reaper.lease_ms / 3).max(10));
            let mut interval = tokio::time::interval(period);
            let mut locality_logged = 0;
            // Health snapshot throttled to ~30s of wall time, independent of the
            // (lease-derived) tick period, so the soak grep gets a steady cadence.
            let health_period_ms = 30_000u64;
            let mut health_logged_ms = 0u64;
            loop {
                interval.tick().await;
                let now = now_ms();
                // expire_workers runs every tick (lease enforcement needs the fast
                // cadence); the terminal-task reclaim and the full-table health
                // scan only run on the ~30s report cadence — no point scanning the
                // whole table ~19×/report just to discard 18 of the results.
                let report = now.saturating_sub(health_logged_ms) >= health_period_ms;
                let (expired, reclaimed, (assigns, owned, hits), health) = {
                    let mut state = reaper.state.lock();
                    let expired = state.expire_workers(now);
                    let reclaimed = if report {
                        state.reap_terminal_tasks(now)
                    } else {
                        0
                    };
                    let health = if report {
                        Some(state.health_counts())
                    } else {
                        None
                    };
                    (
                        expired,
                        reclaimed,
                        (
                            state.sched_assigns,
                            state.sched_owned_input,
                            state.sched_local_hits,
                        ),
                        health,
                    )
                };
                if reclaimed > 0 {
                    eprintln!("crayon.reclaim terminal_tasks={reclaimed}");
                }
                // Locality measurement (evolution-plan gap 3), logged here so the
                // state machine stays IO-free and no lock is held while printing.
                if assigns - locality_logged >= 1024 {
                    locality_logged = assigns;
                    eprintln!(
                        "scheduler locality: {hits}/{owned} owned-input tasks placed locally ({assigns} assigns)"
                    );
                }
                if let Some(health) = health {
                    health_logged_ms = now;
                    eprintln!(
                        "crayon.health tasks={} runnable={} running={} succeeded={} failed={} objects={} available={} reserved={} lost={} workers={} arena_bytes={} retries_total={} failures_total={}",
                        health.tasks, health.runnable, health.running, health.succeeded,
                        health.failed, health.objects, health.available, health.reserved,
                        health.lost, health.workers, reaper.arena.used_bytes(),
                        health.retries_total, health.failures_total
                    );
                }
                match expired {
                    Ok(list) if !list.is_empty() => {
                        for e in &list {
                            eprintln!(
                                "crayon.worker_dead node={} retried={} failed={} cancelled={} objects_lost={}",
                                e.node, e.retried, e.failed, e.cancelled, e.objects_lost
                            );
                        }
                        // Expiry retries or fails tasks; wake parked polls to react.
                        reaper.wakeup.notify_waiters();
                    }
                    _ => {}
                }
            }
        });
        loop {
            let (stream, _) = tokio::select! {
                accepted = listener.accept() => accepted?,
                // SIGTERM/SIGINT: stop accepting and fall through to drain.
                _ = shutdown_signal() => break,
            };
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
        // Graceful drain: no longer accepting, wait for in-flight handlers to
        // finish (each holds a connection permit) up to a bounded deadline, then
        // exit cleanly instead of being hard-killed mid-request.
        let drain = self.connections.acquire_many(MAX_CONNECTIONS as u32);
        let _ = tokio::time::timeout(Duration::from_secs(10), drain).await;
        Ok(())
    }
    async fn handle(&self, mut stream: TcpStream) -> Result<(), Error> {
        // Serve frames until the peer hangs up, so a client reuses one
        // connection across RPCs instead of paying a TCP setup per call.
        // An idle connection is closed by the read timeout; the client's
        // retry reconnects.
        loop {
            let Some(envelope) = timeout_io(read_frame::<Envelope>(&mut stream)).await? else {
                return Ok(());
            };
            envelope.validate(self.cluster_id, now_ms())?;
            let reply = self.dispatch_blocking(&envelope).await?;
            timeout_io(write_frame(&mut stream, &reply)).await?;
        }
    }
    /// Parks long-poll variants (`Poll`, blocking `Get`/`GetBatch`) until ready or
    /// their wait elapses; everything else dispatches immediately.
    async fn dispatch_blocking(&self, envelope: &Envelope) -> Result<RpcReply, Error> {
        match &envelope.body {
            RpcRequest::Worker(WorkerRequest::Poll { wait_ms, .. }) => {
                self.long_poll(envelope, *wait_ms, worker_poll_ready).await
            }
            RpcRequest::Client(ClientRequest::Get { wait_ms, .. }) if *wait_ms > 0 => {
                self.long_poll(envelope, *wait_ms, client_get_ready).await
            }
            RpcRequest::Client(ClientRequest::GetBatch {
                wait_ms, min_ready, ..
            }) if *wait_ms > 0 => {
                let min_ready = *min_ready;
                self.long_poll(envelope, *wait_ms, move |reply| {
                    client_get_batch_ready(reply, min_ready)
                })
                .await
            }
            _ => self.dispatch_and_notify(envelope),
        }
    }
    /// Dispatches once; if state advanced, wakes parked long-polls immediately.
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
        ready: impl Fn(&RpcReply) -> bool,
    ) -> Result<RpcReply, Error> {
        // Poll assigns work and must wake peers; blocking Get/GetBatch are reads
        // that never advance revision, so they skip the notify wrapper.
        let notifies = matches!(
            envelope.body,
            RpcRequest::Worker(WorkerRequest::Poll { .. })
        );
        let deadline =
            tokio::time::Instant::now() + Duration::from_millis(wait_ms.min(MAX_LONG_POLL_MS));
        loop {
            // enable() before dispatch: Notify permits are one-shot, so a state
            // change between dispatch and await cannot be lost.
            let notified = self.wakeup.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            let reply = if notifies {
                self.dispatch_and_notify(envelope)?
            } else {
                self.dispatch_once(envelope)?
            };
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
        // Lazy expiry: check only the entry we touch. A stale hit is dropped so
        // the request re-dispatches; the full sweep is deferred to capacity pressure.
        if let Some(entry) = replay.entries.get(&envelope.request_id) {
            let (expires_at_ms, entry_hash, reply) =
                (entry.expires_at_ms, entry.body_hash, entry.reply.clone());
            if expires_at_ms <= now {
                replay.remove(&envelope.request_id);
            } else if entry_hash != body_hash {
                return Ok(request_error(
                    &envelope.body,
                    Error::Protocol("request id reused with different body".into()),
                ));
            } else {
                return Ok(reply);
            }
        }
        // Capacity check BEFORE dispatch so every dispatched mutation is cached.
        // Reversing this order would commit the side effect but skip the insert on
        // saturation, turning the client's retry into a duplicate task. Returning
        // here has no side effect; the retry can re-dispatch cleanly.
        if replay.at_capacity() {
            // Only now pay the O(n) expiry sweep — reclaim before rejecting.
            replay.sweep(now);
        }
        if replay.at_capacity() {
            return Ok(request_error(
                &envelope.body,
                Error::CapacityExceeded("replay cache limit reached".into()),
            ));
        }
        let reply = self.dispatch(envelope.body.clone());
        let entry_bytes = estimate_reply_bytes(&reply);
        // ponytail: soft byte cap, overshoot <= one max reply; hard-evict oldest only if that ever matters.
        replay.insert(
            envelope.request_id,
            ReplayEntry {
                body_hash,
                reply: reply.clone(),
                // Clamp retention to a server ceiling: the client-controlled
                // deadline is unbounded, so a far-future deadline could otherwise
                // pin an entry indefinitely and let crafted requests DoS the cache.
                expires_at_ms: now.saturating_add(
                    envelope
                        .deadline_unix_ms
                        .saturating_sub(now)
                        .min(MAX_REPLAY_TTL_MS),
                ),
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
    /// Rewrites a payload to its arena form when the object lives in the arena:
    /// inline bytes are dropped and a same-host reader takes the offset instead.
    /// A cross-host client cannot map the arena and refetches via `GetLocal`.
    fn arena_annotate(&self, payload: &mut ObjectPayload) {
        if let Some(meta) = self.arena.meta(payload.id) {
            payload.bytes = None;
            payload.arena = Some(crate::arena::ArenaRef {
                token: self.arena.token().to_string(),
                offset: meta.offset,
            });
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
                            Ok(None) => match state.assigned_unstarted(&identity) {
                                Ok(Some(value)) => WorkerReply::Assignment(Some(value)),
                                Ok(None) => match state.assign_next(identity.node_id, now_ms()) {
                                    Ok(value) => WorkerReply::Assignment(value),
                                    Err(error) => WorkerReply::Error(error),
                                },
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
                    class,
                } => {
                    let task_id = fence.task_id;
                    let reason = message.clone();
                    match self
                        .state
                        .lock()
                        .fail(&identity, fence, message, class, now_ms())
                    {
                        Ok(permanently_failed) => {
                            if permanently_failed {
                                eprintln!(
                                    "crayon.task_failed task={task_id} attempt={} class={class:?} reason={reason:?}",
                                    fence.attempt.0
                                );
                            }
                            WorkerReply::Accepted
                        }
                        Err(error) => WorkerReply::Error(error),
                    }
                }
                WorkerRequest::Drain(identity) => match self.state.lock().drain(&identity) {
                    Ok(()) => WorkerReply::Accepted,
                    Err(error) => WorkerReply::Error(error),
                },
            }),
            RpcRequest::Client(request) => RpcReply::Client(match request {
                ClientRequest::Connect => ClientReply::Connected {
                    coordinator_epoch: self.epoch,
                    arena_token: self.arena.token().to_string(),
                },
                ClientRequest::Put { codec, bytes } => {
                    if bytes.len() > MAX_OBJECT_BYTES {
                        ClientReply::Error(Error::Protocol("object too large".into()))
                    } else {
                        match self.state.lock().put(codec.clone(), bytes) {
                            Ok(crate::coordinator::PutOutcome {
                                id,
                                checksum,
                                stored,
                            }) => {
                                // Publish into the arena so a same-host get is
                                // zero-copy. Reuse the stored Arc + checksum from
                                // put(); no re-hash, no re-clone of the payload.
                                self.arena.put(id, &stored, codec.clone(), checksum);
                                // The put reply is metadata only: every caller
                                // keeps just payload.id (client.rs:106), so
                                // echoing the payload back was a full round-trip
                                // of wasted bytes on the wire.
                                ClientReply::Object(ObjectPayload {
                                    id,
                                    codec,
                                    size_bytes: stored.len() as u64,
                                    checksum,
                                    location: "coordinator".into(),
                                    bytes: None,
                                    arena: None,
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
                        Ok(mut payload) => {
                            self.arena_annotate(&mut payload);
                            ClientReply::Object(payload)
                        }
                        Err(error) => ClientReply::Error(error),
                    }
                }
                ClientRequest::GetBatch { objects, .. } => {
                    let overflow = Err(Error::CapacityExceeded(
                        "get batch response too large".into(),
                    ));
                    let base = bincode::serialized_size(&RpcReply::Client(
                        ClientReply::ObjectBatch(Vec::new()),
                    ))
                    .unwrap_or(u64::MAX);
                    let overflow_size = bincode::serialized_size(&overflow).unwrap_or(u64::MAX);
                    if base.saturating_add(overflow_size.saturating_mul(objects.len() as u64))
                        > MAX_FRAME_BYTES as u64
                    {
                        ClientReply::Error(Error::CapacityExceeded(
                            "get batch response too large".into(),
                        ))
                    } else {
                        let state = self.state.lock();
                        let mut used = base;
                        let mut results = Vec::with_capacity(objects.len());
                        for (index, id) in objects.iter().copied().enumerate() {
                            let resolved = state.resolve_object(id);
                            let size = bincode::serialized_size(&resolved).unwrap_or(u64::MAX);
                            let remaining = (objects.len() - index - 1) as u64;
                            if used
                                .saturating_add(size)
                                .saturating_add(remaining.saturating_mul(overflow_size))
                                <= MAX_FRAME_BYTES as u64
                            {
                                used += size;
                                results.push(resolved);
                            } else {
                                used += overflow_size;
                                results.push(overflow.clone());
                            }
                        }
                        ClientReply::ObjectBatch(results)
                    }
                }
                ClientRequest::GetLocal(id) => {
                    // Explicit byte fetch: the cross-host fallback for an arena
                    // object whose file the caller could not map. Always inlines.
                    match self.state.lock().resolve_object(id) {
                        Ok(mut payload) => match self.arena.read(id) {
                            Some(bytes) if payload.bytes.is_none() => {
                                if bytes.len() <= MAX_OBJECT_BYTES {
                                    payload.bytes = Some(bytes.into());
                                    ClientReply::Object(payload)
                                } else {
                                    // ponytail: >8MB arena objects are same-host
                                    // only; stream in chunks if cross-host big
                                    // objects ever matter.
                                    ClientReply::Error(Error::Protocol(
                                        "object too large for cross-host fetch".into(),
                                    ))
                                }
                            }
                            _ => ClientReply::Object(payload),
                        },
                        Err(error) => ClientReply::Error(error),
                    }
                }
                ClientRequest::ArenaReserve {
                    id,
                    codec,
                    size_bytes,
                    checksum,
                    mode,
                } => {
                    // Content addressing: same checksum => same object, so a
                    // repeat reserve of stored content resolves as a plain get.
                    // (Unhashed large puts use random ids and never dedup.)
                    if let Ok(mut payload) = self.state.lock().resolve_object(id) {
                        self.arena_annotate(&mut payload);
                        ClientReply::Object(payload)
                    } else {
                        match self.arena.reserve(id, size_bytes, codec, checksum, mode) {
                            Ok(reservation) => ClientReply::ArenaReserved {
                                id,
                                offset: reservation.offset,
                                reservation: reservation.reservation,
                            },
                            Err(error) => ClientReply::Error(error),
                        }
                    }
                }
                ClientRequest::ArenaCommit { id, reservation } => {
                    match self.arena.commit(id, reservation) {
                        Ok(meta) => {
                            let mut state = self.state.lock();
                            match state
                                .put_meta(id, meta.codec, meta.size, meta.checksum)
                                .and_then(|id| state.resolve_object(id))
                            {
                                Ok(mut payload) => {
                                    self.arena_annotate(&mut payload);
                                    ClientReply::Object(payload)
                                }
                                Err(error) => {
                                    let _ = self.arena.rollback(id, reservation);
                                    ClientReply::Error(error)
                                }
                            }
                        }
                        Err(error) => ClientReply::Error(error),
                    }
                }
                ClientRequest::PutChunk {
                    id,
                    reservation,
                    offset,
                    bytes,
                } => match self.arena.write_chunk(id, reservation, offset, &bytes) {
                    Ok(()) => ClientReply::ChunkWritten,
                    Err(error) => ClientReply::Error(error),
                },
                ClientRequest::GetChunk { id, offset, len } => {
                    // Cross-host chunked get: serve one range of a committed
                    // object. Frame-bounded per chunk; the client reassembles
                    // and verifies the whole-object checksum.
                    if len > MAX_OBJECT_BYTES as u64 {
                        ClientReply::Error(Error::Protocol("get chunk too large".into()))
                    } else {
                        match self.arena.read_chunk(id, offset, len) {
                            Some(bytes) => ClientReply::Chunk(bytes),
                            None => ClientReply::Error(Error::Protocol(
                                "get chunk out of bounds or object not committed".into(),
                            )),
                        }
                    }
                }
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
                    Ok(()) => {
                        self.arena.release(id);
                        ClientReply::Released
                    }
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
                | ClientRequest::ArenaReserve { .. }
                | ClientRequest::ArenaCommit { .. }
                | ClientRequest::PutChunk { .. }
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

/// Ready once the poll returns anything but "no work yet".
fn worker_poll_ready(reply: &RpcReply) -> bool {
    !matches!(reply, RpcReply::Worker(WorkerReply::Assignment(None)))
}

/// Ready once the object resolves; a pending output keeps parking.
fn client_get_ready(reply: &RpcReply) -> bool {
    !matches!(
        reply,
        RpcReply::Client(ClientReply::Error(Error::ObjectPending(_)))
    )
}

/// Ready once at least `min_ready` batch slots have resolved (i.e. are no longer
/// pending). `min_ready == objects.len()` waits for all (`ray.get`); a smaller
/// value returns as soon as K finish (`ray.wait`).
fn client_get_batch_ready(reply: &RpcReply, min_ready: u32) -> bool {
    match reply {
        RpcReply::Client(ClientReply::ObjectBatch(results)) => {
            let ready = results
                .iter()
                .filter(|result| !matches!(result, Err(Error::ObjectPending(_))))
                .count();
            ready >= min_ready as usize
        }
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

/// Idle connections kept per address for reuse. Capped so a burst of clones
/// doesn't hoard file descriptors.
fn connection_pool() -> &'static std::sync::Mutex<HashMap<String, Vec<TcpStream>>> {
    static POOL: std::sync::OnceLock<std::sync::Mutex<HashMap<String, Vec<TcpStream>>>> =
        std::sync::OnceLock::new();
    POOL.get_or_init(Default::default)
}
const POOL_MAX_PER_ADDR: usize = 8;

pub async fn request(address: &str, envelope: &Envelope) -> Result<RpcReply, Error> {
    for attempt_number in 0..2 {
        let remaining_ms = envelope.deadline_unix_ms.saturating_sub(now_ms());
        if remaining_ms == 0 {
            return Err(Error::DeadlineExceeded);
        }
        // First attempt may reuse a pooled connection; a retry always dials
        // fresh, since a pooled stream failing usually means the server closed
        // it while idle (and its pool-mates are just as stale).
        let pooled = if attempt_number == 0 {
            connection_pool()
                .lock()
                .unwrap()
                .get_mut(address)
                .and_then(Vec::pop)
        } else {
            None
        };
        let attempt = tokio::time::timeout(Duration::from_millis(remaining_ms), async {
            let mut stream = match pooled {
                Some(stream) => stream,
                None => {
                    let stream = TcpStream::connect(address).await?;
                    let _ = stream.set_nodelay(true);
                    stream
                }
            };
            write_frame(&mut stream, envelope).await?;
            let reply = read_frame(&mut stream)
                .await?
                .ok_or_else(|| Error::Io("connection closed before reply".into()))?;
            let mut pool = connection_pool().lock().unwrap();
            let idle = pool.entry(address.to_string()).or_default();
            if idle.len() < POOL_MAX_PER_ADDR {
                idle.push(stream);
            }
            Ok(reply)
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
    // A gigabyte-scale hash is a full memory pass; fan it out across cores.
    if bytes.len() >= 1 << 20 {
        let mut hasher = blake3::Hasher::new();
        hasher.update_rayon(bytes);
        *hasher.finalize().as_bytes()
    } else {
        *blake3::hash(bytes).as_bytes()
    }
}
/// Completes on the first shutdown signal: SIGTERM or SIGINT on Unix, Ctrl-C
/// elsewhere. Drives the coordinator's graceful drain and the worker's.
pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(term) => term,
                // Cannot install SIGTERM (rare): fall back to Ctrl-C only.
                Err(_) => {
                    let _ = tokio::signal::ctrl_c().await;
                    return;
                }
            };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
#[path = "cluster_tests.rs"]
mod tests;
