use std::{collections::HashMap, marker::PhantomData, sync::Arc, time::Duration};

use memmap2::{Mmap, MmapMut};
use parking_lot::Mutex;

use serde::{de::DeserializeOwned, Serialize};

use crate::{
    cluster::{checksum, envelope, request},
    error::Error,
    ids::{ClusterId, CoordinatorEpoch, ObjectId, TaskId},
    operation::{Codec, Operation, TaskArg},
    protocol::{
        ClientReply, ClientRequest, ObjectPayload, RpcReply, RpcRequest, SubmitSpec, TaskStatus,
        TaskView, WorkerView, MAX_OBJECT_BYTES,
    },
    resources::ResourceSet,
};

#[derive(Clone)]
pub struct ClusterClient {
    address: String,
    cluster_id: ClusterId,
    coordinator_epoch: Option<CoordinatorEpoch>,
    /// Per-arena mappings, established once and shared across clones, so a
    /// same-host get reads the arena from RAM without remapping per object.
    arena_maps: Arc<Mutex<HashMap<String, Arc<Mmap>>>>,
    /// Writable mapping of the coordinator's arena, established at connect when
    /// the arena file is mappable (i.e. same host). Puts write payload bytes
    /// straight into it, so a put ships no bytes over the socket and is not
    /// bounded by the RPC frame size.
    arena_writer: Arc<Mutex<Option<MmapMut>>>,
}
impl ClusterClient {
    /// Connects using the default all-zero cluster id used by `crayon-cluster`.
    pub fn connect(address: impl Into<String>) -> Self {
        Self::connect_to(address, ClusterId([0; 16]))
    }
    pub fn connect_to(address: impl Into<String>, cluster_id: ClusterId) -> Self {
        Self {
            address: address.into(),
            cluster_id,
            coordinator_epoch: None,
            arena_maps: Arc::new(Mutex::new(HashMap::new())),
            arena_writer: Arc::new(Mutex::new(None)),
        }
    }
    /// Discovers the current coordinator epoch. Mutation requests after this
    /// call are fenced to the discovered epoch; a restarted coordinator rejects
    /// them with `Error::StaleEpoch`.
    pub async fn connect_epoch(&mut self) -> Result<CoordinatorEpoch, Error> {
        match self.rpc_raw(ClientRequest::Connect, None).await? {
            ClientReply::Connected {
                coordinator_epoch,
                arena_token,
            } => {
                self.coordinator_epoch = Some(coordinator_epoch);
                // Mappability of the arena file is the same-host proof: puts go
                // through shared memory when it maps, over TCP when it doesn't.
                *self.arena_writer.lock() = crate::arena::map_arena_mut(&arena_token);
                Ok(coordinator_epoch)
            }
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected connect reply".into())),
        }
    }
    async fn rpc(&self, body: ClientRequest) -> Result<ClientReply, Error> {
        self.rpc_raw(body, self.coordinator_epoch).await
    }
    async fn rpc_raw(
        &self,
        body: ClientRequest,
        coordinator_epoch: Option<CoordinatorEpoch>,
    ) -> Result<ClientReply, Error> {
        let mut envelope = envelope(self.cluster_id, RpcRequest::Client(body));
        envelope.coordinator_epoch = coordinator_epoch;
        match request(&self.address, &envelope).await? {
            RpcReply::Client(reply) => Ok(reply),
            _ => Err(Error::Protocol("unexpected worker reply".into())),
        }
    }
    /// Stores bytes at the coordinator. The coordinator owns this payload for
    /// the cluster lifetime; distributed reference counting is intentionally unsupported.
    pub async fn put<T: Serialize>(&self, value: &T) -> Result<ObjectRef<T>, Error> {
        let bytes = bincode::serialize(value)?;
        self.put_raw(Codec::BincodeV1, &bytes)
            .await
            .map(ObjectRef::new)
    }
    /// Stores raw bytes with no serialization envelope — the moral twin of
    /// `ray.put(bytes)`. Fetch with `get_bytes`.
    pub async fn put_bytes(&self, bytes: &[u8]) -> Result<ObjectId, Error> {
        self.put_raw(Codec::RawBytes, bytes).await
    }
    async fn put_raw(&self, codec: Codec, bytes: &[u8]) -> Result<ObjectId, Error> {
        if self.arena_writer.lock().is_some() {
            return self.put_arena(codec, bytes).await;
        }
        // Cross-host and larger than one frame: stream it in chunks. A single
        // Put frame is capped at MAX_OBJECT_BYTES, so this is the only way a
        // >8 MiB object crosses a host boundary.
        if bytes.len() > MAX_OBJECT_BYTES {
            return self.put_chunked(codec, bytes).await;
        }
        match self
            .rpc(ClientRequest::Put {
                codec,
                bytes: bytes.to_vec(),
            })
            .await?
        {
            ClientReply::Object(payload) => Ok(payload.id),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected put reply".into())),
        }
    }
    /// Cross-host large put: reserve an arena slot on the coordinator, stream the
    /// payload as frame-sized `PutChunk`s over TCP, then commit. Mirrors
    /// `put_arena` but the coordinator (not the client) writes into the arena,
    /// since a cross-host client cannot map it. `>=1 MiB` puts are unhashed
    /// (random id, all-zero checksum), matching the same-host path.
    async fn put_chunked(&self, codec: Codec, bytes: &[u8]) -> Result<ObjectId, Error> {
        let id = ObjectId::new();
        let reply = self
            .rpc(ClientRequest::ArenaReserve {
                id,
                codec,
                size_bytes: bytes.len() as u64,
                checksum: [0u8; 32], // unhashed: large objects skip the pass
            })
            .await?;
        let id = match reply {
            // Content already stored (dedup): nothing to stream.
            ClientReply::Object(payload) => return Ok(payload.id),
            ClientReply::ArenaReserved { id, .. } => id,
            ClientReply::Error(error) => return Err(error),
            _ => return Err(Error::Protocol("unexpected reserve reply".into())),
        };
        for (i, chunk) in bytes.chunks(MAX_OBJECT_BYTES).enumerate() {
            let offset = (i * MAX_OBJECT_BYTES) as u64;
            match self
                .rpc(ClientRequest::PutChunk {
                    id,
                    offset,
                    bytes: chunk.to_vec(),
                })
                .await?
            {
                ClientReply::ChunkWritten => {}
                ClientReply::Error(error) => return Err(error),
                _ => return Err(Error::Protocol("unexpected put chunk reply".into())),
            }
        }
        match self.rpc(ClientRequest::ArenaCommit(id)).await? {
            ClientReply::Object(payload) => Ok(payload.id),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected commit reply".into())),
        }
    }
    /// Same-host put: reserve an arena slot, write the bytes into shared memory
    /// at the granted offset, then commit. The payload never crosses a socket,
    /// so object size is bounded by the arena, not the RPC frame — gigabytes work.
    async fn put_arena(&self, codec: Codec, bytes: &[u8]) -> Result<ObjectId, Error> {
        // Below 1MB the hash is cheap and buys content-addressed dedup; past it
        // the pass costs real time (a full memory sweep), so large objects take
        // a random id and an all-zero "unhashed" checksum. Same-host reads are
        // straight out of shared memory either way — the payload never crosses
        // a lossy boundary. ponytail: no dedup for >=1MB puts; hash if it matters.
        let (id, checksum) = if bytes.len() < 1 << 20 {
            let checksum = checksum(bytes);
            (ObjectId::from_checksum(checksum), checksum)
        } else {
            (ObjectId::new(), [0u8; 32])
        };
        let reply = self
            .rpc(ClientRequest::ArenaReserve {
                id,
                codec,
                size_bytes: bytes.len() as u64,
                checksum,
            })
            .await?;
        match reply {
            // Content already stored: reserve resolved as a get.
            ClientReply::Object(payload) => Ok(payload.id),
            ClientReply::ArenaReserved { id, offset } => {
                {
                    let mut writer = self.arena_writer.lock();
                    let map = writer
                        .as_mut()
                        .ok_or_else(|| Error::Protocol("arena writer lost after reserve".into()))?;
                    let start = offset as usize;
                    let end = start
                        .checked_add(bytes.len())
                        .filter(|&end| end <= map.len())
                        .ok_or_else(|| Error::Protocol("arena offset out of bounds".into()))?;
                    crate::arena::copy_wide(&mut map[start..end], bytes);
                }
                match self.rpc(ClientRequest::ArenaCommit(id)).await? {
                    ClientReply::Object(payload) => Ok(payload.id),
                    ClientReply::Error(error) => Err(error),
                    _ => Err(Error::Protocol("unexpected commit reply".into())),
                }
            }
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected reserve reply".into())),
        }
    }
    pub async fn submit<A, O>(
        &self,
        operation: &Operation<A, O>,
        args: Vec<TaskArg>,
        resources: ResourceSet,
        max_attempts: u32,
    ) -> Result<TaskHandle<O>, Error> {
        operation.descriptor().validate_args(&args)?;
        match self
            .rpc(ClientRequest::Submit {
                operation: operation.descriptor().key.clone(),
                args,
                resources,
                max_attempts,
            })
            .await?
        {
            ClientReply::Submitted { task_id, output_id } => Ok(TaskHandle {
                task_id,
                output: ObjectRef::new(output_id),
                client: self.clone(),
            }),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected submit reply".into())),
        }
    }
    /// Submits many tasks of one operation in a single RPC — one round-trip for
    /// the whole batch instead of one per task. Each element is that task's
    /// arguments; the returned handles are in the same order. A spec the
    /// coordinator rejects yields an `Err` in that slot, leaving the rest admitted.
    pub async fn submit_batch<A, O>(
        &self,
        operation: &Operation<A, O>,
        batch: Vec<Vec<TaskArg>>,
        resources: ResourceSet,
        max_attempts: u32,
    ) -> Result<Vec<Result<TaskHandle<O>, Error>>, Error> {
        let mut specs = Vec::with_capacity(batch.len());
        for args in batch {
            operation.descriptor().validate_args(&args)?;
            specs.push(SubmitSpec {
                operation: operation.descriptor().key.clone(),
                args,
                resources: resources.clone(),
                max_attempts,
            });
        }
        match self.rpc(ClientRequest::SubmitBatch(specs)).await? {
            ClientReply::SubmittedBatch(results) => Ok(results
                .into_iter()
                .map(|result| {
                    result.map(|task| TaskHandle {
                        task_id: task.task_id,
                        output: ObjectRef::new(task.output_id),
                        client: self.clone(),
                    })
                })
                .collect()),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected submit batch reply".into())),
        }
    }
    /// Fetches many objects in one blocking RPC, waiting up to `timeout` for at
    /// least `min_ready` of them to resolve. `min_ready == ids.len()` is the
    /// all-or-nothing `ray.get([refs])`; a smaller value is `ray.wait`, returning
    /// as soon as K finish so the caller drains fast rollouts without blocking on
    /// the slowest. Results are in request order; unfinished slots are `ObjectPending`.
    async fn fetch_batch(
        &self,
        ids: &[ObjectId],
        min_ready: usize,
        timeout: Duration,
    ) -> Result<Vec<Result<ObjectPayload, Error>>, Error> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let wait_ms = remaining_wait_ms(deadline)?;
            match self
                .rpc(ClientRequest::GetBatch {
                    objects: ids.to_vec(),
                    wait_ms,
                    min_ready: min_ready as u32,
                })
                .await?
            {
                ClientReply::ObjectBatch(results) => {
                    // The server parks until `min_ready` slots resolve or the wait
                    // slice elapses; fewer ready means the slice expired, so retry
                    // until the outer deadline fires.
                    let ready = results
                        .iter()
                        .filter(|result| !matches!(result, Err(Error::ObjectPending(_))))
                        .count();
                    if ready < min_ready {
                        continue;
                    }
                    return Ok(results);
                }
                ClientReply::Error(error) => return Err(error),
                _ => return Err(Error::Protocol("unexpected get batch reply".into())),
            }
        }
    }
    pub async fn status(&self, task_id: TaskId) -> Result<TaskView, Error> {
        match self.rpc(ClientRequest::Status(task_id)).await? {
            ClientReply::Status(status) => Ok(status),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected status reply".into())),
        }
    }
    pub async fn workers(&self) -> Result<Vec<WorkerView>, Error> {
        match self.rpc(ClientRequest::Workers).await? {
            ClientReply::Workers(workers) => Ok(workers),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected workers reply".into())),
        }
    }
    pub async fn get_bytes(&self, id: ObjectId) -> Result<(Codec, Vec<u8>), Error> {
        self.fetch_object(id, 0).await
    }
    /// `get_bytes` that blocks up to `timeout` for a still-running task's
    /// reserved output — the raw-bytes twin of `TaskHandle::result`.
    pub async fn get_bytes_within(
        &self,
        id: ObjectId,
        timeout: Duration,
    ) -> Result<(Codec, Vec<u8>), Error> {
        self.fetch_object(id, timeout.as_millis() as u64).await
    }
    /// Raw-bytes twin of `results`: fetches many objects in one blocking batch,
    /// without decoding. One slot per requested id, in order.
    pub async fn get_bytes_many(
        &self,
        ids: &[ObjectId],
        timeout: Duration,
    ) -> Result<Vec<Result<(Codec, Vec<u8>), Error>>, Error> {
        self.wait_bytes_many(ids, ids.len(), timeout).await
    }
    /// Raw-bytes `ray.wait`: returns once `min_ready` of `ids` resolve, one slot
    /// per id in order. Ready slots carry bytes; unfinished ones are `ObjectPending`.
    pub async fn wait_bytes_many(
        &self,
        ids: &[ObjectId],
        min_ready: usize,
        timeout: Duration,
    ) -> Result<Vec<Result<(Codec, Vec<u8>), Error>>, Error> {
        let payloads = self
            .fetch_batch(ids, min_ready.min(ids.len()), timeout)
            .await?;
        let mut out = Vec::with_capacity(payloads.len());
        for payload in payloads {
            out.push(match payload {
                Ok(payload) => self.payload_bytes(payload).await,
                Err(error) => Err(error),
            });
        }
        Ok(out)
    }
    /// Fetches an object's bytes, blocking up to `wait_ms` for a reserved output
    /// to become available. Small outputs arrive inline; large ones redirect to
    /// the producing worker. `wait_ms: 0` returns immediately.
    async fn fetch_object(&self, id: ObjectId, wait_ms: u64) -> Result<(Codec, Vec<u8>), Error> {
        match self
            .rpc(ClientRequest::Get {
                object: id,
                wait_ms,
            })
            .await?
        {
            ClientReply::Object(payload) => self.payload_bytes(payload).await,
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected get reply".into())),
        }
    }
    /// Resolves a payload to its bytes: inline outputs are verified and returned;
    /// redirected (large) outputs are fetched from the producing worker and
    /// validated against the coordinator's checksum/size. Shared by single and
    /// batch fetch paths.
    /// Returns the cached mapping for arena `token`, establishing it on first
    /// use. `None` when the arena is not on this host (map fails).
    fn arena_map(&self, token: &str) -> Option<Arc<Mmap>> {
        let mut cache = self.arena_maps.lock();
        if let Some(map) = cache.get(token) {
            return Some(map.clone());
        }
        let map = Arc::new(crate::arena::map_arena(token)?);
        cache.insert(token.to_string(), map.clone());
        Some(map)
    }
    async fn payload_bytes(&self, payload: ObjectPayload) -> Result<(Codec, Vec<u8>), Error> {
        let ObjectPayload {
            id,
            codec,
            size_bytes,
            checksum,
            location,
            bytes,
            arena,
        } = payload;
        if let Some(bytes) = bytes {
            verify_object(id, id, &bytes, checksum, size_bytes)?;
            return Ok((codec, bytes.to_vec()));
        }
        // Same-host zero-copy: read the payload straight out of the arena
        // mapping (established once, then cached), skipping serialize + socket.
        // Falls through when the arena is unmappable (a different host), which
        // is proof to refetch over the network.
        if let Some(arena_ref) = &arena {
            if let Some(map) = self.arena_map(&arena_ref.token) {
                let start = arena_ref.offset as usize;
                let end = start.saturating_add(size_bytes as usize);
                if end <= map.len() {
                    // Arena objects are content-addressed (id == blake3(bytes))
                    // and immutable, and the coordinator validated on put, so we
                    // trust the offset and skip re-hashing (as plasma does). Only
                    // the offset bounds, checked above, need guarding.
                    let _ = checksum;
                    return Ok((codec, crate::arena::to_vec_wide(&map[start..end])));
                }
            }
        }
        // Cross-host fetch. An arena-backed object lives at the coordinator
        // (self.address); a worker output lives at `location`.
        let source = if arena.is_some() {
            self.address.clone()
        } else {
            location
        };
        // Larger than one frame: stream it in chunks. Only arena objects can
        // exceed MAX_OBJECT_BYTES (worker outputs are capped at admission), and
        // GetLocal would reject them, so this is the only cross-host large-get path.
        if arena.is_some() && size_bytes > MAX_OBJECT_BYTES as u64 {
            let bytes = self.get_chunked(id, size_bytes, &source).await?;
            verify_object(id, id, &bytes, checksum, size_bytes)?;
            return Ok((codec, bytes));
        }
        let reply = request(
            &source,
            &envelope(
                self.cluster_id,
                RpcRequest::Client(ClientRequest::GetLocal(id)),
            ),
        )
        .await?;
        match reply {
            RpcReply::Client(ClientReply::Object(ObjectPayload {
                id: local_id,
                codec: local_codec,
                bytes: Some(bytes),
                checksum: actual,
                size_bytes: local_size,
                ..
            })) if local_codec == codec && actual == checksum && local_size == size_bytes => {
                verify_object(id, local_id, &bytes, checksum, size_bytes)?;
                Ok((codec, bytes.to_vec()))
            }
            _ => Err(Error::ObjectConflict(id)),
        }
    }
    /// Cross-host large get: pull a committed arena object as frame-sized
    /// `GetChunk`s from `source` and reassemble. The caller verifies the
    /// whole-object checksum, so no per-chunk hashing here.
    async fn get_chunked(
        &self,
        id: ObjectId,
        size_bytes: u64,
        source: &str,
    ) -> Result<Vec<u8>, Error> {
        let mut out = Vec::with_capacity(size_bytes as usize);
        let mut offset = 0u64;
        while offset < size_bytes {
            let len = (size_bytes - offset).min(MAX_OBJECT_BYTES as u64);
            let reply = request(
                source,
                &envelope(
                    self.cluster_id,
                    RpcRequest::Client(ClientRequest::GetChunk { id, offset, len }),
                ),
            )
            .await?;
            match reply {
                RpcReply::Client(ClientReply::Chunk(bytes)) if bytes.len() as u64 == len => {
                    out.extend_from_slice(&bytes);
                    offset += len;
                }
                RpcReply::Client(ClientReply::Error(error)) => return Err(error),
                _ => return Err(Error::ObjectConflict(id)),
            }
        }
        Ok(out)
    }

    pub async fn get<T: DeserializeOwned>(&self, reference: &ObjectRef<T>) -> Result<T, Error> {
        let (codec, bytes) = self.get_bytes(reference.id).await?;
        decode(codec, bytes)
    }
    /// Awaits many task outputs in one blocking RPC and decodes each — the batch
    /// analogue of `TaskHandle::result`, matching `ray.get([refs])`. Results are
    /// in `handles` order; a failed/cancelled task yields its precise error in
    /// that slot without sinking the batch. Redirected large outputs are fetched
    /// per-object after the batch resolves.
    pub async fn results<T: DeserializeOwned>(
        &self,
        handles: &[TaskHandle<T>],
        timeout: Duration,
    ) -> Result<Vec<Result<T, Error>>, Error> {
        let ids: Vec<ObjectId> = handles.iter().map(|handle| handle.output.id).collect();
        let payloads = self.fetch_batch(&ids, ids.len(), timeout).await?;
        let mut out = Vec::with_capacity(payloads.len());
        for (payload, handle) in payloads.into_iter().zip(handles) {
            out.push(match payload {
                Ok(payload) => self
                    .payload_bytes(payload)
                    .await
                    .and_then(|(codec, bytes)| decode(codec, bytes)),
                // non-available terminal: fetch the precise failure/cancel reason
                Err(Error::Protocol(_)) => Err(handle.terminal_error().await),
                Err(error) => Err(error),
            });
        }
        Ok(out)
    }
    /// Awaits *at least* `min_ready` of `handles` to finish (`ray.wait`), returning
    /// one entry per handle in order: `Ready::Done(i, result)` for resolved slots,
    /// `Ready::Pending(i)` for the rest. An RL driver loops — `wait` for K, train on
    /// the ready trajectories, then `wait` again on the stragglers — instead of
    /// blocking the whole step on the slowest rollout.
    pub async fn wait<T: DeserializeOwned>(
        &self,
        handles: &[TaskHandle<T>],
        min_ready: usize,
        timeout: Duration,
    ) -> Result<Vec<Ready<T>>, Error> {
        let ids: Vec<ObjectId> = handles.iter().map(|handle| handle.output.id).collect();
        let min_ready = min_ready.min(ids.len());
        let payloads = self.fetch_batch(&ids, min_ready, timeout).await?;
        let mut out = Vec::with_capacity(payloads.len());
        for (index, (payload, handle)) in payloads.into_iter().zip(handles).enumerate() {
            out.push(match payload {
                Ok(payload) => Ready::Done(
                    index,
                    self.payload_bytes(payload)
                        .await
                        .and_then(|(codec, bytes)| decode(codec, bytes)),
                ),
                Err(Error::ObjectPending(_)) => Ready::Pending(index),
                // non-available terminal: fetch the precise failure/cancel reason
                Err(Error::Protocol(_)) => Ready::Done(index, Err(handle.terminal_error().await)),
                Err(error) => Ready::Done(index, Err(error)),
            });
        }
        Ok(out)
    }
    pub async fn cancel(&self, task_id: TaskId) -> Result<(), Error> {
        match self.rpc(ClientRequest::Cancel(task_id)).await? {
            ClientReply::Cancelled => Ok(()),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected cancel reply".into())),
        }
    }
    pub async fn release(&self, id: ObjectId) -> Result<(), Error> {
        match self.rpc(ClientRequest::Release(id)).await? {
            ClientReply::Released => Ok(()),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected release reply".into())),
        }
    }
}

/// Milliseconds left until `deadline` for a blocking long-poll RPC, floored at 1
/// while any time remains so a sub-millisecond remainder still parks server-side
/// instead of busy-spinning. `Err(DeadlineExceeded)` once the deadline passes.
fn remaining_wait_ms(deadline: tokio::time::Instant) -> Result<u64, Error> {
    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    if remaining.is_zero() {
        return Err(Error::DeadlineExceeded);
    }
    Ok((remaining.as_millis().min(u64::MAX as u128) as u64).max(1))
}

fn decode<T: DeserializeOwned>(codec: Codec, bytes: Vec<u8>) -> Result<T, Error> {
    match codec {
        Codec::BincodeV1 => Ok(bincode::deserialize(&bytes)?),
        _ => Err(Error::Protocol(format!(
            "unsupported codec for get: {codec:?}"
        ))),
    }
}

fn verify_object(
    requested: ObjectId,
    returned: ObjectId,
    bytes: &[u8],
    expected: [u8; 32],
    size: u64,
) -> Result<(), Error> {
    if requested != returned || bytes.len() as u64 != size {
        return Err(Error::ObjectConflict(requested));
    }
    // All-zero means the object was stored unhashed; size is the only check.
    if expected != [0u8; 32] && checksum(bytes) != expected {
        return Err(Error::ObjectConflict(requested));
    }
    Ok(())
}

/// One slot of a `ClusterClient::wait` result, tagged with its position in the
/// input `handles`. `Done` carries the decoded result (or that task's error);
/// `Pending` marks a rollout that had not finished when `min_ready` was met.
pub enum Ready<T> {
    Done(usize, Result<T, Error>),
    Pending(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ObjectRef<T> {
    pub id: ObjectId,
    marker: PhantomData<T>,
}
impl<T> ObjectRef<T> {
    pub fn new(id: ObjectId) -> Self {
        Self {
            id,
            marker: PhantomData,
        }
    }
}

pub struct TaskHandle<T> {
    pub task_id: TaskId,
    pub output: ObjectRef<T>,
    client: ClusterClient,
}
impl<T: DeserializeOwned> TaskHandle<T> {
    /// Waits for the task's output, blocking on the coordinator rather than
    /// polling. Each request parks server-side until the reserved output resolves
    /// or the wait slice elapses, then returns the result — inline for small
    /// outputs, one hop to the producing worker for large ones.
    pub async fn result(&self, timeout: Duration) -> Result<T, Error> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let wait_ms = remaining_wait_ms(deadline)?;
            match self.client.fetch_object(self.output.id, wait_ms).await {
                Ok((codec, bytes)) => return decode(codec, bytes),
                // Held the full slice and the output is still reserved: re-issue
                // until the outer deadline fires.
                Err(Error::ObjectPending(_)) => continue,
                // Output resolved to a non-available terminal (failed/cancelled).
                // Consult task status once for the precise reason.
                Err(Error::Protocol(_)) => return Err(self.terminal_error().await),
                Err(error) => return Err(error),
            }
        }
    }
    /// Resolves a failed/cancelled output to a precise error via a single status
    /// lookup. Only taken on the rare failure path, so the extra hop is cheap.
    async fn terminal_error(&self) -> Error {
        match self.client.status(self.task_id).await {
            Ok(view) => match view.state {
                TaskStatus::Failed(message) => Error::TaskFailed(self.task_id, message),
                TaskStatus::Cancelled => Error::TaskCancelled(self.task_id),
                _ => Error::ObjectConflict(self.output.id),
            },
            Err(error) => error,
        }
    }
    /// Requests cooperative cancellation and waits only for coordinator acceptance.
    /// Operations must return for the worker to acknowledge cancellation; hard process
    /// termination is not supported by this in-process worker implementation.
    pub async fn cancel(&self) -> Result<(), Error> {
        self.client.cancel(self.task_id).await
    }
}
