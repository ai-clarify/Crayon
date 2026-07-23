use std::{marker::PhantomData, time::Duration};

use serde::{de::DeserializeOwned, Serialize};

use crate::{
    cluster::{checksum, envelope, request},
    error::Error,
    ids::{ClusterId, CoordinatorEpoch, ObjectId, TaskId},
    operation::{Codec, Operation, TaskArg},
    protocol::{
        ClientReply, ClientRequest, ObjectPayload, RpcReply, RpcRequest, SubmitSpec, TaskStatus,
        TaskView, WorkerView,
    },
    resources::ResourceSet,
};

#[derive(Clone)]
pub struct ClusterClient {
    address: String,
    cluster_id: ClusterId,
    coordinator_epoch: Option<CoordinatorEpoch>,
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
        }
    }
    /// Discovers the current coordinator epoch. Mutation requests after this
    /// call are fenced to the discovered epoch; a restarted coordinator rejects
    /// them with `Error::StaleEpoch`.
    pub async fn connect_epoch(&mut self) -> Result<CoordinatorEpoch, Error> {
        match self.rpc_raw(ClientRequest::Connect, None).await? {
            ClientReply::Connected { coordinator_epoch } => {
                self.coordinator_epoch = Some(coordinator_epoch);
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
        match self
            .rpc(ClientRequest::Put {
                codec: Codec::BincodeV1,
                bytes,
            })
            .await?
        {
            ClientReply::Object(payload) => Ok(ObjectRef::new(payload.id)),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected put reply".into())),
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
    /// Fetches many objects in a single blocking RPC, waiting up to `timeout` for
    /// all of them to resolve — the batch analogue of `TaskHandle::result` and a
    /// direct parallel to `ray.get([refs])`. Results are in request order.
    async fn fetch_batch(
        &self,
        ids: &[ObjectId],
        timeout: Duration,
    ) -> Result<Vec<Result<ObjectPayload, Error>>, Error> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let wait_ms = remaining_wait_ms(deadline)?;
            match self
                .rpc(ClientRequest::GetBatch {
                    objects: ids.to_vec(),
                    wait_ms,
                })
                .await?
            {
                ClientReply::ObjectBatch(results) => {
                    // The server parks until every object resolves or the wait
                    // slice elapses; a still-pending slot means the slice expired,
                    // so retry until the outer deadline fires.
                    if results
                        .iter()
                        .any(|result| matches!(result, Err(Error::ObjectPending(_))))
                    {
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
    async fn payload_bytes(&self, payload: ObjectPayload) -> Result<(Codec, Vec<u8>), Error> {
        let ObjectPayload {
            id,
            codec,
            size_bytes,
            checksum,
            location,
            bytes,
        } = payload;
        if let Some(bytes) = bytes {
            verify_object(id, id, &bytes, checksum, size_bytes)?;
            return Ok((codec, bytes.to_vec()));
        }
        let reply = request(
            &location,
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
        let payloads = self.fetch_batch(&ids, timeout).await?;
        let mut out = Vec::with_capacity(payloads.len());
        for (payload, handle) in payloads.into_iter().zip(handles) {
            out.push(match payload {
                Ok(payload) => match self.payload_bytes(payload).await {
                    Ok((codec, bytes)) => decode(codec, bytes),
                    Err(error) => Err(error),
                },
                // Object resolved to a non-available terminal; get the precise
                // failure/cancellation reason for this task.
                Err(Error::Protocol(_)) => Err(handle.terminal_error().await),
                Err(error) => Err(error),
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
    if requested != returned || bytes.len() as u64 != size || checksum(bytes) != expected {
        return Err(Error::ObjectConflict(requested));
    }
    Ok(())
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
