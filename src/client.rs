use std::{marker::PhantomData, time::Duration};

use serde::{de::DeserializeOwned, Serialize};

use crate::{
    cluster::{checksum, envelope, request},
    error::{DeadlineContext, Error},
    ids::{ClusterId, ObjectId, TaskId},
    operation::{Codec, Operation, TaskArg},
    protocol::{ClientReply, ClientRequest, RpcReply, RpcRequest, TaskView, WorkerView},
    resources::ResourceSet,
};

#[derive(Clone)]
pub struct ClusterClient {
    address: String,
    cluster_id: ClusterId,
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
        }
    }
    async fn rpc(&self, body: ClientRequest) -> Result<ClientReply, Error> {
        match request(
            &self.address,
            &envelope(self.cluster_id, RpcRequest::Client(body)),
        )
        .await?
        {
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
            ClientReply::Object { id, .. } => Ok(ObjectRef::new(id)),
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
        match self.rpc(ClientRequest::Get(id)).await? {
            ClientReply::Object {
                id: returned,
                codec,
                bytes: Some(bytes),
                checksum,
                size_bytes,
                ..
            } => {
                verify_object(id, returned, &bytes, checksum, size_bytes)?;
                Ok((codec, bytes))
            }
            ClientReply::Object {
                id: returned,
                codec,
                location,
                checksum,
                size_bytes,
                ..
            } => {
                let reply = request(
                    &location,
                    &envelope(
                        self.cluster_id,
                        RpcRequest::Client(ClientRequest::GetLocal(id)),
                    ),
                )
                .await?;
                match reply {
                    RpcReply::Client(ClientReply::Object {
                        id: local_id,
                        codec: local_codec,
                        bytes: Some(bytes),
                        checksum: actual,
                        size_bytes: local_size,
                        ..
                    }) if returned == id
                        && local_codec == codec
                        && actual == checksum
                        && local_size == size_bytes =>
                    {
                        verify_object(id, local_id, &bytes, checksum, size_bytes)?;
                        Ok((codec, bytes))
                    }
                    _ => Err(Error::ObjectConflict(id)),
                }
            }
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected get reply".into())),
        }
    }

    pub async fn get<T: DeserializeOwned>(&self, reference: &ObjectRef<T>) -> Result<T, Error> {
        let (codec, bytes) = self.get_bytes(reference.id).await?;
        if codec != Codec::BincodeV1 {
            return Err(Error::ObjectConflict(reference.id));
        }
        Ok(bincode::deserialize(&bytes)?)
    }
    pub async fn cancel(&self, task_id: TaskId) -> Result<(), Error> {
        match self.rpc(ClientRequest::Cancel(task_id)).await? {
            ClientReply::Cancelled => Ok(()),
            ClientReply::Error(error) => Err(error),
            _ => Err(Error::Protocol("unexpected cancel reply".into())),
        }
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
    pub async fn result(&self, timeout: Duration) -> Result<T, Error> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return Err(Error::DeadlineExceeded(DeadlineContext::TaskResult));
            }
            let remaining = deadline - now;
            match tokio::time::timeout(remaining, self.client.get(&self.output)).await {
                Ok(Ok(value)) => return Ok(value),
                Ok(Err(Error::Protocol(message))) if message.contains("unavailable") => {
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    if remaining.is_zero() {
                        return Err(Error::DeadlineExceeded(DeadlineContext::TaskResult));
                    }
                    tokio::time::sleep(remaining.min(Duration::from_millis(25))).await;
                }
                Ok(Err(error)) => return Err(error),
                Err(_) => return Err(Error::DeadlineExceeded(DeadlineContext::TaskResult)),
            }
        }
    }
    /// Requests cooperative cancellation and waits only for coordinator acceptance.
    /// Operations must return for the worker to acknowledge cancellation; hard process
    /// termination is not supported by this in-process worker implementation.
    pub async fn cancel(&self) -> Result<(), Error> {
        self.client.cancel(self.task_id).await
    }
}
