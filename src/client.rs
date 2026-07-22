use std::{marker::PhantomData, time::Duration};

use serde::{de::DeserializeOwned, Serialize};

use crate::{
    cluster::{checksum, envelope, request},
    error::Error,
    ids::{ClusterId, ObjectId, TaskId},
    operation::{Codec, Operation, TaskArg},
    protocol::{ClientReply, ClientRequest, RpcReply, RpcRequest},
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
            ClientReply::Error(message) => Err(Error::Protocol(message)),
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
            ClientReply::Error(message) => Err(Error::Protocol(message)),
            _ => Err(Error::Protocol("unexpected submit reply".into())),
        }
    }
    pub async fn status(&self, task_id: TaskId) -> Result<String, Error> {
        match self.rpc(ClientRequest::Status(task_id)).await? {
            ClientReply::Status(status) => Ok(status),
            ClientReply::Error(message) => Err(Error::Protocol(message)),
            _ => Err(Error::Protocol("unexpected status reply".into())),
        }
    }
    pub async fn get<T: DeserializeOwned>(&self, reference: &ObjectRef<T>) -> Result<T, Error> {
        match self.rpc(ClientRequest::Get(reference.id)).await? {
            ClientReply::Object {
                id,
                codec,
                bytes: Some(bytes),
                checksum: expected,
                size_bytes,
                ..
            } => decode(reference.id, id, codec, bytes, expected, size_bytes),
            ClientReply::Object {
                id,
                codec,
                location,
                checksum: expected,
                size_bytes,
                ..
            } => {
                let reply = request(
                    &location,
                    &envelope(
                        self.cluster_id,
                        RpcRequest::Client(ClientRequest::GetLocal(reference.id)),
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
                    }) if id == reference.id
                        && local_id == reference.id
                        && codec == local_codec
                        && actual == expected
                        && local_size == size_bytes =>
                    {
                        decode(
                            reference.id,
                            local_id,
                            local_codec,
                            bytes,
                            expected,
                            size_bytes,
                        )
                    }
                    _ => Err(Error::ObjectConflict(reference.id)),
                }
            }
            ClientReply::Error(message) => Err(Error::Protocol(message)),
            _ => Err(Error::Protocol("unexpected get reply".into())),
        }
    }
    pub async fn cancel(&self, task_id: TaskId) -> Result<(), Error> {
        match self.rpc(ClientRequest::Cancel(task_id)).await? {
            ClientReply::Cancelled => Ok(()),
            ClientReply::Error(message) => Err(Error::Protocol(message)),
            _ => Err(Error::Protocol("unexpected cancel reply".into())),
        }
    }
}

fn decode<T: DeserializeOwned>(
    requested: ObjectId,
    returned: ObjectId,
    codec: Codec,
    bytes: Vec<u8>,
    expected: [u8; 32],
    size: u64,
) -> Result<T, Error> {
    if requested != returned
        || codec != Codec::BincodeV1
        || bytes.len() as u64 != size
        || checksum(&bytes) != expected
    {
        return Err(Error::ObjectConflict(requested));
    }
    Ok(bincode::deserialize(&bytes)?)
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
                return Err(Error::DeadlineExceeded("task result"));
            }
            let remaining = deadline - now;
            match tokio::time::timeout(remaining, self.client.get(&self.output)).await {
                Ok(Ok(value)) => return Ok(value),
                Ok(Err(Error::Protocol(message))) if message.contains("unavailable") => {
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    if remaining.is_zero() {
                        return Err(Error::DeadlineExceeded("task result"));
                    }
                    tokio::time::sleep(remaining.min(Duration::from_millis(25))).await;
                }
                Ok(Err(error)) => return Err(error),
                Err(_) => return Err(Error::DeadlineExceeded("task result")),
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
