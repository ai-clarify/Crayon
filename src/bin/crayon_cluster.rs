use std::{sync::Arc, time::Duration};

use crayon::{
    cluster::{checksum, envelope, now_ms, read_frame, request, write_frame, CoordinatorServer},
    data_plane::LocalObjectStore,
    error::Error,
    ids::{ClusterId, NodeId, ObjectId, TaskId, WorkerEpoch},
    operation::{Codec, OperationDescriptor, OperationKey, TaskArg},
    protocol::{
        ClientReply, ClientRequest, Envelope, RegisterWorker, RpcReply, RpcRequest, TaskAssignment,
        TaskCompletion, WorkerIdentity, WorkerReply, WorkerRequest,
    },
    resources::ResourceSet,
    worker::OperationRegistry,
};
use tokio::sync::Semaphore;

const CLUSTER_ID: ClusterId = ClusterId([0; 16]);
const OBJECT_CONNECTIONS: usize = 128;

fn usage() -> ! {
    eprintln!("usage: crayon-cluster coordinator <addr> | worker <coordinator> <advertise> | submit <coordinator> <a> <b>");
    std::process::exit(2)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("coordinator") => {
            CoordinatorServer::new(CLUSTER_ID, 5_000)
                .serve(args.get(2).unwrap_or_else(|| usage()))
                .await?
        }
        Some("worker") => {
            run_worker(
                args.get(2).unwrap_or_else(|| usage()),
                args.get(3).unwrap_or_else(|| usage()),
            )
            .await?
        }
        Some("submit") => {
            let coordinator = args.get(2).unwrap_or_else(|| usage());
            let a: i64 = args.get(3).unwrap_or_else(|| usage()).parse()?;
            let b: i64 = args.get(4).unwrap_or_else(|| usage()).parse()?;
            run_client(coordinator, a, b).await?;
        }
        _ => usage(),
    }
    Ok(())
}

fn add_descriptor() -> OperationDescriptor {
    OperationDescriptor {
        key: OperationKey::new("builtin", "add", 1),
        input_codec: Codec::BincodeV1,
        output_codec: Codec::BincodeV1,
        max_inline_arg_bytes: 1024,
    }
}

async fn run_worker(coordinator: &str, advertise: &str) -> Result<(), Error> {
    let objects = LocalObjectStore::default();
    serve_objects(advertise, objects.clone()).await?;
    let mut registry = OperationRegistry::default();
    registry.register(add_descriptor(), |args| async move {
        if args.len() != 2 {
            return Err(Error::Protocol("add expects two arguments".into()));
        }
        let a: i64 = bincode::deserialize(&args[0])?;
        let b: i64 = bincode::deserialize(&args[1])?;
        Ok(bincode::serialize(&(a + b))?)
    })?;
    let registry = Arc::new(registry);
    let registration = RegisterWorker {
        node_id: NodeId::new(),
        worker_epoch: WorkerEpoch::new(),
        advertise_addr: advertise.into(),
        resources: ResourceSet::cpu_gpu(1.0, 0.0)?,
        slots: 1,
        operations: registry.descriptors(),
    };
    let registered = match rpc(
        coordinator,
        RpcRequest::Worker(WorkerRequest::Register(registration)),
    )
    .await?
    {
        RpcReply::Worker(WorkerReply::Registered(value)) => value,
        other => return Err(Error::Protocol(format!("registration failed: {other:?}"))),
    };
    let identity = registered.identity;
    let heartbeat_identity = identity.clone();
    let heartbeat_coordinator = coordinator.to_string();
    let heartbeat_period = Duration::from_millis((registered.lease_timeout_ms / 3).max(10));
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(heartbeat_period);
        loop {
            interval.tick().await;
            let reply = rpc(
                &heartbeat_coordinator,
                RpcRequest::Worker(WorkerRequest::Heartbeat(heartbeat_identity.clone())),
            )
            .await;
            if !matches!(reply, Ok(RpcReply::Worker(WorkerReply::Accepted))) {
                break;
            }
        }
    });
    loop {
        match rpc(
            coordinator,
            RpcRequest::Worker(WorkerRequest::Poll(identity.clone())),
        )
        .await?
        {
            RpcReply::Worker(WorkerReply::Assignment(Some(assignment))) => {
                execute_assignment(
                    coordinator,
                    advertise,
                    &identity,
                    assignment,
                    registry.clone(),
                    objects.clone(),
                )
                .await?;
            }
            RpcReply::Worker(WorkerReply::Assignment(None)) => {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            RpcReply::Worker(WorkerReply::Cancel(fence)) => {
                accept(
                    coordinator,
                    WorkerRequest::Cancelled {
                        identity: identity.clone(),
                        fence,
                    },
                )
                .await?;
            }
            RpcReply::Worker(WorkerReply::Error(message)) => return Err(Error::Protocol(message)),
            _ => return Err(Error::Protocol("unexpected poll reply".into())),
        }
    }
}

async fn execute_assignment(
    coordinator: &str,
    advertise: &str,
    identity: &WorkerIdentity,
    assignment: TaskAssignment,
    registry: Arc<OperationRegistry>,
    objects: LocalObjectStore,
) -> Result<(), Error> {
    accept(
        coordinator,
        WorkerRequest::Started {
            identity: identity.clone(),
            fence: assignment.fence,
        },
    )
    .await?;
    let inputs = match fetch_inputs(coordinator, &assignment).await {
        Ok(inputs) => inputs,
        Err(error) => {
            accept(
                coordinator,
                WorkerRequest::Failed {
                    identity: identity.clone(),
                    fence: assignment.fence,
                    message: error.to_string(),
                    retryable: true,
                },
            )
            .await?;
            return Ok(());
        }
    };
    let operation = assignment.clone();
    let execution_registry = registry.clone();
    let mut execution =
        Box::pin(async move { execution_registry.execute(&operation, inputs).await });
    let mut poll = tokio::time::interval(Duration::from_millis(50));
    let result = loop {
        tokio::select! {
            result = &mut execution => break Some(result),
            _ = poll.tick() => {
                match rpc(coordinator, RpcRequest::Worker(WorkerRequest::Poll(identity.clone()))).await? {
                    RpcReply::Worker(WorkerReply::Cancel(fence)) if fence == assignment.fence => break None,
                    RpcReply::Worker(WorkerReply::Assignment(None)) => {}
                    RpcReply::Worker(WorkerReply::Error(message)) => return Err(Error::Protocol(message)),
                    _ => return Err(Error::Protocol("worker received assignment while busy".into())),
                }
            }
        }
    };
    let Some(result) = result else {
        accept(
            coordinator,
            WorkerRequest::Cancelled {
                identity: identity.clone(),
                fence: assignment.fence,
            },
        )
        .await?;
        return Ok(());
    };
    match result {
        Ok(bytes) => {
            let descriptor = registry
                .descriptors()
                .into_iter()
                .find(|descriptor| descriptor.key == assignment.operation)
                .ok_or_else(|| Error::OperationUnavailable(assignment.operation.to_string()))?;
            let object = objects.put(assignment.output_id, descriptor.output_codec, bytes)?;
            accept(
                coordinator,
                WorkerRequest::Completed {
                    identity: identity.clone(),
                    report: TaskCompletion {
                        fence: assignment.fence,
                        output_id: assignment.output_id,
                        codec: object.codec,
                        size_bytes: object.bytes.len() as u64,
                        checksum: object.checksum,
                        location: advertise.to_string(),
                    },
                },
            )
            .await
        }
        Err(error) => {
            accept(
                coordinator,
                WorkerRequest::Failed {
                    identity: identity.clone(),
                    fence: assignment.fence,
                    message: error.to_string(),
                    retryable: true,
                },
            )
            .await
        }
    }
}

async fn fetch_inputs(
    coordinator: &str,
    assignment: &TaskAssignment,
) -> Result<Vec<Vec<u8>>, Error> {
    let mut inputs = Vec::with_capacity(assignment.args.len());
    for arg in &assignment.args {
        match arg {
            TaskArg::Inline { bytes, .. } => inputs.push(bytes.clone()),
            TaskArg::Object(id) => {
                match rpc(coordinator, RpcRequest::Client(ClientRequest::Get(*id))).await? {
                    RpcReply::Client(ClientReply::Object {
                        id: returned,
                        codec,
                        bytes: Some(bytes),
                        checksum: expected,
                        size_bytes,
                        ..
                    }) => {
                        verify_object(*id, returned, &bytes, expected, size_bytes)?;
                        let _ = codec;
                        inputs.push(bytes);
                    }
                    RpcReply::Client(ClientReply::Object {
                        id: returned,
                        location,
                        checksum: expected,
                        size_bytes,
                        codec,
                        ..
                    }) => {
                        let reply =
                            rpc(&location, RpcRequest::Client(ClientRequest::GetLocal(*id)))
                                .await?;
                        match reply {
                            RpcReply::Client(ClientReply::Object {
                                id: local_id,
                                codec: local_codec,
                                bytes: Some(bytes),
                                checksum: actual,
                                size_bytes: local_size,
                                ..
                            }) if returned == *id
                                && local_codec == codec
                                && actual == expected
                                && local_size == size_bytes =>
                            {
                                verify_object(*id, local_id, &bytes, expected, size_bytes)?;
                                inputs.push(bytes);
                            }
                            _ => return Err(Error::ObjectConflict(*id)),
                        }
                    }
                    _ => return Err(Error::ObjectNotFound(*id)),
                }
            }
        }
    }
    Ok(inputs)
}

fn verify_object(
    requested: ObjectId,
    returned: ObjectId,
    bytes: &[u8],
    expected: [u8; 32],
    size: u64,
) -> Result<(), Error> {
    if returned != requested || bytes.len() as u64 != size || checksum(bytes) != expected {
        return Err(Error::ObjectConflict(requested));
    }
    Ok(())
}

async fn serve_objects(advertise: &str, objects: LocalObjectStore) -> Result<(), Error> {
    let listener = tokio::net::TcpListener::bind(advertise).await?;
    let permits = Arc::new(Semaphore::new(OBJECT_CONNECTIONS));
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let Ok(permit) = permits.clone().try_acquire_owned() else {
                continue;
            };
            let objects = objects.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let reply = match tokio::time::timeout(
                    Duration::from_secs(5),
                    read_frame::<Envelope>(&mut stream),
                )
                .await
                {
                    Ok(Ok(Some(envelope))) if envelope.validate(CLUSTER_ID, now_ms()).is_ok() => {
                        match envelope.body {
                            RpcRequest::Client(ClientRequest::GetLocal(id)) => {
                                match objects.get(id) {
                                    Ok(object) => RpcReply::Client(ClientReply::Object {
                                        id,
                                        codec: object.codec,
                                        size_bytes: object.bytes.len() as u64,
                                        checksum: object.checksum,
                                        location: String::new(),
                                        bytes: Some(object.bytes),
                                    }),
                                    Err(error) => {
                                        RpcReply::Client(ClientReply::Error(error.to_string()))
                                    }
                                }
                            }
                            _ => RpcReply::Client(ClientReply::Error(
                                "invalid object request".into(),
                            )),
                        }
                    }
                    _ => RpcReply::Client(ClientReply::Error("invalid object request".into())),
                };
                let _ =
                    tokio::time::timeout(Duration::from_secs(5), write_frame(&mut stream, &reply))
                        .await;
            });
        }
    });
    Ok(())
}

async fn rpc(address: &str, body: RpcRequest) -> Result<RpcReply, Error> {
    request(address, &envelope(CLUSTER_ID, body)).await
}
async fn accept(coordinator: &str, request_body: WorkerRequest) -> Result<(), Error> {
    match rpc(coordinator, RpcRequest::Worker(request_body)).await? {
        RpcReply::Worker(WorkerReply::Accepted) => Ok(()),
        RpcReply::Worker(WorkerReply::Error(message)) => Err(Error::Protocol(message)),
        _ => Err(Error::Protocol(
            "coordinator did not accept worker report".into(),
        )),
    }
}

async fn run_client(coordinator: &str, a: i64, b: i64) -> Result<(), Error> {
    let args = vec![
        TaskArg::Inline {
            codec: Codec::BincodeV1,
            bytes: bincode::serialize(&a)?,
        },
        TaskArg::Inline {
            codec: Codec::BincodeV1,
            bytes: bincode::serialize(&b)?,
        },
    ];
    let (task_id, output_id) = match rpc(
        coordinator,
        RpcRequest::Client(ClientRequest::Submit {
            operation: add_descriptor().key,
            args,
            resources: ResourceSet::cpu_gpu(1.0, 0.0)?,
            max_attempts: 1,
        }),
    )
    .await?
    {
        RpcReply::Client(ClientReply::Submitted { task_id, output_id }) => (task_id, output_id),
        other => return Err(Error::Protocol(format!("submit failed: {other:?}"))),
    };
    wait_result(coordinator, task_id, output_id).await
}

async fn wait_result(coordinator: &str, _task: TaskId, output: ObjectId) -> Result<(), Error> {
    for _ in 0..200 {
        match rpc(coordinator, RpcRequest::Client(ClientRequest::Get(output))).await? {
            RpcReply::Client(ClientReply::Object {
                id,
                bytes: Some(bytes),
                checksum: expected,
                size_bytes,
                ..
            }) => {
                verify_object(output, id, &bytes, expected, size_bytes)?;
                println!("{}", bincode::deserialize::<i64>(&bytes)?);
                return Ok(());
            }
            RpcReply::Client(ClientReply::Object {
                id,
                location,
                checksum: expected,
                size_bytes,
                ..
            }) => {
                match rpc(
                    &location,
                    RpcRequest::Client(ClientRequest::GetLocal(output)),
                )
                .await?
                {
                    RpcReply::Client(ClientReply::Object {
                        id: local_id,
                        bytes: Some(bytes),
                        checksum: actual,
                        size_bytes: local_size,
                        ..
                    }) if id == output && actual == expected && local_size == size_bytes => {
                        verify_object(output, local_id, &bytes, expected, size_bytes)?;
                        println!("{}", bincode::deserialize::<i64>(&bytes)?);
                        return Ok(());
                    }
                    _ => return Err(Error::ObjectConflict(output)),
                }
            }
            _ => tokio::time::sleep(Duration::from_millis(25)).await,
        }
    }
    Err(Error::DeadlineExceeded("task result"))
}
