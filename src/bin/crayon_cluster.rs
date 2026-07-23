use std::{str::FromStr, sync::Arc, time::Duration};

use crayon::{
    client::ClusterClient,
    cluster::{envelope, now_ms, read_frame, request, write_frame, CoordinatorServer},
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
    eprintln!("usage: crayon-cluster coordinator <addr> [lease-ms] | worker <coordinator> <advertise> [node-id] [cpu] [operations] | submit <coordinator> <a> <b> | submit-detach <coordinator> <operation> <value> [object-id] [cpu] [max-attempts] | status <coordinator> <task-id> | workers <coordinator> | cancel <coordinator> <task-id> | get <coordinator> <object-id>");
    std::process::exit(2)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("coordinator") => {
            let lease_ms = args
                .get(3)
                .map(|value| value.parse())
                .transpose()?
                .unwrap_or(5_000);
            CoordinatorServer::new(CLUSTER_ID, lease_ms)
                .allow_remote_bind()
                .serve(args.get(2).unwrap_or_else(|| usage()))
                .await?
        }
        Some("worker") => {
            run_worker(
                args.get(2).unwrap_or_else(|| usage()),
                args.get(3).unwrap_or_else(|| usage()),
                args.get(4)
                    .map(|value| NodeId::from_str(value))
                    .transpose()?,
                args.get(5)
                    .map(|value| value.parse())
                    .transpose()?
                    .unwrap_or(1.0),
                args.get(6).map(String::as_str).unwrap_or("all"),
            )
            .await?
        }
        Some("submit") => {
            let coordinator = args.get(2).unwrap_or_else(|| usage());
            let a: i64 = args.get(3).unwrap_or_else(|| usage()).parse()?;
            let b: i64 = args.get(4).unwrap_or_else(|| usage()).parse()?;
            run_client(coordinator, a, b).await?;
        }
        Some("submit-detach") => {
            run_submit_detach(&args).await?;
        }
        Some("status") => run_status(&args).await?,
        Some("workers") => run_workers(&args).await?,
        Some("cancel") => run_cancel(&args).await?,
        Some("get") => run_get(&args).await?,
        _ => usage(),
    }
    Ok(())
}

fn builtin_descriptor(name: &str) -> OperationDescriptor {
    let (input_codec, output_codec, max_inline_arg_bytes) = match name {
        "copy" => (Codec::RawBytes, Codec::RawBytes, 64 * 1024),
        _ => (Codec::BincodeV1, Codec::BincodeV1, 1024),
    };
    OperationDescriptor {
        key: OperationKey::new("builtin", name, 1),
        input_codec,
        output_codec,
        max_inline_arg_bytes,
    }
}

fn bind_addr_for_advertise(advertise: &str) -> String {
    if let Some(port) = advertise.rsplit(':').next() {
        format!("0.0.0.0:{port}")
    } else {
        advertise.to_string()
    }
}

async fn run_worker(
    coordinator: &str,
    advertise: &str,
    node_id: Option<NodeId>,
    cpu: f64,
    operations: &str,
) -> Result<(), Error> {
    let objects = LocalObjectStore::default();
    let bind_addr = bind_addr_for_advertise(advertise);
    serve_objects(&bind_addr, objects.clone()).await?;
    let mut registry = OperationRegistry::default();
    if matches!(operations, "all" | "add") {
        registry.register(builtin_descriptor("add"), |args| async move {
            if args.len() != 2 {
                return Err(Error::Protocol("add expects two arguments".into()));
            }
            let a: i64 = bincode::deserialize(&args[0])?;
            let b: i64 = bincode::deserialize(&args[1])?;
            Ok(bincode::serialize(&(a + b))?)
        })?;
    }
    if matches!(operations, "all" | "copy") {
        registry.register(builtin_descriptor("copy"), |args| async move {
            let [value]: [Vec<u8>; 1] = args
                .try_into()
                .map_err(|_| Error::Protocol("copy expects one argument".into()))?;
            Ok(value)
        })?;
    }
    if matches!(operations, "all" | "sleep") {
        registry.register(builtin_descriptor("sleep"), |args| async move {
            if args.len() != 1 {
                return Err(Error::Protocol("sleep expects one argument".into()));
            }
            let millis: u64 = bincode::deserialize(&args[0])?;
            tokio::time::sleep(Duration::from_millis(millis)).await;
            Ok(bincode::serialize(&millis)?)
        })?;
    }
    if registry.descriptors().is_empty() {
        return Err(Error::OperationUnavailable(operations.into()));
    }
    let registry = Arc::new(registry);
    let registration = RegisterWorker {
        node_id: node_id.unwrap_or_default(),
        worker_epoch: WorkerEpoch::new(),
        advertise_addr: advertise.into(),
        resources: ResourceSet::cpu_gpu(cpu, 0.0)?,
        slots: 1,
        operations: registry.descriptors(),
    };
    let registered = match rpc(
        coordinator,
        RpcRequest::Worker(WorkerRequest::Register(registration)),
        None,
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
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            match rpc(
                &heartbeat_coordinator,
                RpcRequest::Worker(WorkerRequest::Heartbeat(heartbeat_identity.clone())),
                Some(heartbeat_identity.coordinator_epoch),
            )
            .await
            {
                Ok(RpcReply::Worker(WorkerReply::Accepted)) => {}
                Err(Error::Io(_) | Error::DeadlineExceeded) => continue,
                Ok(RpcReply::Worker(WorkerReply::Error(_))) | Ok(_) | Err(_) => break,
            }
        }
    });
    loop {
        match rpc(
            coordinator,
            RpcRequest::Worker(WorkerRequest::Poll(identity.clone())),
            Some(identity.coordinator_epoch),
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
                report(
                    coordinator,
                    WorkerRequest::Cancelled {
                        identity: identity.clone(),
                        fence,
                    },
                    identity.coordinator_epoch,
                )
                .await?;
            }
            RpcReply::Worker(WorkerReply::Error(error)) => return Err(error),
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
    let epoch = identity.coordinator_epoch;
    report(
        coordinator,
        WorkerRequest::Started {
            identity: identity.clone(),
            fence: assignment.fence,
        },
        epoch,
    )
    .await?;
    let client = ClusterClient::connect_to(coordinator, CLUSTER_ID);
    let inputs = match fetch_inputs(&client, &assignment).await {
        Ok(inputs) => inputs,
        Err(error) => {
            report(
                coordinator,
                WorkerRequest::Failed {
                    identity: identity.clone(),
                    fence: assignment.fence,
                    message: error.to_string(),
                    retryable: true,
                },
                epoch,
            )
            .await?;
            return Ok(());
        }
    };
    let operation = assignment.clone();
    let execution_registry = registry.clone();
    let mut execution =
        tokio::spawn(async move { execution_registry.execute(&operation, inputs).await });
    let mut poll = tokio::time::interval(Duration::from_millis(50));
    let result = loop {
        tokio::select! {
            result = &mut execution => break Some(match result {
                Ok(value) => value,
                Err(join_error) if join_error.is_panic() => Err(Error::Protocol(
                    "operation panicked".into(),
                )),
                Err(_) => Err(Error::Protocol("operation task aborted".into())),
            }),
            _ = poll.tick() => {
                match rpc(coordinator, RpcRequest::Worker(WorkerRequest::Poll(identity.clone())), Some(epoch)).await {
                    Ok(RpcReply::Worker(WorkerReply::Cancel(fence))) if fence == assignment.fence => break None,
                    Ok(RpcReply::Worker(WorkerReply::Assignment(None))) => {}
                    Ok(RpcReply::Worker(WorkerReply::Error(Error::StaleEpoch))) => return Ok(()),
                    Ok(RpcReply::Worker(WorkerReply::Error(error))) => return Err(error),
                    Ok(_) => return Err(Error::Protocol("worker received assignment while busy".into())),
                    Err(Error::Io(_) | Error::DeadlineExceeded) => {}
                    Err(error) => return Err(error),
                }
            }
        }
    };
    let Some(result) = result else {
        report(
            coordinator,
            WorkerRequest::Cancelled {
                identity: identity.clone(),
                fence: assignment.fence,
            },
            epoch,
        )
        .await?;
        return Ok(());
    };
    match result {
        Ok(bytes) => {
            let descriptor = registry
                .descriptor(&assignment.operation)
                .ok_or_else(|| Error::OperationUnavailable(assignment.operation.to_string()))?;
            let object =
                objects.put(assignment.output_id, descriptor.output_codec.clone(), bytes)?;
            report(
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
                epoch,
            )
            .await
        }
        Err(error) => {
            report(
                coordinator,
                WorkerRequest::Failed {
                    identity: identity.clone(),
                    fence: assignment.fence,
                    message: error.to_string(),
                    retryable: true,
                },
                epoch,
            )
            .await
        }
    }
}

/// Reports a worker status update. Terminal-state conflicts and stale fences
/// are treated as "the coordinator already knows" so a single task cannot
/// terminate the worker session.
async fn report(
    coordinator: &str,
    request_body: WorkerRequest,
    coordinator_epoch: crayon::ids::CoordinatorEpoch,
) -> Result<(), Error> {
    match rpc(
        coordinator,
        RpcRequest::Worker(request_body),
        Some(coordinator_epoch),
    )
    .await?
    {
        RpcReply::Worker(WorkerReply::Accepted) => Ok(()),
        RpcReply::Worker(WorkerReply::Error(
            Error::IllegalTransition(_) | Error::StaleFence | Error::StaleEpoch,
        )) => Ok(()),
        RpcReply::Worker(WorkerReply::Error(error)) => Err(error),
        _ => Err(Error::Protocol(
            "coordinator did not accept worker report".into(),
        )),
    }
}

async fn fetch_inputs(
    client: &ClusterClient,
    assignment: &TaskAssignment,
) -> Result<Vec<Vec<u8>>, Error> {
    let mut inputs = Vec::with_capacity(assignment.args.len());
    for arg in &assignment.args {
        match arg {
            TaskArg::Inline { bytes, .. } => inputs.push(bytes.clone()),
            TaskArg::Object(id) => inputs.push(client.get_bytes(*id).await?.1),
        }
    }
    Ok(inputs)
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
                                    Err(error) => RpcReply::Client(ClientReply::Error(error)),
                                }
                            }
                            _ => RpcReply::Client(ClientReply::Error(Error::Protocol(
                                "invalid object request".into(),
                            ))),
                        }
                    }
                    _ => RpcReply::Client(ClientReply::Error(Error::Protocol(
                        "invalid object request".into(),
                    ))),
                };
                let _ =
                    tokio::time::timeout(Duration::from_secs(5), write_frame(&mut stream, &reply))
                        .await;
            });
        }
    });
    Ok(())
}

async fn rpc(
    address: &str,
    body: RpcRequest,
    coordinator_epoch: Option<crayon::ids::CoordinatorEpoch>,
) -> Result<RpcReply, Error> {
    let mut envelope = envelope(CLUSTER_ID, body);
    envelope.coordinator_epoch = coordinator_epoch;
    request(address, &envelope).await
}

async fn connected_client(coordinator: &str) -> Result<ClusterClient, Error> {
    let mut client = ClusterClient::connect(coordinator);
    client.connect_epoch().await?;
    Ok(client)
}

async fn run_submit_detach(args: &[String]) -> Result<(), Error> {
    let coordinator = args.get(2).unwrap_or_else(|| usage());
    let operation_name = args.get(3).unwrap_or_else(|| usage());
    let value: u64 = args
        .get(4)
        .unwrap_or_else(|| usage())
        .parse()
        .map_err(|error| Error::Protocol(format!("invalid value: {error}")))?;
    let descriptor = match operation_name.as_str() {
        "copy" => builtin_descriptor("copy"),
        "sleep" => builtin_descriptor("sleep"),
        _ => return Err(Error::OperationUnavailable(operation_name.clone())),
    };
    let input_codec = descriptor.input_codec.clone();
    let operation = crayon::Operation::<u64, u64>::new(descriptor)?;
    let arg = if let Some(id) = args.get(5).filter(|value| value.as_str() != "-") {
        TaskArg::Object(ObjectId::from_str(id).map_err(Error::Protocol)?)
    } else {
        TaskArg::Inline {
            codec: input_codec,
            bytes: bincode::serialize(&value)?,
        }
    };
    let cpu = args
        .get(6)
        .map(|value| value.parse())
        .transpose()
        .map_err(|error| Error::Protocol(format!("invalid cpu: {error}")))?
        .unwrap_or(1.0);
    let max_attempts = args
        .get(7)
        .map(|value| value.parse())
        .transpose()
        .map_err(|error| Error::Protocol(format!("invalid max attempts: {error}")))?
        .unwrap_or(1);
    let task = connected_client(coordinator)
        .await?
        .submit(
            &operation,
            vec![arg],
            ResourceSet::cpu_gpu(cpu, 0.0)?,
            max_attempts,
        )
        .await?;
    println!("{} {}", task.task_id, task.output.id);
    Ok(())
}

async fn run_status(args: &[String]) -> Result<(), Error> {
    let coordinator = args.get(2).unwrap_or_else(|| usage());
    let id = TaskId::from_str(args.get(3).unwrap_or_else(|| usage()))
        .map_err(|error| Error::Protocol(error.to_string()))?;
    let view = connected_client(coordinator).await?.status(id).await?;
    println!(
        "{} {} {} {} {}",
        view.task_id,
        view.output_id,
        view.state,
        view.attempt.0,
        view.worker
            .map(|worker| worker.to_string())
            .unwrap_or_else(|| "-".into())
    );
    Ok(())
}

async fn run_workers(args: &[String]) -> Result<(), Error> {
    let coordinator = args.get(2).unwrap_or_else(|| usage());
    for worker in connected_client(coordinator).await?.workers().await? {
        println!(
            "{} {} {} {} {}",
            worker.identity.node_id,
            worker.advertise_addr,
            worker.alive,
            worker.free_slots,
            worker
                .operations
                .iter()
                .map(|descriptor| descriptor.key.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    Ok(())
}

async fn run_cancel(args: &[String]) -> Result<(), Error> {
    let coordinator = args.get(2).unwrap_or_else(|| usage());
    let id = TaskId::from_str(args.get(3).unwrap_or_else(|| usage()))
        .map_err(|error| Error::Protocol(error.to_string()))?;
    connected_client(coordinator).await?.cancel(id).await?;
    println!("cancelled");
    Ok(())
}

async fn run_get(args: &[String]) -> Result<(), Error> {
    let coordinator = args.get(2).unwrap_or_else(|| usage());
    let id = ObjectId::from_str(args.get(3).unwrap_or_else(|| usage())).map_err(Error::Protocol)?;
    let (_, bytes) = connected_client(coordinator).await?.get_bytes(id).await?;
    let value: u64 = bincode::deserialize(&bytes)?;
    println!("{value}");
    Ok(())
}

async fn run_client(coordinator: &str, a: i64, b: i64) -> Result<(), Error> {
    let operation = crayon::Operation::<(i64, i64), i64>::new(builtin_descriptor("add"))?;
    let mut client = ClusterClient::connect_to(coordinator, CLUSTER_ID);
    client.connect_epoch().await?;
    let task = client
        .submit(
            &operation,
            vec![
                TaskArg::Inline {
                    codec: Codec::BincodeV1,
                    bytes: bincode::serialize(&a)?,
                },
                TaskArg::Inline {
                    codec: Codec::BincodeV1,
                    bytes: bincode::serialize(&b)?,
                },
            ],
            ResourceSet::cpu_gpu(1.0, 0.0)?,
            1,
        )
        .await?;
    println!("{}", task.result(Duration::from_secs(5)).await?);
    Ok(())
}
