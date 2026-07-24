use std::{str::FromStr, sync::Arc, time::Duration};

use crayon::{
    client::ClusterClient,
    cluster::{envelope, now_ms, read_frame, request, write_frame, CoordinatorServer},
    data_plane::LocalObjectStore,
    error::Error,
    ids::{ClusterId, NodeId, ObjectId, TaskId, WorkerEpoch},
    operation::{Codec, OperationDescriptor, OperationKey, TaskArg},
    protocol::{
        ClientReply, ClientRequest, Envelope, FailureClass, ObjectPayload, RegisterWorker,
        RpcReply, RpcRequest, TaskAssignment, TaskCompletion, WorkerIdentity, WorkerReply,
        WorkerRequest,
    },
    resources::ResourceSet,
    worker::OperationRegistry,
};
use tokio::sync::Semaphore;

const CLUSTER_ID: ClusterId = ClusterId([0; 16]);
const OBJECT_CONNECTIONS: usize = 128;
/// How long an idle worker asks the coordinator to hold its poll; an assignment
/// wakes it sooner. Kept under the long-poll cap so the held request answers cleanly.
const WORKER_POLL_WAIT_MS: u64 = 2_000;
/// Outputs at or below this ride inline in the completion; larger stay worker-local.
const INLINE_RESULT_MAX_BYTES: usize = 64 * 1024;
/// SIGTERM grace: an in-flight task gets this long (signal-relative) to finish
/// and report; then the process exits without reporting and the lease reaper
/// re-queues the task. Matches K8s' default 30s terminationGracePeriodSeconds
/// with margin for the pod kill.
const WORKER_DRAIN_GRACE_MS: u64 = 20_000;

fn usage() -> ! {
    eprintln!("usage: crayon-cluster coordinator <addr> [lease-ms] | worker <coordinator> <advertise> [node-id] [cpu] [operations] | submit <coordinator> <a> <b> | submit-detach <coordinator> <operation> <value> [object-id] [cpu] [max-attempts] | status <coordinator> <task-id> | workers <coordinator> | cancel <coordinator> <task-id> | get <coordinator> <object-id> | release <coordinator> <object-id>");
    std::process::exit(2)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic: {info}");
    }));
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
        Some("release") => run_release(&args).await?,
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

fn rl_descriptor() -> OperationDescriptor {
    OperationDescriptor {
        key: OperationKey::new("rl", "rollout", 1),
        input_codec: Codec::BincodeV1,
        output_codec: Codec::BincodeV1,
        max_inline_arg_bytes: 64 * 1024,
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct RolloutRequest {
    policy_seed: u64,
    theta: Vec<f32>,
    env_seed: u64,
    steps: u32,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct RolloutResult {
    env_seed: u64,
    episode_return: f32,
    steps: u32,
}

/// Simple seeded linear-policy bandit rollout. Deterministic given the request,
/// so results are reproducible across runs and workers.
fn run_rollout(request: &RolloutRequest) -> RolloutResult {
    let mut state = request.env_seed ^ request.policy_seed;
    let mut r#return = 0.0f32;
    for _ in 0..request.steps {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let action = ((state >> 33) as usize) % request.theta.len().max(1);
        let reward = ((state >> 1) & 0xff) as f32 / 255.0;
        r#return += reward * request.theta[action];
    }
    RolloutResult {
        env_seed: request.env_seed,
        episode_return: r#return,
        steps: request.steps,
    }
}

// ---- Real-LLM RL pipeline (roles: llm-actor via Python sidecar, llm-judge in
// Rust). Wire shapes are shared with src/bin/llm_rl.rs — keep in sync.

fn llm_descriptor(name: &str) -> OperationDescriptor {
    OperationDescriptor {
        key: OperationKey::new("llm", name, 1),
        // JSON on both sides so a Python driver reads and writes args directly.
        input_codec: Codec::JsonV1,
        output_codec: Codec::JsonV1,
        max_inline_arg_bytes: 64 * 1024,
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct LlmRolloutCfg {
    version: u64,
    /// Hex object id of the policy weights in the coordinator's arena; empty
    /// means "use the base model as loaded" (e.g. a pure evaluation pass).
    #[serde(default)]
    weights: String,
    seeds: Vec<u64>,
    max_new_tokens: u32,
    /// Sampling temperature; 0 means greedy decoding (evaluation).
    #[serde(default = "default_temperature")]
    temperature: f32,
    /// Explicit prompts (e.g. GSM8K questions), one per seed. Empty means the
    /// synthetic arithmetic prompts derived from the seeds.
    #[serde(default)]
    prompts: Vec<String>,
}
fn default_temperature() -> f32 {
    1.0
}

#[derive(serde::Serialize, serde::Deserialize)]
struct LlmRollout {
    seed: u64,
    completion: String,
}

/// Seed -> arithmetic problem; mirrors `prompt_for` in benchmarks/llm_sidecar.py.
fn llm_problem(seed: u64) -> (u64, u64) {
    (50 + (seed >> 8) % 900, 50 + seed % 900)
}

fn last_integer(text: &str) -> Option<i64> {
    // Thousands separators are common in model output ("1,200"); drop them
    // before splitting so the grouped digits parse as one number.
    let text = text.replace(',', "");
    text.split(|c: char| !c.is_ascii_digit() && c != '-')
        .rfind(|s| s.chars().any(|c| c.is_ascii_digit()) && s.len() < 12)
        .and_then(|s| {
            s.trim_start_matches('-').parse::<i64>().ok().map(|n| {
                if s.starts_with('-') {
                    -n
                } else {
                    n
                }
            })
        })
}

/// A resident `benchmarks/llm_sidecar.py` process, spoken to over
/// newline-delimited JSON. Spawned lazily on first use; configuration comes
/// from CRAYON_LLM_SIDECAR (script path) and CRAYON_LLM_MODEL (model dir).
#[derive(Default)]
struct LlmSidecar {
    version: u64,
    io: Option<(
        std::process::ChildStdin,
        std::io::BufReader<std::process::ChildStdout>,
    )>,
}

impl LlmSidecar {
    fn call(&mut self, request: &serde_json::Value) -> Result<serde_json::Value, Error> {
        use std::io::{BufRead, Write};
        if self.io.is_none() {
            let script = std::env::var("CRAYON_LLM_SIDECAR")
                .map_err(|_| Error::Protocol("CRAYON_LLM_SIDECAR not set".into()))?;
            let model = std::env::var("CRAYON_LLM_MODEL")
                .map_err(|_| Error::Protocol("CRAYON_LLM_MODEL not set".into()))?;
            let mut child = std::process::Command::new("python3")
                .arg(&script)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .map_err(|e| Error::Protocol(format!("spawn sidecar: {e}")))?;
            let stdin = child.stdin.take().unwrap();
            let stdout = std::io::BufReader::new(child.stdout.take().unwrap());
            self.io = Some((stdin, stdout));
            self.call(&serde_json::json!({"cmd": "init", "model": model}))?;
        }
        let (stdin, stdout) = self.io.as_mut().unwrap();
        writeln!(stdin, "{request}").map_err(|e| Error::Protocol(format!("sidecar: {e}")))?;
        let mut line = String::new();
        stdout
            .read_line(&mut line)
            .map_err(|e| Error::Protocol(format!("sidecar: {e}")))?;
        let reply: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| Error::Protocol(format!("sidecar reply: {e}")))?;
        if let Some(error) = reply.get("error") {
            return Err(Error::Protocol(format!("sidecar: {error}")));
        }
        Ok(reply)
    }

    fn load(&mut self, version: u64, weights: &[u8]) -> Result<(), Error> {
        let path =
            std::env::temp_dir().join(format!("crayon-llm-{}-{version}.pt", std::process::id()));
        std::fs::write(&path, weights)?;
        let result = self.call(&serde_json::json!({
            "cmd": "load", "path": path, "version": version,
        }));
        let _ = std::fs::remove_file(&path);
        result?;
        self.version = version;
        Ok(())
    }

    fn rollout(
        &mut self,
        seeds: &[u64],
        prompts: &[String],
        max_new_tokens: u32,
        temperature: f32,
    ) -> Result<Vec<String>, Error> {
        let reply = self.call(&serde_json::json!({
            "cmd": "rollout", "seeds": seeds, "prompts": prompts,
            "max_new_tokens": max_new_tokens, "temperature": temperature,
        }))?;
        reply["rollouts"]
            .as_array()
            .map(|rollouts| {
                rollouts
                    .iter()
                    .map(|r| r["completion"].as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .ok_or_else(|| Error::Protocol("sidecar rollout reply malformed".into()))
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
    eprintln!("worker starting: coordinator={coordinator} advertise={advertise}");
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
    if matches!(operations, "all" | "rl") {
        registry.register(rl_descriptor(), |args| async move {
            if args.len() != 1 {
                return Err(Error::Protocol("rl.rollout expects one argument".into()));
            }
            let request: RolloutRequest = bincode::deserialize(&args[0])?;
            let result = run_rollout(&request);
            Ok(bincode::serialize(&result)?)
        })?;
    }
    if operations == "llm-actor" {
        // Real-LLM rollout: a resident Python sidecar holds the model on the
        // GPU; this op syncs it to the requested policy version (weights fetched
        // from the coordinator's arena once per version, not per task) and asks
        // it to generate completions for the seeded arithmetic prompts.
        let sidecar = Arc::new(std::sync::Mutex::new(LlmSidecar::default()));
        let coordinator_addr = coordinator.to_string();
        registry.register(llm_descriptor("rollout"), move |args| {
            let sidecar = sidecar.clone();
            let coordinator_addr = coordinator_addr.clone();
            async move {
                if args.len() != 1 {
                    return Err(Error::Protocol("llm.rollout expects [config]".into()));
                }
                let cfg: LlmRolloutCfg = serde_json::from_slice(&args[0])
                    .map_err(|e| Error::Protocol(format!("llm.rollout config: {e}")))?;
                let stale =
                    !cfg.weights.is_empty() && sidecar.lock().unwrap().version != cfg.version;
                let weights = if stale {
                    let weights_id = ObjectId::from_str(&cfg.weights)
                        .map_err(|e| Error::Protocol(format!("llm.rollout weights id: {e}")))?;
                    let client = ClusterClient::connect_to(&coordinator_addr, CLUSTER_ID);
                    Some(client.get_bytes(weights_id).await?.1)
                } else {
                    None
                };
                tokio::task::spawn_blocking(move || {
                    let mut sidecar = sidecar.lock().unwrap();
                    if let Some(weights) = weights {
                        sidecar.load(cfg.version, &weights)?;
                    }
                    let completions = sidecar.rollout(
                        &cfg.seeds,
                        &cfg.prompts,
                        cfg.max_new_tokens,
                        cfg.temperature,
                    )?;
                    let rollouts: Vec<LlmRollout> = cfg
                        .seeds
                        .iter()
                        .zip(completions)
                        .map(|(&seed, completion)| LlmRollout { seed, completion })
                        .collect();
                    serde_json::to_vec(&rollouts)
                        .map_err(|e| Error::Protocol(format!("llm.rollout encode: {e}")))
                })
                .await
                .map_err(|_| Error::Protocol("llm sidecar task aborted".into()))?
            }
        })?;
    }
    if operations == "llm-judge" {
        // Rule reward: the completion must contain the arithmetic answer as its
        // last integer. Pure Rust — no model, no GPU.
        registry.register(llm_descriptor("judge"), |args| async move {
            if args.is_empty() || args.len() > 2 {
                return Err(Error::Protocol(
                    "llm.judge expects [rollouts, golds?]".into(),
                ));
            }
            let rollouts: Vec<LlmRollout> = serde_json::from_slice(&args[0])
                .map_err(|e| Error::Protocol(format!("llm.judge input: {e}")))?;
            // Optional dataset golds (seed -> answer, e.g. GSM8K); without them
            // the answer derives from the seed's synthetic arithmetic problem.
            let golds: std::collections::HashMap<u64, i64> = match args.get(1) {
                Some(bytes) => serde_json::from_slice::<Vec<(u64, i64)>>(bytes)
                    .map_err(|e| Error::Protocol(format!("llm.judge golds: {e}")))?
                    .into_iter()
                    .collect(),
                None => rollouts
                    .iter()
                    .map(|r| {
                        let (a, b) = llm_problem(r.seed);
                        (r.seed, (a + b) as i64)
                    })
                    .collect(),
            };
            let rewards: Vec<f32> = rollouts
                .iter()
                .map(|r| (last_integer(&r.completion) == golds.get(&r.seed).copied()) as u32 as f32)
                .collect();
            serde_json::to_vec(&rewards)
                .map_err(|e| Error::Protocol(format!("llm.judge encode: {e}")))
        })?;
    }
    if registry.descriptors().is_empty() {
        return Err(Error::OperationUnavailable(operations.into()));
    }
    let registry = Arc::new(registry);
    let node_id = node_id.unwrap_or_default();
    let drain = Arc::new(DrainSignal::default());
    {
        // Shutdown signal → notify the coordinator immediately (even mid-task,
        // so it stops assigning), then grace timer, then exit unreported.
        let drain = drain.clone();
        let coordinator = coordinator.to_string();
        tokio::spawn(async move {
            crayon::cluster::shutdown_signal().await;
            eprintln!("worker draining on shutdown signal");
            drain.set();
            let identity = drain.identity.lock().unwrap().clone();
            if let Some(identity) = identity {
                let epoch = identity.coordinator_epoch;
                let _ = report(&coordinator, WorkerRequest::Drain(identity), epoch).await;
            }
            tokio::time::sleep(Duration::from_millis(WORKER_DRAIN_GRACE_MS)).await;
            eprintln!("worker drain grace expired, exiting");
            std::process::exit(0);
        });
    }
    let mut backoff_ms: u64 = 0;
    loop {
        match run_worker_session(
            coordinator,
            advertise,
            node_id,
            cpu,
            registry.clone(),
            objects.clone(),
            drain.clone(),
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(WorkerSessionEnd::Reconnect) => {
                if drain.is_set() {
                    return Ok(());
                }
                backoff_ms = if backoff_ms == 0 {
                    100
                } else {
                    (backoff_ms * 2).min(5_000)
                };
                let jitter = backoff_ms / 4;
                let delay = backoff_ms + (rand_jitter() % jitter);
                eprintln!("worker reconnecting in {delay}ms");
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            Err(WorkerSessionEnd::Fatal(error)) => return Err(error),
        }
    }
}

enum WorkerSessionEnd {
    Reconnect,
    Fatal(Error),
}

/// Set by the SIGTERM handler; the poll loop exits between tasks, and the
/// handler itself notifies the coordinator using the last registered identity.
#[derive(Default)]
struct DrainSignal {
    flag: std::sync::atomic::AtomicBool,
    identity: std::sync::Mutex<Option<WorkerIdentity>>,
}

impl DrainSignal {
    fn set(&self) {
        self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    fn is_set(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn rand_jitter() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Aborts a background task when dropped, binding its lifetime to a scope.
struct AbortOnDrop(tokio::task::JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn run_worker_session(
    coordinator: &str,
    advertise: &str,
    node_id: NodeId,
    cpu: f64,
    registry: Arc<OperationRegistry>,
    objects: LocalObjectStore,
    drain: Arc<DrainSignal>,
) -> Result<(), WorkerSessionEnd> {
    let registration = RegisterWorker {
        node_id,
        worker_epoch: WorkerEpoch::new(),
        advertise_addr: advertise.into(),
        resources: ResourceSet::cpu_gpu(cpu, 0.0).map_err(WorkerSessionEnd::Fatal)?,
        slots: 1,
        operations: registry.descriptors(),
    };
    let registered = 'register: loop {
        match rpc(
            coordinator,
            RpcRequest::Worker(WorkerRequest::Register(registration.clone())),
            None,
        )
        .await
        {
            Ok(RpcReply::Worker(WorkerReply::Registered(value))) => break 'register value,
            Ok(other) => {
                eprintln!("worker registration failed: {other:?}");
                return Err(WorkerSessionEnd::Fatal(Error::Protocol(format!(
                    "registration failed: {other:?}"
                ))));
            }
            Err(error) => {
                eprintln!("worker registration rpc error: {error}");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    };
    eprintln!("worker registered successfully");
    let identity = registered.identity;
    *drain.identity.lock().unwrap() = Some(identity.clone());
    let heartbeat_identity = identity.clone();
    let heartbeat_coordinator = coordinator.to_string();
    let heartbeat_period = Duration::from_millis((registered.lease_timeout_ms / 3).max(10));
    // Bound to the session: aborted when run_worker_session returns on any path,
    // so the heartbeat task does not leak across reconnects.
    let _heartbeat = AbortOnDrop(tokio::spawn(async move {
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
    }));
    loop {
        if drain.is_set() {
            eprintln!("worker drained, exiting");
            return Ok(());
        }
        let poll_reply = match rpc(
            coordinator,
            RpcRequest::Worker(WorkerRequest::Poll {
                identity: identity.clone(),
                wait_ms: WORKER_POLL_WAIT_MS,
            }),
            Some(identity.coordinator_epoch),
        )
        .await
        {
            Ok(reply) => reply,
            Err(Error::Io(_) | Error::DeadlineExceeded) => {
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
            Err(Error::StaleEpoch | Error::StaleFence) => return Err(WorkerSessionEnd::Reconnect),
            Err(error) => {
                eprintln!("worker poll rpc error: {error}");
                return Err(WorkerSessionEnd::Reconnect);
            }
        };
        match poll_reply {
            RpcReply::Worker(WorkerReply::Assignment(Some(assignment))) => {
                execute_assignment(
                    coordinator,
                    advertise,
                    &identity,
                    assignment,
                    registry.clone(),
                    objects.clone(),
                )
                .await
                .map_err(|error| match error {
                    // Fencing (lease lost mid-execution) and transient transport errors
                    // should reconnect, not kill the worker; only genuine faults fatal.
                    Error::StaleEpoch
                    | Error::StaleFence
                    | Error::Io(_)
                    | Error::DeadlineExceeded => WorkerSessionEnd::Reconnect,
                    other => WorkerSessionEnd::Fatal(other),
                })?;
            }
            // The coordinator held the poll for its full wait and had no work.
            // Loop straight back into another long-poll; no client-side sleep.
            RpcReply::Worker(WorkerReply::Assignment(None)) => {}
            RpcReply::Worker(WorkerReply::Cancel(fence)) => {
                report(
                    coordinator,
                    WorkerRequest::Cancelled {
                        identity: identity.clone(),
                        fence,
                    },
                    identity.coordinator_epoch,
                )
                .await
                .map_err(WorkerSessionEnd::Fatal)?;
            }
            RpcReply::Worker(WorkerReply::DeleteObject(id)) => {
                objects.delete(id);
            }
            RpcReply::Worker(WorkerReply::Error(Error::StaleEpoch | Error::StaleFence)) => {
                eprintln!("worker session fenced, re-registering");
                return Err(WorkerSessionEnd::Reconnect);
            }
            RpcReply::Worker(WorkerReply::Error(error)) => {
                eprintln!("worker poll error reply: {error}");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
            _ => {
                eprintln!("worker unexpected poll reply");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
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
            // ObjectLost (no lineage — gone forever) and codec errors are
            // Permanent; transport faults and not-yet-ready inputs retry.
            let class = error.failure_class();
            report(
                coordinator,
                WorkerRequest::Failed {
                    identity: identity.clone(),
                    fence: assignment.fence,
                    message: error.to_string(),
                    class,
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
    // first tick at +50ms, not t=0, so sub-50ms tasks skip the busy-poll entirely
    let start = tokio::time::Instant::now() + Duration::from_millis(50);
    let mut poll = tokio::time::interval_at(start, Duration::from_millis(50));
    let result = loop {
        tokio::select! {
            result = &mut execution => break Some(match result {
                // Operation errors classify by variant (deterministic errors like
                // bad input shape are Permanent); a panic is always Permanent —
                // retrying a crashing op just burns attempts.
                Ok(value) => value.map_err(|error| {
                    let class = error.failure_class();
                    (error, class)
                }),
                Err(join_error) if join_error.is_panic() => Err((
                    Error::Protocol("operation panicked".into()),
                    FailureClass::Permanent,
                )),
                Err(_) => Err((
                    Error::Protocol("operation task aborted".into()),
                    FailureClass::Transient,
                )),
            }),
            _ = poll.tick() => {
                // Busy worker: a non-blocking poll (wait_ms 0) purely to observe
                // cancellation. The 50ms interval drives the cadence, not the server.
                match rpc(coordinator, RpcRequest::Worker(WorkerRequest::Poll { identity: identity.clone(), wait_ms: 0 }), Some(epoch)).await {
                    Ok(RpcReply::Worker(WorkerReply::Cancel(fence))) if fence == assignment.fence => break None,
                    Ok(RpcReply::Worker(WorkerReply::Assignment(None))) => {}
                    Ok(RpcReply::Worker(WorkerReply::Error(Error::StaleEpoch))) => return Err(Error::StaleEpoch),
                    Ok(RpcReply::Worker(WorkerReply::Error(error))) => return Err(error),
                    // A client release() of a completed output owned by this worker is
                    // served first by the coordinator's Poll handler, even to a busy
                    // worker; honor it here rather than treating it as a stray assignment.
                    Ok(RpcReply::Worker(WorkerReply::DeleteObject(id))) => objects.delete(id),
                    Ok(_) => return Err(Error::Protocol("worker received assignment while busy".into())),
                    Err(Error::Io(_) | Error::DeadlineExceeded) => {}
                    Err(error) => return Err(error),
                }
            }
        }
    };
    let Some(result) = result else {
        // Dropping the JoinHandle only detaches the task in tokio; abort and await
        // so the operation is truly stopped before the coordinator frees the slot.
        execution.abort();
        let _ = (&mut execution).await;
        // ponytail: cooperative abort — a fully CPU-bound op with no .await still runs
        // to its next yield; hard process-kill of workers is out of scope for this
        // in-process worker.
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
            // Ship small outputs inline so the coordinator answers Get in one hop;
            // large outputs stay worker-local and are fetched from `location`.
            let inline =
                (object.bytes.len() <= INLINE_RESULT_MAX_BYTES).then(|| object.bytes.clone());
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
                        bytes: inline,
                    },
                },
                epoch,
            )
            .await
        }
        Err((error, class)) => {
            report(
                coordinator,
                WorkerRequest::Failed {
                    identity: identity.clone(),
                    fence: assignment.fence,
                    message: error.to_string(),
                    class,
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
            let _ = stream.set_nodelay(true);
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
                                    Ok(object) => {
                                        RpcReply::Client(ClientReply::Object(ObjectPayload {
                                            id,
                                            codec: object.codec,
                                            size_bytes: object.bytes.len() as u64,
                                            checksum: object.checksum,
                                            location: String::new(),
                                            bytes: Some(object.bytes),
                                            arena: None,
                                        }))
                                    }
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

async fn run_release(args: &[String]) -> Result<(), Error> {
    let coordinator = args.get(2).unwrap_or_else(|| usage());
    let id = ObjectId::from_str(args.get(3).unwrap_or_else(|| usage())).map_err(Error::Protocol)?;
    connected_client(coordinator).await?.release(id).await?;
    println!("released");
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
