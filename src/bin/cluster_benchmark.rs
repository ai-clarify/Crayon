use std::{
    collections::HashMap,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use crayon::{
    client::ClusterClient,
    error::Error,
    operation::{Codec, Operation, OperationKey, TaskArg},
    resources::ResourceSet,
};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    std::fs::create_dir_all(&args.artifact_dir)?;
    let binary = std::env::current_exe()?
        .parent()
        .unwrap()
        .join("crayon-cluster");

    let coordinator_addr = free_addr().await?;
    let coordinator = spawn(
        &binary,
        &["coordinator", &coordinator_addr, "5000"],
        &args.artifact_dir,
    )?;
    let mut workers = Vec::new();
    for _ in 0..args.workers {
        let advertise = free_addr().await?;
        let node_id = crayon::ids::NodeId::new().to_string();
        let worker = spawn(
            &binary,
            &[
                "worker",
                &coordinator_addr,
                &advertise,
                &node_id,
                "1.0",
                "all",
            ],
            &args.artifact_dir,
        )?;
        workers.push(worker);
    }

    let mut client = ClusterClient::connect(&coordinator_addr);
    wait_coordinator(&mut client).await?;
    client.connect_epoch().await?;
    wait_ready(&mut client, args.workers).await?;

    let mut samples = Vec::new();
    run_task_scenario(&mut client, &args, &mut samples).await?;
    run_dag_scenario(&mut client, &args, &mut samples).await?;

    for worker in workers.drain(..) {
        kill(worker);
    }
    kill(coordinator);

    write_artifacts(&args, &samples)?;
    Ok(())
}

struct Args {
    workers: usize,
    concurrency: usize,
    warmups: usize,
    samples: usize,
    payload_bytes: usize,
    artifact_dir: PathBuf,
}

fn parse_args() -> Args {
    let mut values = HashMap::new();
    let mut iter = std::env::args().skip(1);
    while let Some(key) = iter.next() {
        let Some(value) = iter.next() else {
            eprintln!("missing value for {key}");
            std::process::exit(2);
        };
        values.insert(key, value);
    }
    Args {
        workers: values
            .get("--workers")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
        concurrency: values
            .get("--concurrency")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
        warmups: values
            .get("--warmups")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5),
        samples: values
            .get("--samples")
            .and_then(|v| v.parse().ok())
            .unwrap_or(50),
        payload_bytes: values
            .get("--payload-bytes")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024),
        artifact_dir: values
            .get("--artifact-dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("benchmark_artifacts/cluster")),
    }
}

async fn free_addr() -> Result<String, Error> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    Ok(listener.local_addr()?.to_string())
}

fn spawn(
    binary: &std::path::Path,
    args: &[&str],
    artifact_dir: &std::path::Path,
) -> Result<Child, Error> {
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(artifact_dir.join("process.log"))?;
    Ok(Command::new(binary)
        .args(args)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .spawn()?)
}

fn kill(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

async fn wait_coordinator(client: &mut ClusterClient) -> Result<(), Error> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if client.workers().await.is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(Error::Protocol("coordinator did not start in time".into()));
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_ready(client: &mut ClusterClient, workers: usize) -> Result<(), Error> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if client.workers().await?.len() >= workers {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(Error::Protocol("workers did not register in time".into()));
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn run_task_scenario(
    client: &mut ClusterClient,
    args: &Args,
    rows: &mut Vec<Sample>,
) -> Result<(), Error> {
    let operation = Operation::<(i64, i64), i64>::new(crayon::OperationDescriptor {
        key: OperationKey::new("builtin", "add", 1),
        input_codec: Codec::BincodeV1,
        output_codec: Codec::BincodeV1,
        max_inline_arg_bytes: 1024,
    })?;
    let total = args.warmups + args.samples;
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(args.concurrency));
    let mut handles = Vec::new();
    for index in 0..total {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Error::Protocol("semaphore closed".into()))?;
        let client = client.clone();
        let operation = operation.clone();
        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let start = Instant::now();
            let handle = client
                .submit(
                    &operation,
                    vec![
                        TaskArg::Inline {
                            codec: Codec::BincodeV1,
                            bytes: bincode::serialize(&20i64).unwrap(),
                        },
                        TaskArg::Inline {
                            codec: Codec::BincodeV1,
                            bytes: bincode::serialize(&22i64).unwrap(),
                        },
                    ],
                    ResourceSet::cpu_gpu(1.0, 0.0)?,
                    1,
                )
                .await?;
            let value: i64 = handle.result(Duration::from_secs(30)).await?;
            let elapsed = start.elapsed();
            if value != 42 {
                return Err(Error::Protocol(format!("expected 42, got {value}")));
            }
            Ok((index, elapsed))
        }));
    }
    for handle in handles {
        let (index, elapsed) = handle.await.map_err(join_error)??;
        rows.push(Sample {
            scenario: "task".into(),
            index,
            warmup: index < args.warmups,
            payload_bytes: 0,
            elapsed_ns: elapsed.as_nanos() as u64,
        });
    }
    Ok(())
}

async fn run_dag_scenario(
    client: &mut ClusterClient,
    args: &Args,
    rows: &mut Vec<Sample>,
) -> Result<(), Error> {
    let operation = Operation::<Vec<u8>, Vec<u8>>::new(crayon::OperationDescriptor {
        key: OperationKey::new("builtin", "copy", 1),
        input_codec: Codec::RawBytes,
        output_codec: Codec::RawBytes,
        max_inline_arg_bytes: 64 * 1024,
    })?;
    let payload: Vec<u8> = (0..args.payload_bytes as u64)
        .map(|value| (value % 251) as u8)
        .collect();
    let expected = blake3::hash(&payload);
    let total = args.warmups + args.samples;
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(args.concurrency));
    let mut handles = Vec::new();
    for index in 0..total {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Error::Protocol("semaphore closed".into()))?;
        let client = client.clone();
        let operation = operation.clone();
        let payload = payload.clone();
        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let start = Instant::now();
            let first = client
                .submit(
                    &operation,
                    vec![TaskArg::Inline {
                        codec: Codec::RawBytes,
                        bytes: payload.clone(),
                    }],
                    ResourceSet::cpu_gpu(1.0, 0.0)?,
                    1,
                )
                .await?;
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                let status = client.status(first.task_id).await?;
                if matches!(status.state, crayon::protocol::TaskStatus::Succeeded) {
                    break;
                }
                if matches!(
                    status.state,
                    crayon::protocol::TaskStatus::Failed(_)
                        | crayon::protocol::TaskStatus::Cancelled
                ) {
                    return Err(Error::Protocol(format!(
                        "first dag task terminal: {:?}",
                        status.state
                    )));
                }
                if Instant::now() >= deadline {
                    return Err(Error::DeadlineExceeded);
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let first_output = first.output.id;
            let second = client
                .submit(
                    &operation,
                    vec![TaskArg::Object(first_output)],
                    ResourceSet::cpu_gpu(1.0, 0.0)?,
                    1,
                )
                .await?;
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                let status = client.status(second.task_id).await?;
                if matches!(status.state, crayon::protocol::TaskStatus::Succeeded) {
                    break;
                }
                if matches!(
                    status.state,
                    crayon::protocol::TaskStatus::Failed(_)
                        | crayon::protocol::TaskStatus::Cancelled
                ) {
                    return Err(Error::Protocol(format!(
                        "second dag task terminal: {:?}",
                        status.state
                    )));
                }
                if Instant::now() >= deadline {
                    return Err(Error::DeadlineExceeded);
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let (_, value): (crayon::Codec, Vec<u8>) = client.get_bytes(second.output.id).await?;
            let elapsed = start.elapsed();
            if blake3::hash(&value) != expected {
                return Err(Error::Protocol("dag payload checksum mismatch".into()));
            }
            Ok((index, elapsed))
        }));
    }
    for handle in handles {
        let (index, elapsed) = handle.await.map_err(join_error)??;
        rows.push(Sample {
            scenario: "dag".into(),
            index,
            warmup: index < args.warmups,
            payload_bytes: args.payload_bytes,
            elapsed_ns: elapsed.as_nanos() as u64,
        });
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct Sample {
    scenario: String,
    index: usize,
    warmup: bool,
    payload_bytes: usize,
    elapsed_ns: u64,
}

fn write_artifacts(args: &Args, samples: &[Sample]) -> Result<(), Error> {
    std::fs::create_dir_all(&args.artifact_dir)?;
    let mut csv = String::from("scenario,index,warmup,payload_bytes,elapsed_ns\n");
    for sample in samples {
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            sample.scenario, sample.index, sample.warmup, sample.payload_bytes, sample.elapsed_ns
        ));
    }
    std::fs::write(args.artifact_dir.join("samples.csv"), csv)?;

    let summary = summarize(samples);
    std::fs::write(
        args.artifact_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary).map_err(json_error)?,
    )?;

    let manifest = serde_json::json!({
        "command": std::env::args().collect::<Vec<_>>().join(" "),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "rust_version": rust_version(),
        "os": os_info(),
        "cpu_model": cpu_model(),
        "logical_cpus": logical_cpus(),
        "workers": args.workers,
        "concurrency": args.concurrency,
        "warmups": args.warmups,
        "samples": args.samples,
        "payload_bytes": args.payload_bytes,
        "topology": "loopback coordinator and worker processes",
    });
    std::fs::write(
        args.artifact_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).map_err(json_error)?,
    )?;
    Ok(())
}

fn json_error(error: serde_json::Error) -> Error {
    Error::Protocol(error.to_string())
}

fn join_error(error: tokio::task::JoinError) -> Error {
    Error::Protocol(error.to_string())
}

fn summarize(samples: &[Sample]) -> serde_json::Value {
    let mut by_scenario = serde_json::Map::new();
    for scenario in ["task", "dag"] {
        let measured: Vec<u64> = samples
            .iter()
            .filter(|sample| sample.scenario == scenario && !sample.warmup)
            .map(|sample| sample.elapsed_ns)
            .collect();
        if measured.is_empty() {
            continue;
        }
        by_scenario.insert(scenario.into(), stats(&measured));
    }
    serde_json::Value::Object(by_scenario)
}

fn stats(values: &[u64]) -> serde_json::Value {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let count = sorted.len() as f64;
    let sum: u64 = sorted.iter().sum();
    let mean = sum as f64 / count;
    let variance = sorted
        .iter()
        .map(|value| (*value as f64 - mean).powi(2))
        .sum::<f64>()
        / count;
    serde_json::json!({
        "count": sorted.len(),
        "median_ns": percentile(&sorted, 50.0),
        "mean_ns": mean,
        "stdev_ns": variance.sqrt(),
        "p25_ns": percentile(&sorted, 25.0),
        "p75_ns": percentile(&sorted, 75.0),
        "p90_ns": percentile(&sorted, 90.0),
        "p95_ns": percentile(&sorted, 95.0),
        "min_ns": sorted[0],
        "max_ns": sorted[sorted.len() - 1],
        "ops_per_sec": 1_000_000_000.0 / mean,
    })
}

fn percentile(sorted: &[u64], pct: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0] as f64;
    }
    let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted[lower] as f64;
    }
    let weight = rank - lower as f64;
    sorted[lower] as f64 * (1.0 - weight) + sorted[upper] as f64 * weight
}

fn git_sha() -> String {
    String::from_utf8_lossy(
        &std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .map(|output| output.stdout)
            .unwrap_or_default(),
    )
    .trim()
    .to_string()
}

fn git_dirty() -> bool {
    std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|output| !output.stdout.is_empty())
        .unwrap_or(false)
}

fn rust_version() -> String {
    String::from_utf8_lossy(
        &std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .map(|output| output.stdout)
            .unwrap_or_default(),
    )
    .trim()
    .to_string()
}

fn os_info() -> String {
    String::from_utf8_lossy(
        &std::process::Command::new("uname")
            .args(["-s", "-r", "-m"])
            .output()
            .map(|output| output.stdout)
            .unwrap_or_default(),
    )
    .trim()
    .to_string()
}

fn cpu_model() -> String {
    if let Ok(output) = std::process::Command::new("lscpu").output() {
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Some(value) = line.strip_prefix("Model name:") {
                return value.trim().to_string();
            }
        }
    }
    if let Ok(output) = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
    {
        return String::from_utf8_lossy(&output.stdout).trim().to_string();
    }
    "unknown".into()
}

fn logical_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}
