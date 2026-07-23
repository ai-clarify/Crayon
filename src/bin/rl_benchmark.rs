//! Long-running distributed RL training simulation.
//!
//! Simulates the rollout-collection phase of RL: a coordinator hands out
//! seeded environment rollouts to workers, each worker runs a fixed number of
//! steps with the current policy parameters, and the coordinator aggregates
//! episode returns. This exercises long-running multi-stage task graphs,
//! object dependencies, retries, and cancellation on a real cluster.

use std::{
    collections::HashMap,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use crayon::{
    client::ClusterClient,
    error::Error,
    operation::{Codec, Operation, OperationDescriptor, OperationKey, TaskArg},
    resources::ResourceSet,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Policy {
    seed: u64,
    theta: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct RolloutRequest {
    policy_seed: u64,
    theta: Vec<f32>,
    env_seed: u64,
    steps: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct RolloutResult {
    env_seed: u64,
    episode_return: f32,
    steps: u32,
}

fn rollout_descriptor() -> OperationDescriptor {
    OperationDescriptor {
        key: OperationKey::new("rl", "rollout", 1),
        input_codec: Codec::BincodeV1,
        output_codec: Codec::BincodeV1,
        max_inline_arg_bytes: 64 * 1024,
    }
}

fn rollout_op() -> Operation<RolloutRequest, RolloutResult> {
    Operation::new(rollout_descriptor()).expect("valid descriptor")
}

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
                "rl",
            ],
            &args.artifact_dir,
        )?;
        workers.push(worker);
    }

    let mut client = ClusterClient::connect(&coordinator_addr);
    wait_coordinator(&mut client).await?;
    client.connect_epoch().await?;
    wait_ready(&client, args.workers).await?;

    let op = rollout_op();
    let mut policy = Policy {
        seed: 12345,
        theta: vec![0.1; args.policy_dim],
    };

    let mut samples = Vec::new();
    let mut total_episodes = 0usize;
    let start = Instant::now();

    for iteration in 0..args.iterations {
        let iter_start = Instant::now();

        let mut handles = Vec::new();
        for w in 0..args.parallelism {
            let env_seed = (args.seed_base + iteration as u64 * 1000 + w as u64) % (1 << 31);
            let req = RolloutRequest {
                policy_seed: policy.seed,
                theta: policy.theta.clone(),
                env_seed,
                steps: args.steps,
            };
            let req_bytes = bincode::serialize(&req)?;
            let handle = client
                .submit(
                    &op,
                    vec![TaskArg::Inline {
                        codec: Codec::BincodeV1,
                        bytes: req_bytes,
                    }],
                    ResourceSet::cpu_gpu(1.0, 0.0)?,
                    args.max_attempts,
                )
                .await?;
            handles.push(handle);
        }

        let mut returns = Vec::new();
        for handle in &handles {
            match handle.result(Duration::from_secs(30)).await {
                Ok(result) => returns.push(result.episode_return),
                Err(error) => eprintln!("iteration {iteration} task failed: {error}"),
            }
        }

        let mean_return = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f32>() / returns.len() as f32
        };

        // Simulate a policy update: nudge theta toward the mean return sign.
        for t in policy.theta.iter_mut() {
            *t += 0.01 * mean_return.tanh();
        }

        total_episodes += returns.len();
        let elapsed = iter_start.elapsed();
        samples.push(Sample {
            iteration,
            episodes: returns.len() as u32,
            mean_return,
            elapsed_ms: elapsed.as_millis() as u64,
        });

        if iteration % 10 == 0 || iteration == args.iterations - 1 {
            println!(
                "iter {iteration:4} | episodes {} | mean_return {mean_return:8.3} | {:6.1} it/s | total_episodes {total_episodes}",
                returns.len(),
                1.0 / elapsed.as_secs_f64()
            );
        }
    }

    let total = start.elapsed();
    println!(
        "DONE | iterations {} | total_episodes {total_episodes} | total_time {:.2}s | {:.2} episodes/s",
        args.iterations,
        total.as_secs_f64(),
        total_episodes as f64 / total.as_secs_f64()
    );

    for worker in workers.drain(..) {
        kill(worker);
    }
    kill(coordinator);

    write_artifacts(&args, &samples, total_episodes, total)?;
    Ok(())
}

#[derive(Serialize)]
struct Sample {
    iteration: u32,
    episodes: u32,
    mean_return: f32,
    elapsed_ms: u64,
}

struct Args {
    workers: usize,
    parallelism: usize,
    iterations: u32,
    steps: u32,
    policy_dim: usize,
    max_attempts: u32,
    seed_base: u64,
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
            .unwrap_or(2),
        parallelism: values
            .get("--parallelism")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8),
        iterations: values
            .get("--iterations")
            .and_then(|v| v.parse().ok())
            .unwrap_or(100),
        steps: values
            .get("--steps")
            .and_then(|v| v.parse().ok())
            .unwrap_or(100),
        policy_dim: values
            .get("--policy-dim")
            .and_then(|v| v.parse().ok())
            .unwrap_or(16),
        max_attempts: values
            .get("--max-attempts")
            .and_then(|v| v.parse().ok())
            .unwrap_or(3),
        seed_base: values
            .get("--seed")
            .and_then(|v| v.parse().ok())
            .unwrap_or(42),
        artifact_dir: values
            .get("--artifact-dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("benchmark_artifacts/rl")),
    }
}

async fn free_addr() -> Result<String, Error> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    Ok(listener.local_addr()?.to_string())
}

fn spawn(binary: &std::path::Path, args: &[&str], dir: &std::path::Path) -> Result<Child, Error> {
    let log = std::fs::File::create(dir.join(format!("process-{}.log", std::process::id())))?;
    let child = Command::new(binary)
        .args(args)
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()?;
    Ok(child)
}

fn kill(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

async fn wait_coordinator(client: &mut ClusterClient) -> Result<(), Error> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match client.connect_epoch().await {
            Ok(_) => return Ok(()),
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await
            }
            Err(error) => return Err(error),
        }
    }
}

async fn wait_ready(client: &ClusterClient, workers: usize) -> Result<(), Error> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match client.workers().await {
            Ok(list) if list.len() >= workers => return Ok(()),
            Ok(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await
            }
            Ok(_) => return Err(Error::Protocol("workers did not register in time".into())),
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await
            }
            Err(error) => return Err(error),
        }
    }
}

fn write_artifacts(
    args: &Args,
    samples: &[Sample],
    total_episodes: usize,
    total: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = serde_json::json!({
        "workers": args.workers,
        "parallelism": args.parallelism,
        "iterations": args.iterations,
        "steps": args.steps,
        "policy_dim": args.policy_dim,
        "max_attempts": args.max_attempts,
        "seed_base": args.seed_base,
        "total_episodes": total_episodes,
        "total_time_s": total.as_secs_f64(),
        "episodes_per_s": total_episodes as f64 / total.as_secs_f64(),
    });
    std::fs::write(
        args.artifact_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    let mut csv = String::from("iteration,episodes,mean_return,elapsed_ms\n");
    for s in samples {
        csv.push_str(&format!(
            "{},{},{},{}\n",
            s.iteration, s.episodes, s.mean_return, s.elapsed_ms
        ));
    }
    std::fs::write(args.artifact_dir.join("samples.csv"), csv)?;
    Ok(())
}
