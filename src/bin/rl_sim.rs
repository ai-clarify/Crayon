//! Realistic distributed RL training simulation.
//!
//! Models a typical on-policy RL pipeline (e.g. PPO) at scale:
//! - A parameter server actor holds the global policy weights
//! - N rollout workers generate trajectories using the current policy
//! - A trainer aggregates rollouts, computes a "gradient", and updates the PS
//! - Stale rollouts (from an outdated policy) are cancelled to free resources
//!
//! Run on a GPU node to validate GPU-aware scheduling:
//!   `cargo run --release --bin crayon-rl -- --workers 64 --steps 50`

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crayon::Ray;

/// Policy weights — a stand-in for a real neural network.
#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
struct Policy {
    weights: Vec<f32>,
    step: u64,
}

impl Policy {
    fn new(dim: usize) -> Self {
        Policy {
            weights: vec![0.0; dim],
            step: 0,
        }
    }

    /// "Forward pass" — dot product of weights with an observation.
    fn act(&self, obs: &[f32]) -> f32 {
        self.weights.iter().zip(obs).map(|(w, o)| w * o).sum()
    }

    /// Apply a gradient (simple SGD).
    fn apply_grad(&mut self, grad: &[f32], lr: f32) {
        for (w, g) in self.weights.iter_mut().zip(grad) {
            *w += lr * g;
        }
        self.step += 1;
    }
}

/// A rollout trajectory: observations, actions, and the policy version used.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Trajectory {
    policy_step: u64,
    obs: Vec<Vec<f32>>,
    actions: Vec<f32>,
    returns: Vec<f32>,
}

/// Rollout worker: generates a trajectory using the given policy weights.
/// Simulates environment interaction + forward passes.
fn rollout(policy: Policy, horizon: usize, obs_dim: usize) -> Trajectory {
    let mut rng = 0u64;
    let mut obs = vec![0.0f32; obs_dim];
    let mut traj = Trajectory {
        policy_step: policy.step,
        obs: Vec::with_capacity(horizon),
        actions: Vec::with_capacity(horizon),
        returns: Vec::with_capacity(horizon),
    };
    for _ in 0..horizon {
        // Pseudo-random observation
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for (i, o) in obs.iter_mut().enumerate() {
            *o = ((rng >> (i % 32)) & 0xff) as f32 / 255.0;
        }
        let action = policy.act(&obs);
        // Reward is a noisy function of the action (simulated environment)
        let reward = action.tanh() + ((rng & 0xff) as f32 / 255.0 - 0.5) * 0.1;
        traj.obs.push(obs.clone());
        traj.actions.push(action);
        traj.returns.push(reward);
    }
    traj
}

/// Compute a policy gradient from a batch of trajectories (simplified PPO).
fn compute_grad(trajectories: &[Trajectory], obs_dim: usize) -> Vec<f32> {
    let mut grad = vec![0.0f32; obs_dim];
    let mut count = 0u64;
    for traj in trajectories {
        for (obs, &ret) in traj.obs.iter().zip(traj.returns.iter()) {
            for (g, &o) in grad.iter_mut().zip(obs.iter()) {
                *g += o * ret;
            }
            count += 1;
        }
    }
    if count > 0 {
        for g in grad.iter_mut() {
            *g /= count as f32;
        }
    }
    grad
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let mut workers = 16usize;
    let mut steps = 20u64;
    let mut horizon = 128usize;
    let mut obs_dim = 64usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--workers" => {
                i += 1;
                workers = args[i].parse()?;
            }
            "--steps" => {
                i += 1;
                steps = args[i].parse()?;
            }
            "--horizon" => {
                i += 1;
                horizon = args[i].parse()?;
            }
            "--obs-dim" => {
                i += 1;
                obs_dim = args[i].parse()?;
            }
            _ => {}
        }
        i += 1;
    }

    println!("=== Crayon RL Simulation ===");
    println!("workers={workers} steps={steps} horizon={horizon} obs_dim={obs_dim}");

    // 4 workers, each with 1 CPU. Rollout + trainer tasks all use 1 CPU.
    let ray = Ray::init(4);

    let ps = ray.create_actor("ps", Policy::new(obs_dim));

    let total_samples = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    for step in 0..steps {
        // 1. Pull current policy from the parameter server
        let r = ps.call(|p| p.clone()).await.unwrap();
        let policy: Policy = ray.get(&r).await.unwrap();
        assert_eq!(policy.step, step, "PS step should match training step");

        // 2. Spawn N rollout workers in parallel (fan-out)
        let policy_ref = ray.put(policy.clone());
        let rollout_refs: Vec<_> = (0..workers)
            .map(|_| {
                let p = policy_ref.clone();
                ray.spawn((p,), move |(p,): (Policy,)| rollout(p, horizon, obs_dim))
            })
            .collect();

        // 3. Collect all rollouts (fan-in)
        let trajs: Vec<Trajectory> = ray
            .get_batch(&rollout_refs)
            .await
            .into_iter()
            .filter_map(|t| t.ok())
            .collect();
        assert_eq!(trajs.len(), workers, "all rollouts should succeed");

        let samples_this_step: u64 = trajs.iter().map(|t| t.obs.len() as u64).sum();
        total_samples.fetch_add(samples_this_step, Ordering::Relaxed);

        // 4. Compute gradient (CPU-bound in this simulation; in production this
        //    would be a GPU task using the actual GPU for backprop)
        let trajs_ref = ray.put(trajs.clone());
        let grad_ref = ray.spawn((trajs_ref,), move |(t,): (Vec<Trajectory>,)| {
            compute_grad(&t, obs_dim)
        });
        let grad: Vec<f32> = ray.get(&grad_ref).await.unwrap();

        // 5. Update the parameter server
        let lr = 0.01;
        let r = ps
            .call(move |p| {
                p.apply_grad(&grad, lr);
                p.step
            })
            .await
            .unwrap();
        let new_step: u64 = ray.get(&r).await.unwrap();
        assert_eq!(new_step, step + 1);

        if step % 5 == 0 || step == steps - 1 {
            let elapsed = start.elapsed();
            let sps = total_samples.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64();
            println!(
                "step {step:>3}/{steps} | samples={} | {:.0} samples/sec | elapsed={:.1?}",
                total_samples.load(Ordering::Relaxed),
                sps,
                elapsed
            );
        }
    }

    // Verify the policy actually changed (learning happened)
    let r = ps.call(|p| p.weights.clone()).await.unwrap();
    let final_weights: Vec<f32> = ray.get(&r).await.unwrap();
    let learned = final_weights.iter().any(|w| w.abs() > 1e-6);
    println!("\n=== Results ===");
    println!("policy learned: {learned}");
    println!(
        "total samples: {} in {:.1?} ({:.0} samples/sec)",
        total_samples.load(Ordering::Relaxed),
        start.elapsed(),
        total_samples.load(Ordering::Relaxed) as f64 / start.elapsed().as_secs_f64()
    );
    println!("status: {:?}", ray.status());

    if learned {
        println!("SUCCESS: RL training loop completed, policy weights updated");
        Ok(())
    } else {
        Err("policy weights did not change — training failed".into())
    }
}
