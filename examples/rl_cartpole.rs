//! Real distributed RL training with candle (neural network) + Crayon.
//!
//! Implements REINFORCE on CartPole:
//! - Policy network: MLP defined in candle
//! - Rollout workers (Crayon tasks): collect trajectories using current policy
//! - Parameter server (Crayon actor): holds global policy weights
//! - Trainer: aggregates rollouts, computes policy gradient, updates PS
//!
//! Run:
//!   cargo run --release --example rl_cartpole -- --steps 200 --workers 16

use std::time::Instant;

use candle_core::{Device, Tensor};
use crayon::Ray;
use rand::Rng;

// ---- CartPole environment (pure Rust, no gym dependency) ----

struct CartPole {
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
    steps: u32,
    max_steps: u32,
}

impl CartPole {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        CartPole {
            x: rng.gen_range(-0.05..0.05),
            x_dot: rng.gen_range(-0.05..0.05),
            theta: rng.gen_range(-0.05..0.05),
            theta_dot: rng.gen_range(-0.05..0.05),
            steps: 0,
            max_steps: 500,
        }
    }

    /// Returns (observation, reward, done).
    fn step(&mut self, action: u8) -> ([f32; 4], f32, bool) {
        let force = if action == 0 { -10.0 } else { 10.0 };
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let total_mass = mass_cart + mass_pole;
        let length = 0.5;
        let polemass_length = mass_pole * length;
        let gravity = 9.8;

        let temp =
            (force + polemass_length * self.theta.powi(2) * self.theta_dot.sin()) / total_mass;
        let theta_acc = (gravity * self.theta.sin() - temp * self.theta.cos())
            / (length * (4.0 / 3.0 - mass_pole * self.theta.cos().powi(2) / total_mass));
        let x_acc = temp - polemass_length * theta_acc * self.theta.cos() / total_mass;

        let dt = 0.02;
        self.x += self.x_dot * dt;
        self.x_dot += x_acc * dt;
        self.theta += self.theta_dot * dt;
        self.theta_dot += theta_acc * dt;
        self.steps += 1;

        let obs = [self.x, self.x_dot, self.theta, self.theta_dot];
        let done = self.theta.abs() > 12.0 * std::f32::consts::PI / 180.0
            || self.x.abs() > 2.4
            || self.steps >= self.max_steps;
        let reward = if done { 0.0 } else { 1.0 };
        (obs, reward, done)
    }
}

// ---- Policy network (candle MLP, manual layers) ----

struct PolicyNet {
    w1: Tensor, // [64, 4]
    b1: Tensor, // [64]
    w2: Tensor, // [2, 64]
    b2: Tensor, // [2]
}

impl PolicyNet {
    fn new(device: &Device) -> candle_core::Result<Self> {
        let w1 = Tensor::randn(0f32, 0.1, (64, 4), device)?;
        let b1 = Tensor::zeros((64,), candle_core::DType::F32, device)?;
        let w2 = Tensor::randn(0f32, 0.1, (2, 64), device)?;
        let b2 = Tensor::zeros((2,), candle_core::DType::F32, device)?;
        Ok(PolicyNet { w1, b1, w2, b2 })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.matmul(&self.w1.t()?)?;
        let x = (x + self.b1.unsqueeze(0)?)?;
        let x = x.relu()?;
        let x = x.matmul(&self.w2.t()?)?;
        let x = (x + self.b2.unsqueeze(0)?)?;
        Ok(x)
    }

    /// Sample an action from the policy. Returns (action, log_prob).
    fn act(&self, obs: &[f32; 4], device: &Device) -> candle_core::Result<(u8, f32)> {
        let obs_t = Tensor::new(obs, device)?.unsqueeze(0)?;
        let logits = self.forward(&obs_t)?;
        let logits = logits.squeeze(0)?.to_vec1::<f32>()?;

        // Manual softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|l| (l - max_logit).exp() / exp_sum)
            .collect();

        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cum = 0.0;
        let mut action = 0u8;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                action = i as u8;
                break;
            }
        }
        let log_prob = probs[action as usize].ln();
        Ok((action, log_prob))
    }

    /// Get all parameters as a flat vector (for serialization to PS).
    fn params_flat(&self) -> Vec<f32> {
        let mut params = Vec::new();
        params.extend(self.w1.to_vec2::<f32>().unwrap().into_iter().flatten());
        params.extend(self.b1.to_vec1::<f32>().unwrap());
        params.extend(self.w2.to_vec2::<f32>().unwrap().into_iter().flatten());
        params.extend(self.b2.to_vec1::<f32>().unwrap());
        params
    }

    /// Create a PolicyNet from a flat parameter vector.
    fn from_params(params: &[f32], device: &Device) -> candle_core::Result<Self> {
        let mut offset = 0;
        let w1_data: Vec<f32> = params[offset..offset + 64 * 4].to_vec();
        offset += 64 * 4;
        let w1 = Tensor::from_vec(w1_data, (64, 4), device)?;

        let b1_data: Vec<f32> = params[offset..offset + 64].to_vec();
        offset += 64;
        let b1 = Tensor::from_vec(b1_data, (64,), device)?;

        let w2_data: Vec<f32> = params[offset..offset + 2 * 64].to_vec();
        offset += 2 * 64;
        let w2 = Tensor::from_vec(w2_data, (2, 64), device)?;

        let b2_data: Vec<f32> = params[offset..offset + 2].to_vec();
        let b2 = Tensor::from_vec(b2_data, (2,), device)?;

        Ok(PolicyNet { w1, b1, w2, b2 })
    }
}

// ---- Rollout: collect a trajectory using the given policy params ----

/// A collected rollout: (observations, actions, rewards, total_reward)
type Rollout = (Vec<[f32; 4]>, Vec<u8>, Vec<f32>, f32);

fn rollout(params: Vec<f32>, max_steps: u32) -> Rollout {
    let device = Device::Cpu;
    let policy = PolicyNet::from_params(&params, &device).unwrap();

    let mut env = CartPole::new();
    let mut obs_list = Vec::new();
    let mut action_list = Vec::new();
    let mut reward_list = Vec::new();
    let mut total_reward = 0.0;

    for _ in 0..max_steps {
        let obs = [env.x, env.x_dot, env.theta, env.theta_dot];
        let (action, _log_prob) = policy.act(&obs, &device).unwrap();
        obs_list.push(obs);
        action_list.push(action);

        let (_, reward, done) = env.step(action);
        reward_list.push(reward);
        total_reward += reward;
        if done {
            break;
        }
    }

    (obs_list, action_list, reward_list, total_reward)
}

// ---- Compute discounted returns ----

fn compute_returns(rewards: &[f32], gamma: f32) -> Vec<f32> {
    let mut returns = vec![0.0; rewards.len()];
    let mut running = 0.0;
    for i in (0..rewards.len()).rev() {
        running = rewards[i] + gamma * running;
        returns[i] = running;
    }
    returns
}

// ---- Parameter server state ----

#[derive(Clone)]
struct PSState {
    params: Vec<f32>,
    step: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let mut steps = 100u64;
    let mut workers = 8usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                i += 1;
                steps = args[i].parse()?;
            }
            "--workers" => {
                i += 1;
                workers = args[i].parse()?;
            }
            _ => {}
        }
        i += 1;
    }

    println!("=== Crayon Real RL (candle + CartPole + REINFORCE) ===");
    println!("steps={steps} workers={workers}");

    let ray = Ray::init(4);
    let device = Device::Cpu;

    // Initialize policy network and put params in the PS
    let policy = PolicyNet::new(&device)?;
    let init_params = policy.params_flat();
    println!("policy params: {} floats", init_params.len());

    let ps = ray.create_actor(
        "ps",
        PSState {
            params: init_params,
            step: 0,
        },
    );

    let gamma = 0.99;
    let lr = 0.01;
    let start = Instant::now();
    let mut total_episodes = 0u64;

    for step in 0..steps {
        // 1. Get current params from PS
        let r = ps.call(|s| s.params.clone()).await.unwrap();
        let params: Vec<f32> = ray.get(&r).await.unwrap();

        // 2. Fan out: collect rollouts in parallel
        let rollout_refs: Vec<_> = (0..workers)
            .map(|_| {
                let p = params.clone();
                ray.spawn((), move |()| rollout(p.clone(), 500))
            })
            .collect();

        // 3. Fan in: collect all trajectories
        let results: Vec<Rollout> = ray
            .get_batch(&rollout_refs)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        let avg_reward: f32 = results.iter().map(|r| r.3).sum::<f32>() / results.len() as f32;
        total_episodes += results.len() as u64;

        // 4. Compute policy gradient (aggregate all rollouts)
        let mut grad = vec![0.0f32; params.len()];
        let mut count = 0u32;

        for (obs_list, action_list, reward_list, _) in &results {
            let returns = compute_returns(reward_list, gamma);
            let mean: f32 = returns.iter().sum::<f32>() / returns.len() as f32;
            let std: f32 = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f32>()
                / returns.len() as f32)
                .sqrt()
                + 1e-8;
            let advantages: Vec<f32> = returns.iter().map(|r| (r - mean) / std).collect();

            // Numerical gradient: finite differences on log_prob * advantage
            let eps = 1e-3;
            for (pi, g) in grad.iter_mut().enumerate() {
                // Perturb param up
                let mut params_up = params.clone();
                params_up[pi] += eps;
                let policy_up = PolicyNet::from_params(&params_up, &device).unwrap();
                let mut lp_up = 0.0;
                for (obs, &action) in obs_list.iter().zip(action_list.iter()) {
                    let (_, lp) = policy_up.act(obs, &device).unwrap();
                    if action == 0 {
                        lp_up += lp;
                    }
                }

                // Perturb param down
                let mut params_down = params.clone();
                params_down[pi] -= eps;
                let policy_down = PolicyNet::from_params(&params_down, &device).unwrap();
                let mut lp_down = 0.0;
                for (obs, &action) in obs_list.iter().zip(action_list.iter()) {
                    let (_, lp) = policy_down.act(obs, &device).unwrap();
                    if action == 0 {
                        lp_down += lp;
                    }
                }

                let d_log_prob = (lp_up - lp_down) / (2.0 * eps);
                let adv_sum: f32 = advantages.iter().sum();
                *g += d_log_prob * adv_sum;
                count += 1;
            }
        }

        // Average gradient
        if count > 0 {
            for g in grad.iter_mut() {
                *g /= count as f32;
            }
        }

        // 5. Update params (SGD)
        let mut new_params = params.clone();
        for (p, g) in new_params.iter_mut().zip(grad.iter()) {
            *p += lr * g;
        }

        // 6. Push new params to PS
        let new_params_clone = new_params.clone();
        let r = ps
            .call(move |s| {
                s.params = new_params_clone;
                s.step += 1;
                s.step
            })
            .await
            .unwrap();
        let new_step: u64 = ray.get(&r).await.unwrap();
        assert_eq!(new_step, step + 1);

        if step % 10 == 0 || step == steps - 1 {
            let elapsed = start.elapsed();
            println!(
                "step {step:>4}/{steps} | avg_reward={avg_reward:>6.1} | episodes={total_episodes:>5} | {:.1?}",
                elapsed
            );
        }
    }

    println!("\n=== Done ===");
    println!(
        "trained {steps} steps, {total_episodes} episodes in {:.1?}",
        start.elapsed()
    );
    println!("status: {:?}", ray.status());
    Ok(())
}
