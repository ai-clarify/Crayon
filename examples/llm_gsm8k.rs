//! LLM RL on GSM8K — distributed rollout + REINFORCE with Crayon + candle.
//!
//! Pipeline:
//! - A mini Transformer language model (candle) generates solutions to GSM8K math problems
//! - Reward: extract the final numeric answer, +1 if correct, 0 otherwise
//! - Distributed rollout: N workers generate solutions in parallel (Crayon tasks)
//! - Trainer: aggregates rollouts, computes REINFORCE gradient, updates the policy
//!
//! Run:
//!   cargo run --release --example llm_gsm8k -- --steps 50 --workers 8

use std::time::Instant;

use candle_core::{Device, Tensor};
use crayon::Ray;
use rand::Rng;

// ---- GSM8K dataset (subset) ----

struct Gsm8kProblem {
    question: &'static str,
    answer: f64,
}

const PROBLEMS: &[Gsm8kProblem] = &[
    Gsm8kProblem { question: "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder at $2 each. How much does she make per day?", answer: 18.0 },
    Gsm8kProblem { question: "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?", answer: 3.0 },
    Gsm8kProblem { question: "Josh decides to flip a coin to decide what to do. If it's heads he goes to the park. If it's tails he goes to the movies. The probability of heads is 50%. What is the probability he goes to the movies?", answer: 0.5 },
    Gsm8kProblem { question: "Tina makes $18 an hour. If she works more than 8 hours per day, she gets paid 1.5 times her wage. If she works 10 hours, how much does she earn?", answer: 204.0 },
    Gsm8kProblem { question: "Mark has 3 apples. He buys 5 more apples. How many apples does he have?", answer: 8.0 },
    Gsm8kProblem { question: "A store sells 12 pencils per box. If a teacher buys 5 boxes, how many pencils does she have?", answer: 60.0 },
    Gsm8kProblem { question: "Tom drives 60 miles per hour for 2.5 hours. How far does he travel?", answer: 150.0 },
    Gsm8kProblem { question: "A pizza is cut into 8 slices. If 3 people eat 2 slices each, how many slices are left?", answer: 2.0 },
    Gsm8kProblem { question: "Sarah has $40. She buys a book for $15 and a pen for $5. How much money does she have left?", answer: 20.0 },
    Gsm8kProblem { question: "A factory produces 250 widgets per hour. How many widgets does it produce in 8 hours?", answer: 2000.0 },
];

// ---- Mini Transformer language model ----

const VOCAB_SIZE: usize = 100; // simplified vocab: digits 0-9, operators, etc.
const HIDDEN_DIM: usize = 64;
const SEQ_LEN: usize = 32;

struct PolicyNet {
    embed: Tensor, // [VOCAB_SIZE, HIDDEN_DIM]
    w_q: Tensor,   // [HIDDEN_DIM, HIDDEN_DIM]
    w_k: Tensor,   // [HIDDEN_DIM, HIDDEN_DIM]
    w_v: Tensor,   // [HIDDEN_DIM, HIDDEN_DIM]
    w_out: Tensor, // [HIDDEN_DIM, VOCAB_SIZE]
    b_out: Tensor, // [VOCAB_SIZE]
}

impl PolicyNet {
    fn new(device: &Device) -> candle_core::Result<Self> {
        let embed = Tensor::randn(0f32, 0.1, (VOCAB_SIZE, HIDDEN_DIM), device)?;
        let w_q = Tensor::randn(0f32, 0.1, (HIDDEN_DIM, HIDDEN_DIM), device)?;
        let w_k = Tensor::randn(0f32, 0.1, (HIDDEN_DIM, HIDDEN_DIM), device)?;
        let w_v = Tensor::randn(0f32, 0.1, (HIDDEN_DIM, HIDDEN_DIM), device)?;
        let w_out = Tensor::randn(0f32, 0.1, (HIDDEN_DIM, VOCAB_SIZE), device)?;
        let b_out = Tensor::zeros((VOCAB_SIZE,), candle_core::DType::F32, device)?;
        Ok(PolicyNet {
            embed,
            w_q,
            w_k,
            w_v,
            w_out,
            b_out,
        })
    }

    /// Forward pass: given token ids [batch, seq], return logits [batch, seq, VOCAB]
    fn forward(&self, tokens: &Tensor) -> candle_core::Result<Tensor> {
        // tokens: [batch, seq]
        let batch = tokens.dim(0)?;
        let seq = tokens.dim(1)?;

        // Embed: [batch, seq, HIDDEN]
        let tok_flat = tokens.flatten(0, 1)?;
        let emb = tok_flat.index_select(&self.embed, 0)?;
        let emb = emb.reshape((batch, seq, HIDDEN_DIM))?;

        // Self-attention (simplified, no causal mask for brevity)
        let emb_flat = emb.flatten(0, 1)?; // [batch*seq, HIDDEN]
        let q = emb_flat.matmul(&self.w_q)?;
        let k = emb_flat.matmul(&self.w_k)?;
        let v = emb_flat.matmul(&self.w_v)?;

        // Scaled dot-product attention (simplified, no causal mask for brevity)
        let q = q.reshape((batch, seq, HIDDEN_DIM))?;
        let k = k.reshape((batch, seq, HIDDEN_DIM))?;
        let v = v.reshape((batch, seq, HIDDEN_DIM))?;

        let scale = 1.0 / (HIDDEN_DIM as f32).sqrt();
        let scale_t = Tensor::new(scale, tokens.device())?;
        let scores = q.matmul(&k.t()?)?;
        let scores = (scores * scale_t)?; // [batch, seq, seq]
                                          // Manual softmax over last dim
        let scores_flat = scores.flatten(0, 1)?; // [batch*seq, seq]
        let max_vals = scores_flat.max(1)?; // [batch*seq, 1]
        let scores_shifted = (scores_flat - max_vals)?;
        let exp_scores = scores_shifted.exp()?;
        let sum_exp = exp_scores.sum(1)?; // [batch*seq, 1]
        let attn_flat = (exp_scores / sum_exp)?;
        let attn = attn_flat.reshape((batch, seq, seq))?;
        let ctx = attn.matmul(&v)?; // [batch, seq, HIDDEN]

        // Output projection
        let ctx_flat = ctx.flatten(0, 1)?;
        let logits = ctx_flat.matmul(&self.w_out)?;
        let logits = (logits + self.b_out.unsqueeze(0)?)?;
        let logits = logits.reshape((batch, seq, VOCAB_SIZE))?;
        Ok(logits)
    }

    /// Sample a sequence of tokens. Returns (token_ids, log_probs).
    fn sample(
        &self,
        prompt: &[usize],
        max_len: usize,
        device: &Device,
    ) -> candle_core::Result<(Vec<usize>, Vec<f32>)> {
        let mut tokens = prompt.to_vec();
        let mut log_probs = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..max_len {
            if tokens.len() >= SEQ_LEN {
                break;
            }
            // Pad to SEQ_LEN
            let mut input = tokens.clone();
            while input.len() < SEQ_LEN {
                input.push(0);
            }
            let input_t = Tensor::from_vec(
                input.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                (1, SEQ_LEN),
                device,
            )?;
            // We need integer tokens for embedding lookup. Convert to u32.
            let input_t = input_t.to_dtype(candle_core::DType::U32)?;

            let logits = self.forward(&input_t)?;
            let pos = tokens.len() - 1;
            let logits = logits.narrow(1, pos, 1)?.squeeze(1)?; // [1, VOCAB]
            let logits = logits.squeeze(0)?; // [VOCAB]

            // Softmax
            let logits_vec = logits.to_vec1::<f32>()?;
            let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits_vec.iter().map(|l| (l - max_l).exp()).sum();
            let probs: Vec<f32> = logits_vec
                .iter()
                .map(|l| (l - max_l).exp() / exp_sum)
                .collect();

            // Sample
            let r: f32 = rng.gen();
            let mut cum = 0.0;
            let mut next_token = 0usize;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if r < cum {
                    next_token = i;
                    break;
                }
            }
            log_probs.push(probs[next_token].ln());
            tokens.push(next_token);

            // EOS token is 1
            if next_token == 1 {
                break;
            }
        }
        Ok((tokens, log_probs))
    }

    fn params_flat(&self) -> Vec<f32> {
        let mut p = Vec::new();
        p.extend(self.embed.to_vec2::<f32>().unwrap().into_iter().flatten());
        p.extend(self.w_q.to_vec2::<f32>().unwrap().into_iter().flatten());
        p.extend(self.w_k.to_vec2::<f32>().unwrap().into_iter().flatten());
        p.extend(self.w_v.to_vec2::<f32>().unwrap().into_iter().flatten());
        p.extend(self.w_out.to_vec2::<f32>().unwrap().into_iter().flatten());
        p.extend(self.b_out.to_vec1::<f32>().unwrap());
        p
    }

    fn from_params(params: &[f32], device: &Device) -> candle_core::Result<Self> {
        let mut off = 0;
        let mut n = |len: usize| {
            let v = params[off..off + len].to_vec();
            off += len;
            v
        };

        let embed = Tensor::from_vec(n(VOCAB_SIZE * HIDDEN_DIM), (VOCAB_SIZE, HIDDEN_DIM), device)?;
        let w_q = Tensor::from_vec(n(HIDDEN_DIM * HIDDEN_DIM), (HIDDEN_DIM, HIDDEN_DIM), device)?;
        let w_k = Tensor::from_vec(n(HIDDEN_DIM * HIDDEN_DIM), (HIDDEN_DIM, HIDDEN_DIM), device)?;
        let w_v = Tensor::from_vec(n(HIDDEN_DIM * HIDDEN_DIM), (HIDDEN_DIM, HIDDEN_DIM), device)?;
        let w_out = Tensor::from_vec(n(HIDDEN_DIM * VOCAB_SIZE), (HIDDEN_DIM, VOCAB_SIZE), device)?;
        let b_out = Tensor::from_vec(n(VOCAB_SIZE), (VOCAB_SIZE,), device)?;
        Ok(PolicyNet {
            embed,
            w_q,
            w_k,
            w_v,
            w_out,
            b_out,
        })
    }
}

// ---- Tokenize / detokenize (simplified) ----

fn tokenize(text: &str) -> Vec<usize> {
    // Simple tokenization: map chars to token ids
    // 0 = pad, 1 = eos, 2-11 = digits 0-9, 12+ = other chars
    let mut tokens = vec![2]; // BOS-ish
    for c in text.chars() {
        let t = if c.is_ascii_digit() {
            2 + (c.to_digit(10).unwrap() as usize)
        } else if c == ' ' {
            12
        } else if c == '.' {
            13
        } else if c == '+' {
            14
        } else if c == '-' {
            15
        } else if c == '*' {
            16
        } else if c == '/' {
            17
        } else if c == '=' {
            18
        } else {
            19 + ((c as usize) % 80)
        };
        tokens.push(t);
    }
    tokens.push(1); // EOS
    tokens
}

fn extract_answer(tokens: &[usize]) -> Option<f64> {
    // Extract the last number from the token sequence
    let mut num_str = String::new();
    for &t in tokens.iter().rev() {
        if (2..=11).contains(&t) {
            num_str.insert(0, char::from_digit((t - 2) as u32, 10).unwrap());
        } else if t == 13 && !num_str.is_empty() {
            num_str.insert(0, '.');
        } else if !num_str.is_empty() {
            break;
        }
    }
    num_str.parse::<f64>().ok()
}

// ---- Rollout: generate a solution and compute reward ----

fn rollout(params: Vec<f32>, problem_idx: usize) -> (Vec<usize>, Vec<f32>, f64) {
    let device = Device::Cpu;
    let policy = PolicyNet::from_params(&params, &device).unwrap();
    let problem = &PROBLEMS[problem_idx % PROBLEMS.len()];
    let prompt = tokenize(problem.question);

    let (tokens, log_probs) = policy.sample(&prompt, 20, &device).unwrap();

    let predicted = extract_answer(&tokens);
    let reward = match predicted {
        Some(p) if (p - problem.answer).abs() < 0.01 => 1.0,
        _ => 0.0,
    };

    (tokens, log_probs, reward)
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
    let mut steps = 50u64;
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

    println!("=== Crayon LLM RL (GSM8K + mini Transformer + REINFORCE) ===");
    println!(
        "steps={steps} workers={workers} problems={}",
        PROBLEMS.len()
    );

    let ray = Ray::init(4);
    let device = Device::Cpu;

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

    let lr = 0.01;
    let start = Instant::now();
    let mut total_reward = 0.0;
    let mut total_rollouts = 0u64;

    for step in 0..steps {
        // 1. Get current params
        let r = ps.call(|s| s.params.clone()).await.unwrap();
        let params: Vec<f32> = ray.get(&r).await.unwrap();

        // 2. Fan out: rollouts on different problems
        let rollout_refs: Vec<_> = (0..workers)
            .map(|i| {
                let p = params.clone();
                ray.spawn((), move |()| rollout(p.clone(), i))
            })
            .collect();

        // 3. Fan in: collect rollouts
        let results: Vec<(Vec<usize>, Vec<f32>, f64)> = ray
            .get_batch(&rollout_refs)
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        let avg_reward: f64 = results.iter().map(|r| r.2).sum::<f64>() / results.len() as f64;
        total_reward += results.iter().map(|r| r.2).sum::<f64>();
        total_rollouts += results.len() as u64;

        // 4. Compute REINFORCE gradient (numerical, per-parameter finite differences)
        let mut grad = vec![0.0f32; params.len()];
        let eps = 1e-3;

        for (pi, g) in grad.iter_mut().enumerate() {
            // Perturb up
            let mut params_up = params.clone();
            params_up[pi] += eps;
            let r_up = rollout(params_up, step as usize % PROBLEMS.len());

            // Perturb down
            let mut params_down = params.clone();
            params_down[pi] -= eps;
            let r_down = rollout(params_down, step as usize % PROBLEMS.len());

            // REINFORCE: dJ/dtheta ≈ (R_up - R_down) / (2*eps)
            *g = (r_up.2 - r_down.2) as f32 / (2.0 * eps);
        }

        // 5. Update params (SGD)
        let mut new_params = params.clone();
        for (p, g) in new_params.iter_mut().zip(grad.iter()) {
            *p += lr * g;
        }

        // 6. Push to PS
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

        if step % 5 == 0 || step == steps - 1 {
            println!(
                "step {step:>4}/{steps} | avg_reward={avg_reward:.3} | mean_reward={:.3} | {:.1?}",
                total_reward / total_rollouts as f64,
                start.elapsed()
            );
        }
    }

    println!("\n=== Done ===");
    println!(
        "trained {steps} steps, {total_rollouts} rollouts, mean_reward={:.3}",
        total_reward / total_rollouts as f64
    );
    println!("status: {:?}", ray.status());
    Ok(())
}
