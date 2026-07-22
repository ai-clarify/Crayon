"""GRPO RL benchmark: Crayon vs Ray distribution backends.

Same PyTorch 0.6B transformer, same GRPO training logic, same hyperparameters.
The ONLY difference is the distribution framework (--backend crayon|ray).

Run:
    python benchmark_rl.py --backend crayon --steps 10 --workers 4 --batch 16
    python benchmark_rl.py --backend ray    --steps 10 --workers 4 --batch 16
"""

import argparse
import csv
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model config — 0.6B transformer (matches examples/llm_0_6b.rs)
# ---------------------------------------------------------------------------
VOCAB_SIZE = 1000
HIDDEN_DIM = 2048
NUM_LAYERS = 12
NUM_HEADS = 16
INTER_DIM = 8192

# Sequence config (configurable via CLI)
PROMPT_LEN = 16
GEN_LEN = 16
SEQ_LEN = PROMPT_LEN + GEN_LEN

# GRPO / slime defaults
LR = 1e-6
CLIP = 0.2          # PPO ratio clip
TEMPERATURE = 1.0
TOP_P = 1.0
CLIP_GRAD = 1.0
GROUP_SIZE = 4      # completions per prompt
EPS = 1e-8


# ---------------------------------------------------------------------------
# Transformer model (manual attention, matches Rust example architecture)
# ---------------------------------------------------------------------------
class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.w_k = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.w_v = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.w_o = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.w_ff1 = nn.Linear(HIDDEN_DIM, INTER_DIM, bias=False)
        self.w_ff2 = nn.Linear(INTER_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x):
        # x: [batch, seq, hidden]
        b, s, h = x.shape
        head_dim = h // NUM_HEADS
        q = self.w_q(x).view(b, s, NUM_HEADS, head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, s, NUM_HEADS, head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, s, NUM_HEADS, head_dim).transpose(1, 2)
        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, s, h)
        x = x + self.w_o(ctx)
        x = x + self.w_ff2(F.relu(self.w_ff1(x)))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(NUM_LAYERS)])
        self.w_out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=True)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        # tokens: [batch, seq] -> logits: [batch, vocab] (predicts next token)
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.w_out(x[:, -1, :])

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Rollout task — runs INSIDE workers (forward only, no training)
# ---------------------------------------------------------------------------
def rollout_task(state_dict_bytes, prompts_np):
    """Generate completions + log_probs for a batch of prompts.

    Args:
        state_dict_bytes: pickled model state_dict (auto-resolved from ObjectRef)
        prompts_np: [batch, PROMPT_LEN] int32 numpy array
    Returns:
        (completions [batch, GEN_LEN] int64, log_probs [batch, GEN_LEN] float32)
    """
    state_dict = pickle.loads(state_dict_bytes)
    model = Transformer()
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompts = torch.from_numpy(np.array(prompts_np, copy=True)).long().to(device)
    cur = prompts.clone()
    log_probs_list = []

    with torch.no_grad():
        for _ in range(GEN_LEN):
            logits = model(cur) / TEMPERATURE
            lp = F.log_softmax(logits, dim=-1)
            if TOP_P >= 1.0:
                probs = lp.exp()
            else:
                sorted_lp, sorted_idx = lp.sort(dim=-1, descending=True)
                cumprobs = sorted_lp.exp().cumsum(dim=-1)
                mask = cumprobs <= TOP_P
                mask[:, 0] = True
                sorted_lp = sorted_lp.masked_fill(~mask, float("-inf"))
                probs = torch.zeros_like(lp).scatter_(-1, sorted_idx, sorted_lp.exp())
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            tok_lp = lp.gather(1, next_tok.unsqueeze(-1)).squeeze(-1)
            log_probs_list.append(tok_lp)
            cur = torch.cat([cur, next_tok.unsqueeze(-1)], dim=1)

    completions = cur[:, PROMPT_LEN:].cpu().numpy().astype(np.int64)
    log_probs = torch.stack(log_probs_list, dim=1).cpu().numpy().astype(np.float32)
    return completions, log_probs


# ---------------------------------------------------------------------------
# Reward + GRPO advantage
# ---------------------------------------------------------------------------
def compute_reward(completions):
    """completions: [batch, GEN_LEN] -> rewards: [batch] float32.

    Token-sum reward normalized by vocab size (gives within-group variation).
    """
    return completions.sum(axis=1).astype(np.float32) / float(VOCAB_SIZE)


def compute_grpo_advantages(rewards, group_size):
    """Group-relative normalization: (r - mean_group) / (std_group + eps)."""
    n = len(rewards)
    adv = np.zeros_like(rewards)
    for i in range(0, n, group_size):
        g = rewards[i:i + group_size]
        adv[i:i + group_size] = (g - g.mean()) / (g.std() + EPS)
    return adv


# ---------------------------------------------------------------------------
# Trainer — aggregates rollouts, computes GRPO loss, updates model
# ---------------------------------------------------------------------------
def train_step(model, optimizer, all_prompts, all_completions, all_old_lps, group_size):
    """all_*: lists of numpy arrays from workers. Returns loss value."""
    prompts = np.concatenate(all_prompts, axis=0)
    completions = np.concatenate(all_completions, axis=0)
    old_lps = np.concatenate(all_old_lps, axis=0)

    rewards = compute_reward(completions)
    advantages = compute_grpo_advantages(rewards, group_size)

    device = next(model.parameters()).device
    prompts_t = torch.from_numpy(prompts).long().to(device)
    comp_t = torch.from_numpy(completions).long().to(device)
    old_lps_t = torch.from_numpy(old_lps).float().to(device)
    adv_t = torch.from_numpy(advantages).float().to(device).unsqueeze(-1)

    full = torch.cat([prompts_t, comp_t], dim=1)

    model.train()
    new_lps_list = []
    for step in range(comp_t.shape[1]):
        seq = full[:, : prompts_t.shape[1] + step + 1]
        logits = model(seq) / TEMPERATURE
        lp = F.log_softmax(logits, dim=-1)
        tok = comp_t[:, step]
        new_lps_list.append(lp.gather(1, tok.unsqueeze(-1)).squeeze(-1))
    new_lps = torch.stack(new_lps_list, dim=1)

    # PPO/GRPO clipped surrogate loss
    ratio = torch.exp(new_lps - old_lps_t)
    clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
    surr1 = ratio * adv_t
    surr2 = clipped * adv_t
    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Backend wrappers — same rollout work, different distribution framework
# ---------------------------------------------------------------------------
class CrayonBackend:
    def __init__(self, workers):
        import crayon
        self.ray = crayon.Ray(workers)

    def distribute(self, state_dict_bytes, prompt_chunks):
        state_ref = self.ray.put(state_dict_bytes)
        refs = [self.ray.spawn(rollout_task, state_ref, chunk) for chunk in prompt_chunks]
        results = self.ray.get_batch(refs)
        out = []
        for r in results:
            if isinstance(r, Exception):
                raise r
            out.append(r)
        return out

    def shutdown(self):
        pass  # crayon cleans up on GC


class RayBackend:
    def __init__(self, workers):
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=workers, ignore_reinit_error=True, include_dashboard=False)
        self._rollout_remote = ray.remote(rollout_task)

    def distribute(self, state_dict_bytes, prompt_chunks):
        import ray
        state_ref = ray.put(state_dict_bytes)
        refs = [self._rollout_remote.remote(state_ref, chunk) for chunk in prompt_chunks]
        return ray.get(refs)

    def shutdown(self):
        import ray
        if ray.is_initialized():
            ray.shutdown()


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------
def gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GRPO RL benchmark: crayon vs ray")
    parser.add_argument("--backend", choices=["crayon", "ray"], required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16, help="total samples per step")
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    parser.add_argument("--gen-len", type=int, default=GEN_LEN)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="CSV output path")
    args = parser.parse_args()

    # Apply sequence config overrides at module level so workers see them
    import sys
    _mod = sys.modules[__name__]
    _mod.PROMPT_LEN = args.prompt_len
    _mod.GEN_LEN = args.gen_len
    _mod.SEQ_LEN = args.prompt_len + args.gen_len

    backend = args.backend
    if args.output is None:
        args.output = f"benchmark_rl_{backend}.csv"

    print(f"=== GRPO RL Benchmark — backend={backend} ===")
    print(f"steps={args.steps} workers={args.workers} batch={args.batch} "
          f"group_size={args.group_size}")
    print(f"prompt_len={PROMPT_LEN} gen_len={GEN_LEN} seq_len={SEQ_LEN}")
    print(f"lr={args.lr} clip={CLIP} temperature={TEMPERATURE} "
          f"top_p={TOP_P} clip_grad={CLIP_GRAD}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Build model + optimizer
    model = Transformer().to(device)
    n_params = model.num_params()
    print(f"model params: {n_params:,} ({n_params / 1e9:.3f}B)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Sanity: batch must be divisible by group_size
    assert args.batch % args.group_size == 0, \
        f"batch ({args.batch}) must be divisible by group_size ({args.group_size})"
    num_prompts = args.batch // args.group_size

    # CSV setup
    csv_file = open(args.output, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "backend", "step", "step_time_ms", "samples_per_sec",
        "tokens_per_sec", "gpu_memory_mb", "loss", "num_samples",
    ])

    # Initialize distribution backend once (reused across all steps)
    if backend == "crayon":
        dist = CrayonBackend(args.workers)
    else:
        dist = RayBackend(args.workers)

    total_start = time.time()
    rng = np.random.default_rng(args.seed)

    for step in range(args.steps):
        step_start = time.time()

        # 1. Serialize current model weights
        state_dict_bytes = pickle.dumps(model.state_dict())

        # 2. Generate prompts (each prompt repeated group_size times)
        base_prompts = rng.integers(0, VOCAB_SIZE, size=(num_prompts, PROMPT_LEN), dtype=np.int32)
        prompts = np.repeat(base_prompts, args.group_size, axis=0)

        # 3. Split into chunks for workers
        chunk_size = (args.batch + args.workers - 1) // args.workers
        prompt_chunks = [prompts[i:i + chunk_size] for i in range(0, args.batch, chunk_size)]

        # 4. Fan out rollouts via chosen backend
        results = dist.distribute(state_dict_bytes, prompt_chunks)

        # 5. Unpack results
        all_prompts = []
        all_completions = []
        all_old_lps = []
        for i, chunk in enumerate(prompt_chunks):
            completions, log_probs = results[i]
            all_prompts.append(chunk)
            all_completions.append(completions)
            all_old_lps.append(log_probs)

        # 6. Train step (aggregate + GRPO + update)
        loss = train_step(
            model, optimizer, all_prompts, all_completions, all_old_lps,
            args.group_size,
        )

        # 7. Metrics
        step_time = time.time() - step_start
        step_ms = step_time * 1000.0
        samples_per_sec = args.batch / step_time
        tokens_per_sec = args.batch * GEN_LEN / step_time
        gpu_mem = gpu_memory_mb()

        writer.writerow([
            backend, step, f"{step_ms:.3f}", f"{samples_per_sec:.3f}",
            f"{tokens_per_sec:.3f}", f"{gpu_mem:.3f}", f"{loss:.6f}",
            args.batch,
        ])
        csv_file.flush()

        print(
            f"  step {step:>4}/{args.steps} | {step_ms:>8.1f}ms | "
            f"{samples_per_sec:>7.1f} samp/s | {tokens_per_sec:>8.1f} tok/s | "
            f"gpu={gpu_mem:>7.1f}MB | loss={loss:.4f}"
        )

    total_time = time.time() - total_start
    csv_file.close()

    print(f"\n=== Done: {args.steps} steps in {total_time:.1f}s ===")
    print(f"avg step: {total_time / args.steps * 1000:.0f}ms")
    print(f"avg throughput: {args.steps * args.batch / total_time:.1f} samp/s")
    print(f"results -> {args.output}")

    dist.shutdown()


if __name__ == "__main__":
    main()
