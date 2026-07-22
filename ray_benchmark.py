"""Ray equivalent of Crayon's arithmetic learning benchmark.

Same task: learn a+b=c (0-4 + 0-4) with a simple model:
  embed(a) + embed(b) -> linear -> answer token
  Supervised cross-entropy, analytic gradients.

Run: python ray_benchmark.py --steps 200 --workers 4 --rounds 3
"""

import argparse
import time
import csv
import numpy as np
import ray

# Vocab: 0=pad, 1=eos, 2-11=digits 0-9, 12=+, 13==
VOCAB_SIZE = 14
PAD = 0
PLUS = 12
EQ = 13
HIDDEN_DIM = 64
SEQ_LEN = 8


def make_problem(rng):
    a = rng.integers(0, 5)
    b = rng.integers(0, 5)
    return int(a), int(b), int(a + b)


def encode_prompt(a, b):
    p = [a + 2, PLUS, b + 2, EQ]
    while len(p) < SEQ_LEN:
        p.append(PAD)
    return p


class PolicyNet:
    def __init__(self, rng):
        self.embed = rng.normal(0, 0.1, (VOCAB_SIZE, HIDDEN_DIM)).astype(np.float32)
        self.w_out = rng.normal(0, 0.1, (HIDDEN_DIM, VOCAB_SIZE)).astype(np.float32)
        self.b_out = np.zeros(VOCAB_SIZE, dtype=np.float32)

    def params_flat(self):
        return np.concatenate([self.embed.ravel(), self.w_out.ravel(), self.b_out])

    @staticmethod
    def from_params(params):
        net = PolicyNet(np.random.default_rng(0))
        e = VOCAB_SIZE * HIDDEN_DIM
        w = HIDDEN_DIM * VOCAB_SIZE
        net.embed = params[:e].reshape(VOCAB_SIZE, HIDDEN_DIM)
        net.w_out = params[e:e+w].reshape(HIDDEN_DIM, VOCAB_SIZE)
        net.b_out = params[e+w:]
        return net

    def forward(self, tokens):
        # tokens: [batch, seq]
        emb = self.embed[tokens]  # [batch, seq, hidden]
        pos0 = emb[:, 0, :]  # [batch, hidden]
        pos2 = emb[:, 2, :]  # [batch, hidden]
        pooled = pos0 + pos2  # [batch, hidden]
        logits = pooled @ self.w_out + self.b_out  # [batch, vocab]
        return logits


@ray.remote
def forward_batch(params, problems):
    net = PolicyNet.from_params(params)
    batch = len(problems)
    tokens = np.array([encode_prompt(a, b) for a, b, _ in problems], dtype=np.int32)
    logits = net.forward(tokens)
    return logits


@ray.remote
class PS:
    def __init__(self, params):
        self.params = params
        self.step = 0

    def get_params(self):
        return self.params

    def update(self, grad, lr):
        self.params = self.params - lr * grad
        self.step += 1
        return self.step


def compute_gradient(params, logits, targets, problems):
    batch = logits.shape[0]
    net = PolicyNet.from_params(params)

    # Softmax
    mx = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - mx)
    softmax = exp / exp.sum(axis=1, keepdims=True)

    # dL/dlogits = (softmax - onehot) / batch
    onehot = np.zeros_like(logits)
    for i, t in enumerate(targets):
        onehot[i, t] = 1.0
    dlogits = (softmax - onehot) / batch

    # Recompute embeddings
    tokens = np.array([encode_prompt(a, b) for a, b, _ in problems], dtype=np.int32)
    emb = net.embed[tokens]  # [batch, seq, hidden]
    pos0 = emb[:, 0, :]
    pos2 = emb[:, 2, :]
    pooled = pos0 + pos2  # [batch, hidden]

    # dL/dw_out = pooled^T @ dlogits
    gw = pooled.T @ dlogits  # [hidden, vocab]
    # dL/db_out = sum(dlogits, axis=0)
    gb = dlogits.sum(axis=0)  # [vocab]

    # dL/dpooled = dlogits @ w_out^T
    dpooled = dlogits @ net.w_out.T  # [batch, hidden]

    # dL/dembed: scatter to positions 0 and 2
    gembed = np.zeros_like(net.embed)
    for i in range(batch):
        tok0 = tokens[i, 0]
        tok2 = tokens[i, 2]
        gembed[tok0] += dpooled[i]
        gembed[tok2] += dpooled[i]

    grad = np.concatenate([gembed.ravel(), gw.ravel(), gb])
    return grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    ray.init(num_cpus=args.workers, ignore_reinit_error=True)

    print(f"=== Ray Distributed Arithmetic Learning ===")
    print(f"rounds={args.rounds} steps={args.steps} workers={args.workers} batch={args.batch}")

    rng = np.random.default_rng(42)
    lr = 0.1

    with open("ray_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "step", "accuracy", "loss", "step_time_ms", "samples_per_sec"])

        for run in range(args.rounds):
            print(f"\n--- Round {run}/{args.rounds} ---")
            init_net = PolicyNet(rng)
            init_params = init_net.params_flat()
            if run == 0:
                print(f"params: {len(init_params)}")

            ps = PS.remote(init_params)
            run_start = time.time()
            acc_hist = []

            for step in range(args.steps):
                step_start = time.time()

                params = ray.get(ps.get_params.remote())

                problems = [make_problem(rng) for _ in range(args.batch)]
                targets = [c + 2 for _, _, c in problems]

                # Distributed forward: split batch across workers
                chunk = (args.batch + args.workers - 1) // args.workers
                refs = []
                for i in range(0, args.batch, chunk):
                    refs.append(forward_batch.remote(params, problems[i:i+chunk]))
                logits_list = ray.get(refs)
                logits = np.concatenate(logits_list, axis=0)

                # Accuracy
                preds = logits.argmax(axis=1)
                correct = sum(1 for p, t in zip(preds, targets) if p == t)
                acc = correct / len(targets)
                acc_hist.append(acc)

                # Loss
                mx = logits.max(axis=1, keepdims=True)
                log_probs = logits - mx - np.log(np.exp(logits - mx).sum(axis=1, keepdims=True))
                loss = -np.mean([log_probs[i, t] for i, t in enumerate(targets)])

                # Gradient
                grad = compute_gradient(params, logits, targets, problems)

                # Update
                new_step = ray.get(ps.update.remote(grad, lr))
                assert new_step == step + 1

                step_ms = (time.time() - step_start) * 1000
                sps = args.batch / (time.time() - step_start)
                writer.writerow([run, step, f"{acc:.6f}", f"{loss:.6f}", f"{step_ms:.3f}", f"{sps:.3f}"])
                f.flush()

                if step % 20 == 0 or step == args.steps - 1:
                    print(f"  step {step:>4}/{args.steps} | acc={acc:.3f} | loss={loss:.3f} | {step_ms:.0f}ms | {sps:.0f} samp/s")

            t = time.time() - run_start
            max_acc = max(acc_hist)
            mean_acc = sum(acc_hist) / len(acc_hist)
            final_acc = acc_hist[-1]
            print(f"  Round {run}: {t:.1}s | mean={mean_acc:.3f} max={max_acc:.3f} final={final_acc:.3f}")

    print(f"\n=== Done. Results -> ray_results.csv ===")
    ray.shutdown()


if __name__ == "__main__":
    main()
