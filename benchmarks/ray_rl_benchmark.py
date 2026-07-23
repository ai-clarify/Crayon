"""Ray equivalent of Crayon's rl_benchmark for apples-to-apples comparison.

Same workload: seeded bandit env, linear policy, distributed rollout collection,
per-iteration policy update. Outputs manifest.json + samples.csv matching Crayon.
"""
import argparse
import csv
import json
import math
import os
import time

import ray


@ray.remote
def rollout(policy_seed, theta, env_seed, steps):
    # Exact replica of Crayon's run_rollout (LCG bandit) so the per-task compute
    # is identical and we measure system overhead, not algorithm speed.
    state = (env_seed ^ policy_seed) & 0xFFFFFFFFFFFFFFFF
    n = max(len(theta), 1)
    total = 0.0
    for _ in range(steps):
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        action = (state >> 33) % n
        reward = ((state >> 1) & 0xFF) / 255.0
        total += reward * theta[action]
    return {"env_seed": env_seed, "episode_return": total, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parallelism", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--policy-dim", type=int, default=32)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifact-dir", type=str, default="benchmark_artifacts/rl")
    args = ap.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)

    ray.init(num_cpus=args.workers, ignore_reinit_error=True)

    theta = [0.1] * args.policy_dim
    policy_seed = 12345

    samples = []
    total_episodes = 0
    start = time.monotonic()

    for iteration in range(args.iterations):
        iter_start = time.monotonic()
        refs = []
        for w in range(args.parallelism):
            env_seed = (args.seed + iteration * 1000 + w) % (1 << 31)
            refs.append(rollout.remote(policy_seed, theta, env_seed, args.steps))

        returns = []
        for ref in refs:
            try:
                res = ray.get(ref, timeout=30)
                returns.append(res["episode_return"])
            except Exception as e:
                print(f"iteration {iteration} task failed: {e}")

        mean_return = sum(returns) / len(returns) if returns else 0.0
        # same policy update as Crayon
        delta = 0.01 * math.tanh(mean_return)
        for i in range(len(theta)):
            theta[i] += delta

        total_episodes += len(returns)
        elapsed = time.monotonic() - iter_start
        samples.append({
            "iteration": iteration,
            "episodes": len(returns),
            "mean_return": mean_return,
            "elapsed_ms": int(elapsed * 1000),
        })

        if iteration % 10 == 0 or iteration == args.iterations - 1:
            rate = 1.0 / elapsed if elapsed > 0 else 0
            print(
                f"iter {iteration:4} | episodes {len(returns)} | "
                f"mean_return {mean_return:8.3} | {rate:6.1} it/s | "
                f"total_episodes {total_episodes}"
            )

    total = time.monotonic() - start
    print(
        f"DONE | iterations {args.iterations} | total_episodes {total_episodes} | "
        f"total_time {total:.2f}s | {total_episodes / total:.2f} episodes/s"
    )

    ray.shutdown()

    manifest = {
        "workers": args.workers,
        "parallelism": args.parallelism,
        "iterations": args.iterations,
        "steps": args.steps,
        "policy_dim": args.policy_dim,
        "max_attempts": args.max_attempts,
        "seed_base": args.seed,
        "total_episodes": total_episodes,
        "total_time_s": total,
        "episodes_per_s": total_episodes / total,
    }
    with open(os.path.join(args.artifact_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(args.artifact_dir, "samples.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "episodes", "mean_return", "elapsed_ms"])
        for s in samples:
            w.writerow([s["iteration"], s["episodes"], s["mean_return"], s["elapsed_ms"]])


if __name__ == "__main__":
    main()
