#!/usr/bin/env python3
"""Multi-actor, multi-role RL end-to-end baseline on Ray.

Three roles as long-lived Ray actors: N rollout actors sample trajectories
against the current policy blob, M judges score each trajectory, one learner
folds trajectories into the next policy. The driver broadcasts the policy via
`ray.put`, chains the DAG through object refs (trajectory bytes never touch the
driver), and reports per-stage latency and episode throughput.

    driver --ray.put(policy)--> plasma
    actors:  rollout(policy_ref, seed)      -> trajectory   (parallelism P)
    judges:  judge(trajectory_ref)          -> verdict      (P)
    learner: learn(policy_ref, *traj_refs)  -> next policy  (1)
    driver:  fetch verdicts + swap policy ref, release old refs, repeat.
"""
import argparse
import json
import statistics
import time

import ray

MASK64 = (1 << 64) - 1


@ray.remote
class RolloutActor:
    def rollout(self, policy, env_seed, steps, traj_bytes):
        rng = env_seed | 1
        ret = 0.0
        for step in range(steps):
            rng = (rng * 6364136223846793005 + 1) & MASK64
            weight = policy[rng % len(policy)] / 255.0
            ret += (weight - 0.5) / (step + 1)
        # Trajectory payload: content is irrelevant to the pipeline, size isn't.
        data = env_seed.to_bytes(8, "little") * (traj_bytes // 8)
        return {"ret": ret, "steps": steps, "data": data}


@ray.remote
class Judge:
    def judge(self, traj):
        data = traj["data"]
        sampled = sum(data[:: max(1, len(data) // 256)])
        return {"score": traj["ret"] + (sampled % 100) * 1e-4, "bytes": len(data)}


@ray.remote
class Learner:
    def learn(self, policy, *trajs):
        ret_sum = sum(t["ret"] for t in trajs)
        nudge = int(max(-127.0, min(127.0, ret_sum * 127.0))) & 0xFF
        updated = bytearray(policy)
        for i in range(min(4096, len(updated))):
            updated[i] = (updated[i] + (nudge ^ (i & 0xFF))) & 0xFF
        return bytes(updated)


def pct(xs, p):
    xs = sorted(xs)
    return xs[round(p / 100.0 * (len(xs) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", type=int, default=4)
    ap.add_argument("--judges", type=int, default=2)
    ap.add_argument("--parallelism", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--policy-mb", type=int, default=4)
    ap.add_argument("--traj-kb", type=int, default=1024)
    ap.add_argument("--out", default="/tmp/ray_rl_e2e.json")
    a = ap.parse_args()

    ray.init(ignore_reinit_error=True)
    actors = [RolloutActor.remote() for _ in range(a.actors)]
    judges = [Judge.remote() for _ in range(a.judges)]
    learner = Learner.remote()

    policy_ref = ray.put(b"\x42" * (a.policy_mb * 1024 * 1024))
    traj_bytes = a.traj_kb * 1024
    rows = []
    total_episodes = 0
    run_start = time.perf_counter()

    for it in range(a.iterations):
        t0 = time.perf_counter()
        traj_refs = [
            actors[i % a.actors].rollout.remote(
                policy_ref, 1 + it * 10_000 + i, a.steps, traj_bytes
            )
            for i in range(a.parallelism)
        ]
        verdict_refs = [
            judges[i % a.judges].judge.remote(traj_refs[i]) for i in range(a.parallelism)
        ]
        new_policy_ref = learner.learn.remote(policy_ref, *traj_refs)

        verdicts = ray.get(verdict_refs)
        judge_ms = (time.perf_counter() - t0) * 1e3
        ray.wait([new_policy_ref], num_returns=1)
        learn_ms = (time.perf_counter() - t0) * 1e3

        del traj_refs, verdict_refs
        policy_ref = new_policy_ref  # idiomatic Ray: swap refs, never pull bytes
        scores = [v["score"] for v in verdicts]
        mean_score = sum(scores) / len(scores)
        total_episodes += len(scores)
        wall_ms = (time.perf_counter() - t0) * 1e3
        rows.append(
            {
                "iteration": it,
                "episodes": len(scores),
                "judge_ms": round(judge_ms, 1),
                "learn_ms": round(learn_ms, 1),
                "wall_ms": round(wall_ms, 1),
                "mean_score": mean_score,
            }
        )
        print(
            f"iter {it:3} | episodes {len(scores):2} | judge {judge_ms:7.1f}ms"
            f" | learn {learn_ms:7.1f}ms | iter {wall_ms:7.1f}ms | mean_score {mean_score:.4f}"
        )

    total_s = time.perf_counter() - run_start
    walls = [r["wall_ms"] for r in rows]
    summary = {
        "actors": a.actors,
        "judges": a.judges,
        "learners": 1,
        "parallelism": a.parallelism,
        "iterations": a.iterations,
        "steps": a.steps,
        "policy_mb": a.policy_mb,
        "traj_kb": a.traj_kb,
        "total_episodes": total_episodes,
        "total_time_s": round(total_s, 3),
        "episodes_per_s": round(total_episodes / total_s, 1),
        "iter_p50_ms": round(pct(walls, 50), 1),
        "iter_p99_ms": round(pct(walls, 99), 1),
        "rows": rows,
    }
    print("DONE", json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
