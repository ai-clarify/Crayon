#!/usr/bin/env python3
"""Real-LLM RL end-to-end on Crayon, driven entirely from Python.

Three roles on a real cluster: llm-actor workers hold a real model resident
(via benchmarks/llm_sidecar.py) and generate completions for seeded arithmetic
prompts; an llm-judge worker scores them in Rust; the learner (attached to this
driver) does a REINFORCE step and the new weights (~2GB for Qwen3.5-0.8B)
broadcast to every actor through Crayon's shared-memory arena in one `put`.

    driver --put(weights)--> arena
    actors:  llm.rollout(cfg{weights_id, seeds}) -> [{seed, completion}]
    judge:   llm.judge(rollout_output)           -> [reward]
    learner: REINFORCE(prompt, completion, reward) -> next weights
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid

import crayon


def prompt_for(seed):  # keep in sync with llm_sidecar.py / crayon_cluster.rs
    a = 50 + (seed >> 8) % 900
    b = 50 + seed % 900
    return f"What is {a}+{b}? Answer with just the number."


def free_addr():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    addr = f"127.0.0.1:{s.getsockname()[1]}"
    s.close()
    return addr


def connect(addr, deadline_s=15):
    end = time.time() + deadline_s
    while True:
        try:
            return crayon.Client(addr)
        except Exception:
            if time.time() > end:
                raise
            time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", type=int, default=2)
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default="target/release/crayon-cluster")
    ap.add_argument("--sidecar", default="benchmarks/llm_sidecar.py")
    ap.add_argument("--out", default="/tmp/crayon_llm_rl.json")
    ap.add_argument("--workdir", default="/tmp/crayon-llm-rl")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)

    coord = free_addr()
    procs = []

    def spawn(args, name, env=None):
        log = open(os.path.join(a.workdir, f"{name}.log"), "w")
        full_env = dict(os.environ, **(env or {}))
        p = subprocess.Popen(args, stdout=log, stderr=log, env=full_env)
        procs.append(p)
        return p

    spawn([a.binary, "coordinator", coord, "5000"], "coordinator")
    actor_env = {
        "CRAYON_LLM_SIDECAR": os.path.abspath(a.sidecar),
        "CRAYON_LLM_MODEL": a.model,
    }
    for i in range(a.actors):
        spawn(
            [a.binary, "worker", coord, free_addr(), uuid.uuid4().hex, "1.0", "llm-actor"],
            f"actor-{i}",
            actor_env,
        )
    spawn([a.binary, "worker", coord, free_addr(), uuid.uuid4().hex, "1.0", "llm-judge"], "judge")

    learner_log = open(os.path.join(a.workdir, "learner.log"), "w")
    learner = subprocess.Popen(
        [sys.executable, a.sidecar],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=learner_log,
        text=True,
    )
    procs.append(learner)

    def learner_call(msg):
        learner.stdin.write(json.dumps(msg) + "\n")
        learner.stdin.flush()
        reply = json.loads(learner.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    try:
        client = connect(coord)
        while len(json.loads(client.workers())) < a.actors + 1:
            time.sleep(0.2)

        print("initializing learner + saving initial policy ...", flush=True)
        learner_call({"cmd": "init", "model": a.model, "lr": a.lr})
        w_path = os.path.join(a.workdir, "weights.pt")
        learner_call({"cmd": "save", "out": w_path})
        with open(w_path, "rb") as f:
            weights = f.read()
        os.remove(w_path)
        t0 = time.perf_counter()
        weights_id = client.put(weights)
        put_ms = (time.perf_counter() - t0) * 1e3
        print(f"policy {len(weights)/1e9:.2f}GB broadcast in {put_ms:.0f}ms", flush=True)

        version, rows, wait = 0, [], 600_000
        run_start = time.perf_counter()
        for it in range(a.iterations):
            t0 = time.perf_counter()
            specs = []
            for t in range(a.parallelism):
                seeds = [1 + it * 100_000 + t * 1000 + s for s in range(a.batch)]
                cfg = {
                    "version": version,
                    "weights": weights_id,
                    "seeds": seeds,
                    "max_new_tokens": a.max_new_tokens,
                }
                specs.append([json.dumps(cfg).encode()])
            rollout_tasks = client.submit_batch("llm", "rollout", 1, specs)
            judge_tasks = client.submit_batch(
                "llm", "judge", 1, [[out] for _, out in rollout_tasks]
            )

            rewards = [
                r
                for blob in client.get_many([o for _, o in judge_tasks], wait)
                for r in json.loads(blob)
            ]
            rollouts = [
                r
                for blob in client.get_many([o for _, o in rollout_tasks], wait)
                for r in json.loads(blob)
            ]
            rollout_ms = (time.perf_counter() - t0) * 1e3

            items = [
                {
                    "prompt": prompt_for(r["seed"]),
                    "completion": r["completion"],
                    "reward": rw,
                }
                for r, rw in zip(rollouts, rewards)
            ]
            learn = learner_call({"cmd": "learn", "items": items, "out": w_path})
            learn_ms = (time.perf_counter() - t0) * 1e3

            with open(w_path, "rb") as f:
                weights = f.read()
            os.remove(w_path)
            for oid in (
                [o for _, o in rollout_tasks] + [o for _, o in judge_tasks] + [weights_id]
            ):
                client.release(oid)
            weights_id = client.put(weights)
            version += 1
            wall_ms = (time.perf_counter() - t0) * 1e3

            accuracy = sum(rewards) / len(rewards)
            rows.append(
                {
                    "iteration": it,
                    "episodes": len(rewards),
                    "accuracy": accuracy,
                    "loss": learn["loss"],
                    "rollout_ms": round(rollout_ms, 1),
                    "learn_ms": round(learn_ms, 1),
                    "wall_ms": round(wall_ms, 1),
                }
            )
            print(
                f"iter {it:2} | episodes {len(rewards):3} | accuracy {accuracy:.3f}"
                f" | loss {learn['loss']:+.4f} | rollout {rollout_ms:7.0f}ms"
                f" | learn {learn_ms - rollout_ms:6.0f}ms"
                f" | broadcast {wall_ms - learn_ms:5.0f}ms | iter {wall_ms:7.0f}ms",
                flush=True,
            )

        total_s = time.perf_counter() - run_start
        summary = {
            "actors": a.actors,
            "judges": 1,
            "parallelism": a.parallelism,
            "batch": a.batch,
            "iterations": a.iterations,
            "max_new_tokens": a.max_new_tokens,
            "model": a.model,
            "weights_bytes": len(weights),
            "total_time_s": round(total_s, 2),
            "episodes_per_s": round(a.iterations * a.parallelism * a.batch / total_s, 2),
            "rows": rows,
        }
        print("DONE", json.dumps({k: v for k, v in summary.items() if k != "rows"}))
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)
    finally:
        for p in procs:
            p.kill()


if __name__ == "__main__":
    main()
