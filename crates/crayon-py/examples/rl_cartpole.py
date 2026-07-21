"""Distributed RL (CartPole + REINFORCE) with Crayon in pure Python.

Mirrors the Rust `examples/rl_cartpole.rs` but uses numpy for the policy net
so it runs with zero extra dependencies beyond `crayon-dist` and `numpy`.

Pipeline:
  - Parameter server (Crayon actor): holds global policy weights + step count
  - Rollout workers (Crayon tasks): collect trajectories using current policy
  - Trainer: aggregates rollouts, computes REINFORCE gradient, updates PS

Run:
    pip install crayon-dist numpy
    python rl_cartpole.py --steps 100 --workers 8
"""

import argparse
import time

import numpy as np

import crayon


# ---- CartPole environment (pure Python, no gym dependency) ----

class CartPole:
    def __init__(self):
        rng = np.random.default_rng()
        self.x = rng.uniform(-0.05, 0.05)
        self.x_dot = rng.uniform(-0.05, 0.05)
        self.theta = rng.uniform(-0.05, 0.05)
        self.theta_dot = rng.uniform(-0.05, 0.05)
        self.steps = 0
        self.max_steps = 500

    def step(self, action):
        force = -10.0 if action == 0 else 10.0
        mass_cart, mass_pole = 1.0, 0.1
        total_mass = mass_cart + mass_pole
        length = 0.5
        polemass_length = mass_pole * length
        gravity = 9.8

        temp = (force + polemass_length * self.theta ** 2 * np.sin(self.theta_dot)) / total_mass
        theta_acc = (
            gravity * np.sin(self.theta) - temp * np.cos(self.theta)
        ) / (length * (4.0 / 3.0 - mass_pole * np.cos(self.theta) ** 2 / total_mass))
        x_acc = temp - polemass_length * theta_acc * np.cos(self.theta) / total_mass

        dt = 0.02
        self.x += self.x_dot * dt
        self.x_dot += x_acc * dt
        self.theta += self.theta_dot * dt
        self.theta_dot += theta_acc * dt
        self.steps += 1

        obs = np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=np.float32)
        done = (
            abs(self.theta) > 12.0 * np.pi / 180.0
            or abs(self.x) > 2.4
            or self.steps >= self.max_steps
        )
        reward = 0.0 if done else 1.0
        return obs, reward, done


# ---- Policy network (2-layer MLP in numpy) ----

def policy_forward(obs, w1, b1, w2, b2):
    """obs: [4] -> logits: [2]"""
    h = np.maximum(0, obs @ w1.T + b1)  # [64]
    return h @ w2.T + b2  # [2]


def act(obs, params, rng):
    """Sample action from policy. Returns (action, log_prob)."""
    w1, b1, w2, b2 = params
    logits = policy_forward(obs, w1, b1, w2, b2)
    # softmax
    logits -= logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    action = rng.choice(2, p=probs)
    log_prob = np.log(probs[action] + 1e-8)
    return int(action), float(log_prob)


def init_params(rng):
    w1 = rng.normal(0, 0.1, (64, 4)).astype(np.float32)
    b1 = np.zeros(64, dtype=np.float32)
    w2 = rng.normal(0, 0.1, (2, 64)).astype(np.float32)
    b2 = np.zeros(2, dtype=np.float32)
    return [w1, b1, w2, b2]


def params_flat(params):
    return np.concatenate([p.ravel() for p in params])


def params_from_flat(flat):
    off = 0
    w1 = flat[off:off + 64 * 4].reshape(64, 4); off += 64 * 4
    b1 = flat[off:off + 64]; off += 64
    w2 = flat[off:off + 2 * 64].reshape(2, 64); off += 2 * 64
    b2 = flat[off:off + 2]
    return [w1, b1, w2, b2]


# ---- Rollout: collect a trajectory ----

def rollout(params_flat_bytes, max_steps):
    """Runs inside a Crayon task. params is pickled numpy array."""
    import pickle
    flat = pickle.loads(params_flat_bytes)
    params = params_from_flat(flat)
    rng = np.random.default_rng()

    env = CartPole()
    obs_list, action_list, reward_list = [], [], []
    total_reward = 0.0

    for _ in range(max_steps):
        obs = np.array([env.x, env.x_dot, env.theta, env.theta_dot], dtype=np.float32)
        action, _ = act(obs, params, rng)
        obs_list.append(obs)
        action_list.append(action)
        _, reward, done = env.step(action)
        reward_list.append(reward)
        total_reward += reward
        if done:
            break

    return obs_list, action_list, reward_list, total_reward


# ---- Parameter server state ----

class PS:
    def __init__(self, params_flat):
        self.params = params_flat
        self.step = 0

    def get_params(self):
        return self.params

    def set_params(self, params_flat):
        self.params = params_flat
        self.step += 1
        return self.step


# ---- Trainer ----

def compute_returns(rewards, gamma):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print("=== Crayon Python RL (CartPole + REINFORCE) ===")
    print(f"steps={args.steps} workers={args.workers}")

    ray = crayon.Ray(4)
    rng = np.random.default_rng(42)

    init_p = init_params(rng)
    flat = params_flat(init_p)
    print(f"policy params: {flat.size} floats")

    ps = ray.create_actor("ps", PS(flat))

    # Versioned artifact store: keep last 5 checkpoints + metrics.
    # Old versions are evicted automatically — bounds memory on long runs.
    vs = crayon.VersionedStore(ray, keep_last=5)

    gamma = 0.99
    lr = 0.01
    start = time.time()
    total_episodes = 0

    for step in range(args.steps):
        # 1. Pull current params from PS
        params_bytes = ray.get(ps.call("get_params"))
        params = params_from_flat(params_bytes)

        # 2. Fan out: rollouts in parallel
        import pickle
        pb = pickle.dumps(params_bytes)
        refs = [ray.spawn(rollout, pb, 500) for _ in range(args.workers)]
        results = [r for r in ray.get_batch(refs) if not isinstance(r, Exception)]

        avg_reward = np.mean([r[3] for r in results]) if results else 0.0
        total_episodes += len(results)

        # 3. Compute policy gradient (REINFORCE)
        grad = np.zeros_like(params_bytes)
        count = 0
        eps = 1e-3

        for obs_list, action_list, reward_list, _ in results:
            returns = compute_returns(reward_list, gamma)
            adv = (returns - returns.mean()) / (returns.std() + 1e-8)

            for pi in range(len(grad)):
                # finite differences on log_prob * advantage
                p_up = params_bytes.copy(); p_up[pi] += eps
                p_down = params_bytes.copy(); p_down[pi] -= eps
                params_up = params_from_flat(p_up)
                params_down = params_from_flat(p_down)

                lp_up = sum(
                    act(o, params_up, rng)[1] for o, a in zip(obs_list, action_list) if a == 0
                )
                lp_down = sum(
                    act(o, params_down, rng)[1] for o, a in zip(obs_list, action_list) if a == 0
                )
                d_log_prob = (lp_up - lp_down) / (2.0 * eps)
                grad[pi] += d_log_prob * adv.sum()
                count += 1

        if count > 0:
            grad /= count

        # 4. SGD update
        new_params = params_bytes + lr * grad
        ray.get(ps.call("set_params", new_params))

        # 5. Save versioned checkpoint + metrics every 10 steps
        if step % 10 == 0 or step == args.steps - 1:
            vs.put("policy", new_params, version=step)
            vs.put("metrics", {"step": step, "avg_reward": float(avg_reward),
                               "episodes": int(total_episodes)}, version=step)

        if step % 10 == 0 or step == args.steps - 1:
            print(
                f"step {step:>4}/{args.steps} | avg_reward={avg_reward:>6.1f} "
                f"| episodes={total_episodes:>5} | {time.time() - start:.1f}s"
            )

    print(f"\n=== Done: {args.steps} steps, {total_episodes} episodes ===")
    print(f"checkpoint history: {vs.history('policy')}")
    print(f"latest metrics: {vs.get('metrics')}")
    print(ray.status())


if __name__ == "__main__":
    main()
