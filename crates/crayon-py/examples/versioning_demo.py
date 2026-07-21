"""Versioned artifact storage for RL training.

Shows how Crayon's VersionedStore keeps a bounded history of checkpoints,
rollout batches, and metrics — so you can roll back, reproduce, and compare
across training steps without memory growing unboundedly.

Run:
    pip install crayon-dist
    python versioning_demo.py
"""

import crayon


def main():
    ray = crayon.Ray(2)
    # keep_last=3: only the 3 most recent versions of each name are retained.
    # Older versions are evicted automatically — part of OOM prevention.
    vs = crayon.VersionedStore(ray, keep_last=3)

    print("=== Versioned Store Demo ===")

    # Simulate saving policy checkpoints at each training step
    for step in range(5):
        params = [float(step), float(step * 2), float(step * 3)]
        vs.put("policy", params, version=step)
        vs.put("metrics", {"step": step, "reward": step * 10.0}, version=step)

    # Only last 3 versions survive (steps 2, 3, 4)
    print("policy history:", vs.history("policy"))
    print("metrics history:", vs.history("metrics"))

    # Get the latest version
    latest_policy = vs.get("policy")
    print("latest policy:", latest_policy)

    # Get a specific version
    v3_policy = vs.get_at("policy", 3)
    print("policy @ step 3:", v3_policy)

    # Evicted versions raise an error
    try:
        vs.get_at("policy", 0)
    except RuntimeError as e:
        print("step 0 evicted (expected):", e)

    # Latest version number
    print("latest policy version:", vs.latest_version("policy"))

    # Remove all versions of a name
    vs.remove("metrics")
    print("metrics after remove:", vs.history("metrics"))

    print("\n=== All versioning checks passed ===")


if __name__ == "__main__":
    main()
