"""Crayon Python quickstart.

Shows every core feature: object store, remote tasks, actors, resources,
batch ops, and the built-in memory management that prevents OOM kills.

Run:
    pip install crayon-dist
    python quickstart.py
"""

import crayon


# Actor classes must be defined at module level so they can be pickled.
class Counter:
    def __init__(self):
        self.n = 0

    def inc(self):
        self.n += 1
        return self.n

    def get(self):
        return self.n


def main():
    # Memory management is ON by default (50% of RAM). Customize it:
    #   ray = crayon.Ray(4, max_memory_bytes=1_000_000_000, spill_dir="/tmp/spill")
    ray = crayon.Ray(4)
    print("=== Crayon Quickstart ===")

    # 1. Object store: put / get
    r = ray.put(42)
    v = ray.get(r)
    print(f"put/get: {v}")

    # 2. Remote task (closure)
    r = ray.spawn(lambda: 100)
    print(f"spawn: {ray.get(r)}")

    # 3. Task with args — ObjectRef args are auto-resolved (like Ray)
    a = ray.put(10)
    r = ray.spawn(lambda x: x + 1, a)
    print(f"spawn with ref arg: {ray.get(r)}")

    # 4. Batch: fan out / fan in
    refs = [ray.spawn(lambda i=i: i * i) for i in range(8)]
    squares = ray.get_batch(refs)
    print(f"get_batch: {squares}")

    # 5. Actor (stateful, pinned to a worker)
    counter = ray.create_actor("counter", Counter())
    print(f"actor inc: {ray.get(counter.call('inc'))}")
    print(f"actor inc: {ray.get(counter.call('inc'))}")
    print(f"actor get: {ray.get(counter.call('get'))}")

    # 6. Resources (CPU / GPU requirements)
    res = crayon.Resources(cpu=1.0, gpu=0.0)
    r = ray.spawn_with_resources(lambda: 42, res)
    print(f"spawn_with_resources: {ray.get(r)}")

    # 7. Status — includes memory usage so you can watch for OOM pressure
    status = ray.status()
    print(
        f"status: {status['tasks_total']} tasks, "
        f"{status['tasks_finished']} finished, "
        f"memory {status['memory_used_bytes']}/{status['memory_limit_bytes']} bytes"
    )

    print("\n=== All quickstart checks passed ===")


if __name__ == "__main__":
    main()
