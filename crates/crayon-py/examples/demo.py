"""Crayon Python API demo.

Shows the core features: object store, remote tasks, actors, and resources.
"""

import crayon


class Counter:
    def __init__(self):
        self.n = 0

    def inc(self):
        self.n += 1
        return self.n

    def get(self):
        return self.n


def main():
    ray = crayon.Ray(4)
    print("=== Crayon Python Demo ===")

    # Object store
    r = ray.put(42)
    v = ray.get(r)
    print(f"put/get: {v}")

    # Remote task
    r = ray.spawn(lambda: 100)
    v = ray.get(r)
    print(f"spawn: {v}")

    # Task with captured value
    x = 7
    r = ray.spawn(lambda: x * 6)
    v = ray.get(r)
    print(f"spawn closure: {v}")

    # Batch operations
    refs = [ray.put(i) for i in range(10)]
    values = ray.get_batch(refs)
    print(f"get_batch: {values}")

    # Actor
    counter = ray.create_actor("counter", Counter())
    r = counter.call("inc")
    print(f"actor inc: {ray.get(r)}")
    r = counter.call("inc")
    print(f"actor inc: {ray.get(r)}")
    r = counter.call("get")
    print(f"actor get: {ray.get(r)}")

    # Resources
    res = crayon.Resources(1.0, 0.0)
    r = ray.spawn_with_resources(lambda: 42, res)
    print(f"spawn_with_resources: {ray.get(r)}")

    # Status
    status = ray.status()
    print(f"status: {status['tasks_total']} tasks, {status['tasks_finished']} finished")

    print("\n=== All demos passed ===")


if __name__ == "__main__":
    main()
