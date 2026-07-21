import crayon

ray = crayon.Ray(4)

# Test put/get
print("Test 1: put/get")
r = ray.put(42)
v = ray.get(r)
assert v == 42, f"expected 42, got {v}"
print(f"  put(42) -> get -> {v} OK")

# Test spawn
print("Test 2: spawn")
r = ray.spawn(lambda: 42)
v = ray.get(r)
assert v == 42, f"expected 42, got {v}"
print(f"  spawn(lambda: 42) -> get -> {v} OK")

# Test spawn with args
print("Test 3: spawn with args")
a = ray.put(10)
r = ray.spawn(lambda x: x + 1, a)
v = ray.get(r)
assert v == 11, f"expected 11, got {v}"
print(f"  spawn(lambda x: x+1, put(10)) -> get -> {v} OK")

# Test create_actor
print("Test 4: create_actor")
class Counter:
    def __init__(self):
        self.n = 0
    def increment(self):
        self.n += 1
        return self.n
    def get(self):
        return self.n

actor = ray.create_actor("counter", Counter())
r = actor.call("increment")
assert ray.get(r) == 1, f"expected 1"
r = actor.call("increment")
assert ray.get(r) == 2, f"expected 2"
r = actor.call("get")
assert ray.get(r) == 2, f"expected 2"
print(f"  counter actor state OK")

# Test status
print("Test 5: status")
status = ray.status()
print(f"  status: {status}")

print("\nAll tests passed!")
