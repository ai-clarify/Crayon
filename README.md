# Crayon

A Rust reimplementation of [Ray](https://github.com/ray-project/ray)'s core:
distributed object store, task scheduling, actor management, resource
accounting, and multi-node execution. Designed as an RL training infrastructure
backend.

> Ray in Rust. Grab the core, leave the rest.

## Features

| Ray concept          | Crayon equivalent                          | Where            |
|----------------------|--------------------------------------------|------------------|
| Plasma store         | `ObjectStore` — serialized bytes + disk spill | `object_store.rs`|
| `ray.put`/`get`      | `Ray::put` / `Ray::get`                    | `lib.rs`         |
| `@ray.remote` fn     | `Ray::spawn(args, closure)`                | `lib.rs`         |
| `@ray.remote` cls    | `Ray::create_actor` + `ActorHandle::call`  | `actor.rs`       |
| Named actors         | `Ray::get_actor(name)`                     | `lib.rs`         |
| Task dependencies    | `ObjectRef` args auto-resolved             | `args.rs`        |
| GCS                  | `Gcs` — metadata for objects/tasks/actors  | `gcs.rs`         |
| raylet/scheduler     | `Scheduler` + `WorkerPool` (resource-aware)| `scheduler.rs`   |
| Resource tracking    | `Resources` (CPU/GPU) + `ResourceTracker`  | `resources.rs`   |
| Memory/disk spilling | `MemoryManager` — LRU eviction to disk     | `memory.rs`      |
| Multi-node           | `Node` — TCP server, peer discovery        | `node.rs`        |
| `ray status`         | `Ray::status()` → `SystemStatus`           | `status.rs`      |

## Quick start

```rust
use crayon::Ray;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ray = Ray::init(4); // 4 workers, 1 CPU each

    // Object store with reference counting + GC
    let r = ray.put(42);
    let v: i32 = ray.get(&r).await?;

    // Remote task with dependency resolution
    let a = ray.put(10);
    let b = ray.put(20);
    let sum = ray.spawn((a, b), |(a, b): (i32, i32)| a + b);
    let s: i32 = ray.get(&sum).await?;

    // Named actor (stateful, serial execution)
    struct Counter(i64);
    let counter = ray.create_actor("counter", Counter(0));
    let r1 = counter.call(|c| { c.0 += 1; c.0 }).await?;
    let n: i64 = ray.get(&r1).await?;

    // Look up actor by name
    let counter2 = ray.get_actor::<Counter>("counter").unwrap();

    // Resource-aware scheduling
    use crayon::resources::Resources;
    let gpu_task = ray.spawn_with_resources((), Resources::new(0.0, 1.0), |()| {
        // GPU work here
        42
    });

    // Disk spilling (when memory exceeds budget)
    let ray = Ray::init_with_memory(4, 1024 * 1024, "/tmp/crayon_spill".into());

    // Status
    println!("{}", ray.status().pretty());
    Ok(())
}
```

## Multi-node

```rust
use crayon::node::Node;
use crayon::object_store::ObjectStore;

// Head node
let store1 = ObjectStore::new();
let head = Node::start("127.0.0.1:0", None, store1.clone()).await?;

// Worker node connects to head
let store2 = ObjectStore::new();
let worker = Node::start("127.0.0.1:0", Some(&head.addr), store2.clone()).await?;

// Attach remote fetchers
let store1 = store1.with_remote(head.clone());
let store2 = store2.with_remote(worker.clone());

// Put on head, fetch from worker (transparent remote fetch)
let r = store1.put(42i32);
let v: i32 = store2.get(r.id).await?;
```

## RL training pattern

Crayon supports the common RL training pattern:
- **Parameter server** actor holds model weights, updated by trainers
- **Rollout workers** generate experience using current weights
- **Object store** transfers weights and trajectories between nodes
- **Resource scheduling** assigns GPUs to training/inference tasks

See `tests/e2e.rs::actor_as_parameter_server` for a minimal example.

## Run

```bash
cargo run --bin crayon-demo   # single-node demo
cargo test                     # all tests (unit + e2e)
```

## Design notes

- **Objects** stored as bincode-serialized bytes (matching Plasma's design),
  enabling zero-copy cross-node transfer.
- **Reference counting**: each `ObjectRef` holds a refcount; the object is
  evicted when the last ref drops (Plasma-style GC).
- **Disk spilling**: `MemoryManager` tracks per-object sizes and evicts LRU
  objects to disk when memory exceeds `max_memory_bytes`. Evicted objects are
  transparently reloaded on next access.
- **Actors** run as a single tokio task with a mpsc mailbox — state is never
  accessed concurrently (Ray's default model).
- **Scheduler** is resource-aware: tasks declare CPU/GPU requirements; the
  `ResourceTracker` only assigns a task to a worker with sufficient free
  resources. Tasks that don't fit are queued and retried when resources free.
- **Multi-node**: each `Node` runs a TCP server. A head node tracks peers;
  worker nodes register and receive the peer list. `ObjectStore` with a
  `RemoteFetcher` attached transparently fetches objects from remote peers.
