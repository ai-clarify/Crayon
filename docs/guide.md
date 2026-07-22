# Crayon 使用手册

Crayon 是 Ray 核心的 Rust 重实现，为大规模 RL 训练基础设施设计。
本手册覆盖所有核心 API 和典型用法。

## 目录

1. [安装](#1-安装)
2. [Python 绑定](#2-python-绑定)
3. [初始化运行时](#3-初始化运行时)
4. [对象存储](#4-对象存储)
5. [远程任务](#5-远程任务)
6. [Actors](#6-actors)
7. [资源管理](#7-资源管理)
8. [多节点](#8-多节点)
9. [磁盘溢出](#9-磁盘溢出)
10. [系统状态](#10-系统状态)
11. [RL 训练模式](#11-rl-训练模式)

---

## 1. 安装

### Rust

在 `Cargo.toml` 中添加：

```toml
[dependencies]
crayon-rs = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Python

```bash
pip install crayon
```

```python
from crayon import Ray
```

## 2. Python 绑定

Crayon 提供 Python 绑定（PyO3），API 心智与 Ray 一致：

```python
from crayon import Ray, Resources

ray = Ray.init(4)

# 对象存储
r = ray.put(42)
v = ray.get(r)  # 42

# 远程任务
r = ray.spawn(lambda: 42)
v = ray.get(r)  # 42

# 带参数
a = ray.put(10)
r = ray.spawn(lambda x: x + 1, a)
v = ray.get(r)  # 11

# 批量
refs = [ray.put(i) for i in range(100)]
values = ray.get_batch(refs)

# Actor
counter = ray.create_actor("c", {"n": 0})
r = counter.call(lambda c: c.__setitem__("n", c["n"] + 1) or c["n"])

# 资源
r = ray.spawn_with_resources(lambda: 42, Resources(1.0, 0.0))

# 状态
status = ray.status()
print(status.tasks_total, status.tasks_failed)
```

导入：

```rust
use crayon::Ray;
```

## 3. 初始化运行时

```rust
use crayon::Ray;

// 4 个 worker，每个 1 CPU / 0 GPU（默认）
let ray = Ray::init(4);

// 自定义每个 worker 的资源
use crayon::resources::Resources;
let ray = Ray::init_with_resources(8, Resources::new(2.0, 1.0));

// 带内存管理（启用磁盘溢出）
let ray = Ray::init_with_memory(4, 1024 * 1024 * 1024, "/tmp/crayon-spill".into());
```

`Ray` 是 `Arc` 包装的，廉价可克隆。

## 4. 对象存储

Crayon 的对象存储类似 Ray 的 Plasma store。对象以 bincode 序列化存储，
引用计数 GC——最后一个引用 drop 时对象被驱逐。

### 基本 put/get

```rust
let r = ray.put(42i32);
let v: i32 = ray.get(&r).await?;
assert_eq!(v, 42);
```

### 批量操作

```rust
// 批量 put（一次锁获取，高吞吐）
let refs = ray.store().put_batch((0..1000).collect::<Vec<i32>>());

// 批量 get（并发 fan-out，RL 拉取 rollout 的热路径）
let values: Vec<Result<i32, _>> = ray.get_batch(&refs).await;
```

### 引用计数 GC

```rust
let r = ray.put(vec![1u8; 1024]);
let id = r.id;
assert!(ray.store().contains(id));

drop(r);
// 对象被驱逐
tokio::time::sleep(std::time::Duration::from_millis(10)).await;
assert!(!ray.store().contains(id));
```

### 超时

`get` 默认 300s 对象等待上限；网络 RPC 使用独立的端到端 deadline：

```rust
use std::time::Duration;
let result: Result<i32, _> = ray.store().get_bytes_timeout(id, Duration::from_secs(5)).await;
```

## 5. 远程任务

任务是一个在 worker 上执行的闭包。`ObjectRef` 参数会被自动解析
（Ray 的自动依赖解析）。

### 基本 spawn

```rust
let a = ray.put(10);
let b = ray.put(20);
let sum = ray.spawn((a, b), |(a, b): (i32, i32)| a + b);
let v: i32 = ray.get(&sum).await?;
assert_eq!(v, 30);
```

### 依赖链

```rust
let a = ray.put(1);
let b = ray.spawn((a,), |(a,): (i32,)| a + 1);
let c = ray.spawn((b,), |(b,): (i32,)| b * 10);
let v: i32 = ray.get(&c).await?;
assert_eq!(v, 20); // (1+1)*10
```

### 失败重试

任务 panic 或返回 Err 时自动重试，指数退避（100ms * 2^retries，上限 5s）：

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
static COUNT: AtomicUsize = AtomicUsize::new(0);

// 默认重试 3 次
let r = ray.spawn((), |()| {
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    if n < 2 { panic!("intentional"); }
    42
});
let v: i32 = ray.get(&r).await?; // 第三次成功
```

自定义重试次数：

```rust
// 0 = 不重试
let r = ray.spawn_with_retry((), 0, |()| panic!("fail"));
let result: Result<i32, _> = ray.get(&r).await;
assert!(result.is_err()); // 立即失败，不重试
```

### 任务取消

```rust
let r = ray.spawn((), |()| { std::thread::sleep(Duration::from_secs(10)); 42 });
let cancelled = ray.cancel(r.task_id().unwrap());
// 调用方立即看到 TaskCancelled；已运行的闭包不会被强制终止，迟到结果会被丢弃。
```

### 优先级调度

任务按优先级排序（BinaryHeap），高优先级先调度。

## 6. Actors

Actor 是有状态的邮箱，通过 mpsc 串行处理方法调用。同步方法在 blocking
线程池执行，因此不会阻塞 Tokio I/O；同一 Actor 仍严格串行。

### 创建与调用

```rust
#[derive(Default, Clone)]
struct Counter { n: i64 }

let counter = ray.create_actor("c", Counter::default());
let r = counter.call(|c| { c.n += 1; c.n }).await?;
let v: i64 = ray.get(&r).await?;
assert_eq!(v, 1);
```

### 命名查找

```rust
let c1 = ray.create_actor("shared", Counter::default());
let c2 = ray.get_actor::<Counter>("shared").unwrap();
// c1 和 c2 指向同一个 actor，状态共享
```

### 故障重启

```rust
// 最多重启 2 次，每次从初始状态恢复
let actor = ray.create_actor_with_restarts("f", Flaky::default(), 2);
```

### Kill

```rust
counter.kill();
let result = counter.call(|c| c.n).await;
assert!(result.is_err()); // ActorDead
```

### 移除命名 Actor

```rust
assert!(ray.remove_actor("shared"));
assert!(ray.get_actor::<Counter>("shared").is_none());
```

## 7. 资源管理

每个 worker 有 CPU/GPU 资源配额。任务声明所需资源，调度器只把任务
分配给有足够资源的 worker。

### 声明资源

```rust
use crayon::resources::Resources;

// 需要 1 CPU
let r = ray.spawn_with_resources((), Resources::new(1.0, 0.0), |()| 42);

// 需要 0.5 GPU（分数资源）
let r = ray.spawn_with_resources((), Resources::new(0.0, 0.5), |()| 42);
```

### GPU-aware 调度

不需要 GPU 的任务优先分配给无 GPU 的 worker，避免浪费昂贵的 GPU 节点：

```rust
// 集群：3 个 CPU-only worker + 1 个 GPU worker
// CPU 任务自动去 CPU worker，GPU 任务去 GPU worker
```

### 死锁检测

如果任务所需资源超过任何 worker 的总量，立即失败而不是永久排队：

```rust
let ray = Ray::init_with_resources(1, Resources::new(1.0, 0.0));
let r = ray.spawn_with_resources((), Resources::new(2.0, 0.0), |()| 42);
let result: Result<i32, _> = ray.get(&r).await;
assert!(result.is_err()); // 资源不够，立即失败
```

## 8. 多节点

Crayon 通过 TCP 点对点连接实现多节点。对象在本地找不到时自动从
远程 peer 拉取。

### 启动 Head 节点

```bash
cargo run --bin crayon-multi -- --head --addr 0.0.0.0:7000
```

### 启动 Worker 节点

```bash
cargo run --bin crayon-multi -- --worker 127.0.0.1:7000 --addr 0.0.0.0:7001
```

### 编程接口

```rust
use crayon::node::Node;
use crayon::object_store::ObjectStore;

let store = ObjectStore::new();
let head = Node::start("127.0.0.1:0", None, store.clone()).await?;
let head_addr = head.addr.clone();

let store2 = ObjectStore::new();
let worker = Node::start("127.0.0.1:0", Some(&head_addr), store2.clone()).await?;

let store1 = store.with_remote(head.clone());
let store2 = store2.with_remote(worker.clone());

// 在 head 上 put，从 worker 上 get（透明远程拉取）
let r = store1.put(42i32);
let v: i32 = store2.get(r.id).await?;
assert_eq!(v, 42);
```

### 心跳与故障检测

Worker 每 5s 向 head 发送自己的真实 NodeID。head 返回带 epoch 的完整
membership snapshot；RPC 连接、请求和响应共享一个绝对 deadline。对象和 Actor
payload 分块传输，但接收端仍会在发布前完整组装到内存。

## 9. 磁盘溢出

内存超过阈值时，LRU 对象被异步写入磁盘，下次访问时透明加载。

```rust
use crayon::memory::MemoryManager;

let mem = MemoryManager::new(1024 * 1024, "/tmp/crayon-spill".into()); // 1MB
let store = ObjectStore::new().with_memory(mem);

// 放超过 1MB 的对象，LRU 溢出到磁盘
let refs: Vec<_> = (0..100).map(|i| store.put(vec![i as u8; 50_000])).collect();

// 仍然可以读取所有对象（透明从磁盘加载）
for r in &refs {
    let v: Vec<u8> = store.get(r.id).await?;
}
```

## 10. 系统状态

```rust
let status = ray.status();
println!("tasks total: {}", status.tasks_total);
println!("tasks failed: {}", status.tasks_failed);
println!("workers: {}", status.workers.len());
println!("utilization: {}", status.worker_utilization);
for w in &status.workers {
    println!("  worker {}: busy={}, tasks_done={}", w.id, w.busy, w.tasks_done);
}
```

## 11. RL 训练模式

Crayon 为 RL 训练的典型模式设计：

### 参数服务器

```rust
#[derive(Clone, Default)]
struct Params { weights: Vec<f32>, step: u64 }

let ps = ray.create_actor("ps", Params { weights: vec![0.0; 100], step: 0 });

// Worker 更新参数
let r = ps.call(|p| { p.step += 1; p.step }).await?;
```

### Fan-out / Fan-in

```rust
// 并行 spawn N 个 rollout worker
let refs: Vec<_> = (0..64).map(|i| ray.spawn((), move |()| i * i)).collect();

// 并发收集所有结果
let results: Vec<Result<i64, _>> = ray.get_batch(&refs).await;
```

### 完整训练循环

参见 `examples/rl_training.rs` 和 `src/bin/rl_sim.rs`。

```bash
# 模拟 RL 训练（无外部依赖）
cargo run --release --bin crayon-rl -- --workers 64 --steps 50

# 真实 RL 训练（需要 candle）
cargo run --release --example rl_real -- --steps 100
```
