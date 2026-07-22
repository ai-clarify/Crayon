<div align="center">

<img src="logo.svg" width="120" height="120" alt="Crayon">

# Crayon

**Ray 的核心，用 Rust 重写。** 分布式对象存储 · Actor · 任务调度 · 资源管理 · 多节点

为 RL 训练基础设施而生。Rust 核心 + Python 绑定。

</div>

---

## 为什么是 Crayon

| | Ray (Python) | Crayon (Rust + Python) |
|---|---|---|
| 语言 | Python + C++ | Rust 核心 + PyO3 绑定 |
| Python execution | Separate processes | In-process blocking threads; Python still uses the GIL |
| Isolation | Process isolation | No process isolation |
| Core scope | Production distributed platform | Experimental single-process runtime plus object/actor networking |

**同样的 API 心智，零 Python 开销。**

## 快速开始

### Rust

```rust
use crayon::Ray;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ray = Ray::init(4);

    // 对象存储 + 引用计数 GC
    let r = ray.put(42);
    let v: i32 = ray.get(&r).await?;

    // 远程任务 + 依赖自动解析
    let a = ray.put(10);
    let b = ray.put(20);
    let sum = ray.spawn((a, b), |(a, b): (i32, i32)| a + b);

    // 有状态 Actor + 命名查找
    struct Counter(i64);
    let c = ray.create_actor("c", Counter(0));
    c.call(|c| { c.0 += 1; c.0 }).await?;

    Ok(())
}
```

### Python

```python
from crayon import Ray

ray = Ray.init(4)

# 对象存储
r = ray.put(42)
v = ray.get(r)  # 42

# 远程任务
r = ray.spawn(lambda: 42)
v = ray.get(r)  # 42

# 带参数的任务
a = ray.put(10)
r = ray.spawn(lambda x: x + 1, a)
v = ray.get(r)  # 11

# Actor
counter = ray.create_actor("c", {"n": 0})
r = counter.call(lambda c: c.__setitem__("n", c["n"] + 1) or c["n"])
```

## 核心能力

- **Object Store** — Plasma 风格，bincode 序列化，引用计数 GC，LRU 磁盘溢出，批量 put/get
- **Tasks** — 依赖解析、资源声明、失败重试、可观测取消；运行中代码不会被强制终止
- **Actors** — 串行邮箱、命名查找、故障重启；同步方法在线程池执行
- **Resources** — CPU/GPU 记账和分数资源
- **Multi-Node** — 对象与 Actor RPC、权威 membership、端到端 deadline、分块传输
- **Boundaries** — 无进程隔离、无通用跨节点任务调度、无 durable GCS、无 TLS/auth

## 运行

```bash
# Rust
cargo run --bin crayon-demo          # 单节点 demo
cargo run --bin crayon-rl            # RL 训练模拟
cargo test                           # 全部测试 (unit + e2e)

# Python
pip install crayon                   # 安装 Python 包

# 多节点
docker compose up --build            # head + worker
```

## Benchmark

仓库提供可复现的 RL benchmark harness，但不内置跨机器通用的性能结论。
主比较是 `crayon-shared` 与持久化 `ray-actor`；`crayon-serialized` 和
`ray-stateless` 仅用于诊断不同数据路径。

```bash
python benchmark_rl.py --backend crayon-shared --small-model \
  --warmup-steps 1 --measure-steps 2 --repetitions 1 \
  --artifact-dir artifacts/smoke-crayon
```

方法、限制和制品格式见 [docs/benchmark_rl.md](docs/benchmark_rl.md)。

## 文档

- [使用手册](docs/guide.md) — 完整 API 文档和示例
- [架构](docs/architecture.md) — 系统设计

## 解决的 Ray 核心 Issue

| Ray Issue | Crayon 解决方案 |
|-----------|---------------|
| #18916 无超时机制 | `get` 默认 30s 超时，可自定义 |
| #43102 spot 实例 actor 死亡 | `max_restarts` 自动重启 |
| #64470 关键 actor 死亡不 fast-fail | actor panic 标记 dead，新调用立即失败 |
| #53261 RSS 内存泄漏 | refcount 归零时清理 spill 文件 |
| #47866 CPU 任务浪费 GPU 节点 | GPU-aware 调度，CPU 任务优先 CPU 节点 |
| #43624 lineage 大小计算错误 | MemoryManager 用真实对象大小 |
| #62093 plasma 死锁 | 超时 + 死锁检测（资源超 worker 总量时报错）|
| #32952 actor 清理不可靠 | `kill()` 真正关闭 actor task |
| #27499 idle worker 无限 spawn | 固定 worker pool，无动态创建 |

## 架构

```
┌─────────────────────────────────────────────┐
│  Ray (lib.rs)                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │  Store   │ │Scheduler│ │   Actors     │ │
│  │ (Plasma) │ │ (raylet) │ │  (mailbox)   │ │
│  └────┬─────┘ └────┬─────┘ └──────┬───────┘ │
│       │            │              │         │
│  ┌────▼────────────▼──────────────▼───────┐ │
│  │              GCS (metadata)            │ │
│  └────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────┘
                   │ TCP
          ┌────────▼────────┐
          │  Node (peers)   │
          └─────────────────┘
```

---

<div align="center">

*Grab the core, leave the rest.*

</div>
