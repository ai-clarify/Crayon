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
| GIL | 有，吞吐瓶颈 | 无 |
| 内存安全 | 依赖 GC | 编译期保证 |
| 代码量 | ~1,000,000 行 | ~3,000 行 |
| 核心能力 | 完整 | 完整 |
| Python API | ✅ | ✅ (兼容 Ray 心智) |

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
- **Tasks** — 依赖自动解析，资源声明，失败自动重试 + 指数退避，任务取消，优先级调度
- **Actors** — 串行邮箱，命名查找，参数服务器模式，故障自动重启，显式 kill
- **Resources** — CPU/GPU 记账，分数资源，GPU-aware 调度（CPU 任务优先用 CPU 节点）
- **Multi-Node** — TCP 点对点，心跳检测，透明远程拉取，消息大小上限
- **Robustness** — 超时保护，死锁检测，背压（bounded mailbox），spill 文件自动清理

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

## Benchmark: Crayon vs Ray

0.6B Transformer · GRPO · PyTorch · V100 · 2 workers · batch=8

| Mode | Crayon | Ray | 倍数 |
|------|--------|-----|------|
| **A 各自最优** | **1.67s** | 21.2s | **Crayon 12.7x** |
| B 都序列化 | 50.0s | 21.2s | Ray 2.4x |

**Mode A**：Crayon 传模型引用（进程内线程，零拷贝）；Ray 每步 pickle 2.4GB 权重。
**Mode B**：Crayon 也走序列化，隔离框架开销。差距来自 bincode < plasma + 无模型缓存。

详细数据和对抗式审核见 [docs/benchmark_rl.md](docs/benchmark_rl.md)。

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
