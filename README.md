<div align="center">

<img src="logo.svg" width="120" height="120" alt="Crayon">

# Crayon

**Ray 的核心，用 Rust 重写。** 分布式对象存储 · Actor · 任务调度 · 资源管理 · 多节点

为 RL 训练基础设施而生。

</div>

---

## 为什么是 Crayon

| | Ray (Python) | Crayon (Rust) |
|---|---|---|
| 语言 | Python + C++ | 纯 Rust |
| GIL | 有，吞吐瓶颈 | 无 |
| 内存安全 | 依赖 GC | 编译期保证 |
| 代码量 | ~1,000,000 行 | ~2,500 行 |
| 核心能力 | 完整 | 完整 |

**同样的 API 心智，零 Python 开销。**

## 快速开始

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

    // 资源感知调度 (CPU/GPU)
    use crayon::resources::Resources;
    ray.spawn_with_resources((), Resources::new(0.0, 1.0), |()| 42);

    Ok(())
}
```

## 核心能力

- **Object Store** — Plasma 风格，bincode 序列化，引用计数 GC，LRU 磁盘溢出
- **Tasks** — 依赖自动解析，资源声明，失败自动重试
- **Actors** — 串行邮箱，命名查找，参数服务器模式
- **Resources** — CPU/GPU 记账，分数资源，调度器只派给能 fit 的 worker
- **Multi-Node** — TCP 点对点，心跳检测，透明远程拉取

## 运行

```bash
cargo run --bin crayon-demo     # 单节点 demo
cargo test                       # 全部测试 (unit + e2e)
docker compose up --build        # 多节点 (head + worker)
```

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
