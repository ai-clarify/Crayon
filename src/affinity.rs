//! CPU core affinity — pin worker threads to specific cores.
//!
//! On multi-socket / multi-NUMA-node machines, cross-node memory access is
//! 2-3x slower than local access. Pinning each worker thread to a dedicated
//! core keeps its cache and NUMA node local, which matters a lot for
//! CPU-bound RL rollout workers.
//!
//! [`build_runtime`] creates a tokio multi-thread runtime where each worker
//! thread is pinned to a distinct core. Use it instead of `#[tokio::main]`
//! for maximum throughput:
//!
//! ```ignore
//! fn main() {
//!     let rt = crayon::affinity::build_runtime(8);
//!     rt.block_on(async {
//!         let ray = crayon::Ray::init(8);
//!         // ...
//!     });
//! }
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};

/// Build a tokio multi-thread runtime with `num_workers` worker threads, each
/// pinned to a distinct physical core.
///
/// If core detection fails (e.g. containers without cpuset), falls back to a
/// regular multi-thread runtime without pinning — no hard failure.
pub fn build_runtime(num_workers: usize) -> tokio::runtime::Runtime {
    let cores = core_affinity::get_core_ids().unwrap_or_default();
    let counter = AtomicUsize::new(0);

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder
        .worker_threads(num_workers)
        .enable_all()
        .on_thread_start(move || {
            if cores.is_empty() {
                return;
            }
            let idx = counter.fetch_add(1, Ordering::Relaxed);
            if let Some(core) = cores.get(idx % cores.len()) {
                core_affinity::set_for_current(*core);
            }
        });

    builder.build().expect("failed to build tokio runtime")
}

/// Number of physical cores available, or 0 if detection fails.
pub fn num_cores() -> usize {
    core_affinity::get_core_ids().map(|v| v.len()).unwrap_or(0)
}
