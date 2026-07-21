//! Performance benchmarks for Crayon.
//!
//! Compares pinned (NUMA-affine) runtime vs default runtime, and measures
//! object store throughput + task scheduling overhead.
//!
//! Run:
//!   cargo run --release --example bench

use std::time::Instant;

use crayon::Ray;

/// A CPU-bound rollout simulation: do some fake work to mimic an RL step.
fn rollout_work(n: u64) -> u64 {
    let mut x = 0u64;
    for i in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
    }
    x
}

/// Run N tasks on the given runtime and print throughput.
fn bench_task_throughput(rt: tokio::runtime::Runtime, num_workers: usize, n: u64) {
    let start = Instant::now();
    rt.block_on(async {
        let ray = Ray::init(num_workers);
        let refs: Vec<_> = (0..n)
            .map(|i| ray.spawn((), move |()| rollout_work(i % 1000)))
            .collect();
        let results = ray.get_batch(&refs).await;
        let ok = results.iter().filter(|r| r.is_ok()).count();
        let elapsed = start.elapsed();
        println!(
            "  {n} tasks in {:.2?} -> {:.0} tasks/sec, {ok} ok",
            elapsed,
            n as f64 / elapsed.as_secs_f64()
        );
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Crayon Benchmarks ===\n");

    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!("workers: {num_workers}");
    println!("cores: {}", crayon::affinity::num_cores());

    // ---- Bench 1: task throughput (default runtime) ----
    println!("\n--- Task throughput (default runtime) ---");
    {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_workers)
            .enable_all()
            .build()?;
        bench_task_throughput(rt, num_workers, 10_000);
    }

    // ---- Bench 2: task throughput (pinned runtime) ----
    println!("\n--- Task throughput (pinned/NUMA runtime) ---");
    {
        let rt = crayon::affinity::build_runtime(num_workers);
        bench_task_throughput(rt, num_workers, 10_000);
    }

    // ---- Bench 3: object store put/get throughput ----
    println!("\n--- Object store put/get ---");
    {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_workers)
            .enable_all()
            .build()?;
        rt.block_on(async {
            let ray = Ray::init(num_workers);
            let payload: Vec<u8> = vec![42u8; 1024]; // 1 KiB objects
            let n = 100_000u64;

            let start = Instant::now();
            let refs: Vec<_> = (0..n).map(|_| ray.put(payload.clone())).collect();
            let put_elapsed = start.elapsed();
            println!(
                "  put {n} x 1KiB in {:.2?} -> {:.0} puts/sec",
                put_elapsed,
                n as f64 / put_elapsed.as_secs_f64()
            );

            let start = Instant::now();
            let _ = ray.get_batch(&refs).await;
            let get_elapsed = start.elapsed();
            println!(
                "  get {n} x 1KiB in {:.2?} -> {:.0} gets/sec",
                get_elapsed,
                n as f64 / get_elapsed.as_secs_f64()
            );

            // ---- Bench 4: object size tracking ----
            println!("\n--- Object size tracking ---");
            let r = ray.put(vec![0u8; 4096]);
            let size = ray.store().object_size(r.id);
            println!("  4096-byte object reported size: {size} bytes");
            assert!(size > 0, "size tracking should report non-zero");
        });
    }

    // ---- Bench 5: GPU detection ----
    println!("\n--- GPU detection ---");
    let gpus = crayon::device::detect_gpu_count();
    println!("  detected {gpus} GPU(s)");
    println!("  current_device from main thread: {:?}", crayon::device::current_device());

    println!("\n=== All benchmarks done ===");
    Ok(())
}
