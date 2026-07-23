//! Storage-path benchmark: measures the object store's put/get latency,
//! throughput, and read (memcpy) amplification. Two independent measurements:
//!
//!   * e2e   — a real coordinator process; client does put(payload)+get(id) over
//!             TCP, the same boundary `ray.put`/`ray.get` cross. Reports p50/p99
//!             latency and MB/s per payload size.
//!   * amp   — in-process `LocalObjectStore`: store one object, GET it N times
//!             under a counting allocator. Bytes allocated / logical bytes read
//!             is the read amplification. Content-addressed immutability makes
//!             this ~0 once payloads are `Arc<[u8]>` (a GET is a refcount bump).
//!
//! Run both, emit one JSON summary. Compare across git revisions for before/after.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant},
};

use crayon::{
    client::ClusterClient,
    data_plane::LocalObjectStore,
    error::Error,
    ids::ObjectId,
    operation::Codec,
};
use tokio::net::TcpListener;

/// Allocator that tallies bytes handed out, but only while `COUNTING` is set, so
/// the amplification window excludes setup/teardown noise.
struct Counting;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    std::fs::create_dir_all(&args.artifact_dir)?;

    let mut report = String::from("{\n");
    report.push_str(&format!(
        "  \"samples\": {}, \"warmups\": {},\n",
        args.samples, args.warmups
    ));

    // --- e2e latency / throughput against a real coordinator process ---
    let binary = std::env::current_exe()?
        .parent()
        .unwrap()
        .join("crayon-cluster");
    let coordinator_addr = free_addr().await?;
    let coordinator = spawn(
        &binary,
        &["coordinator", &coordinator_addr, "5000"],
        &args.artifact_dir,
    )?;
    let mut client = ClusterClient::connect(&coordinator_addr);
    wait_coordinator(&mut client).await?;
    client.connect_epoch().await?;

    report.push_str("  \"e2e\": [\n");
    for (i, &size) in args.sizes.iter().enumerate() {
        let row = run_e2e(&client, size, &args).await?;
        report.push_str(&format!("    {row}"));
        report.push_str(if i + 1 < args.sizes.len() { ",\n" } else { "\n" });
    }
    report.push_str("  ],\n");
    kill(coordinator);

    // --- in-process read amplification ---
    report.push_str("  \"read_amplification\": [\n");
    for (i, &size) in args.sizes.iter().enumerate() {
        let row = measure_read_amp(size, args.samples);
        report.push_str(&format!("    {row}"));
        report.push_str(if i + 1 < args.sizes.len() { ",\n" } else { "\n" });
    }
    report.push_str("  ]\n}\n");

    let out = args.artifact_dir.join("storage_summary.json");
    std::fs::write(&out, &report)?;
    print!("{report}");
    eprintln!("wrote {}", out.display());
    Ok(())
}

/// One payload size, end to end: warm up, then time `samples` put/get pairs
/// separately. Latency is per-op; throughput is payload bytes over op time.
async fn run_e2e(client: &ClusterClient, size: usize, args: &Args) -> Result<String, Error> {
    let mut payload = vec![0xA5u8; size];
    // Raw-bytes path, the same shape as `ray.put(bytes)`: no serialization
    // envelope, the round trip returns exactly `size` bytes.
    let expected = size;
    let mut put_us = Vec::with_capacity(args.samples);
    let mut get_us = Vec::with_capacity(args.samples);

    for iter in 0..(args.warmups + args.samples) {
        // Stamp a unique prefix so each put is a fresh insert, not a
        // content-addressed dedup hit — matching `ray.put`'s per-call object.
        payload[..8.min(size)].copy_from_slice(&(iter as u64).to_le_bytes()[..8.min(size)]);

        let start = Instant::now();
        let id: ObjectId = client.put_bytes(&payload).await?;
        let put_elapsed = start.elapsed();

        let start = Instant::now();
        let (_codec, got) = client.get_bytes(id).await?;
        let get_elapsed = start.elapsed();
        if got.len() != expected {
            return Err(Error::Protocol("payload size mismatch".into()));
        }
        // Free it so memory stays bounded, like Ray dropping the ref after get.
        client.release(id).await?;
        if iter >= args.warmups {
            put_us.push(put_elapsed.as_secs_f64() * 1e6);
            get_us.push(get_elapsed.as_secs_f64() * 1e6);
        }
    }
    Ok(format!(
        "{{\"size_bytes\": {size}, \
         \"put_p50_us\": {:.2}, \"put_p99_us\": {:.2}, \"put_mb_s\": {:.1}, \
         \"get_p50_us\": {:.2}, \"get_p99_us\": {:.2}, \"get_mb_s\": {:.1}}}",
        pct(&mut put_us, 50.0),
        pct(&mut put_us, 99.0),
        mb_per_s(size, mean(&put_us)),
        pct(&mut get_us, 50.0),
        pct(&mut get_us, 99.0),
        mb_per_s(size, mean(&get_us)),
    ))
}

/// Read amplification: store one object, then GET it `reads` times with the
/// allocator counting. Amplification = bytes allocated / (reads * size).
fn measure_read_amp(size: usize, reads: usize) -> String {
    let store = LocalObjectStore::default();
    let id = ObjectId::new();
    store
        .put(id, Codec::RawBytes, vec![0xA5u8; size])
        .expect("put");

    ALLOCATED.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let mut sink = 0usize;
    for _ in 0..reads {
        let obj = store.get(id).expect("get");
        // Touch the bytes so the read is not optimized away.
        sink = sink.wrapping_add(obj.bytes.len());
    }
    COUNTING.store(false, Ordering::Relaxed);
    std::hint::black_box(sink);

    let allocated = ALLOCATED.load(Ordering::Relaxed);
    let logical = (reads * size) as f64;
    format!(
        "{{\"size_bytes\": {size}, \"reads\": {reads}, \
         \"bytes_allocated\": {allocated}, \"amplification\": {:.4}}}",
        allocated as f64 / logical
    )
}

// --- helpers ---

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}
fn pct(xs: &mut [f64], p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (p / 100.0 * (xs.len() - 1) as f64).round() as usize;
    xs[rank]
}
fn mb_per_s(size: usize, us: f64) -> f64 {
    if us <= 0.0 {
        return 0.0;
    }
    (size as f64 / 1_048_576.0) / (us / 1e6)
}

struct Args {
    samples: usize,
    warmups: usize,
    sizes: Vec<usize>,
    artifact_dir: PathBuf,
}
fn parse_args() -> Args {
    let mut m = std::collections::HashMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        if let Some(v) = it.next() {
            m.insert(k, v);
        }
    }
    let sizes = m
        .get("--sizes")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1024, 16384, 262_144, 1_048_576, 4_194_304]);
    Args {
        samples: m.get("--samples").and_then(|v| v.parse().ok()).unwrap_or(2000),
        warmups: m.get("--warmups").and_then(|v| v.parse().ok()).unwrap_or(200),
        sizes,
        artifact_dir: m
            .get("--artifact-dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("benchmark_artifacts/storage")),
    }
}
async fn free_addr() -> Result<String, Error> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    Ok(listener.local_addr()?.to_string())
}
fn spawn(binary: &Path, args: &[&str], dir: &Path) -> Result<Child, Error> {
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("process.log"))?;
    Ok(Command::new(binary)
        .args(args)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .spawn()?)
}
fn kill(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}
async fn wait_coordinator(client: &mut ClusterClient) -> Result<(), Error> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if client.workers().await.is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(Error::Protocol("coordinator did not start in time".into()));
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}
