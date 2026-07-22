use crayon::Ray;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_inflight_task_result_is_reinserted_without_owner() {
    let spill = std::env::temp_dir().join(format!("crayon-audit-orphan-{}", std::process::id()));
    let ray = Ray::init_with_memory(1, 1024 * 1024, spill.clone());
    let finished = Arc::new(AtomicBool::new(false));
    let finished2 = finished.clone();

    let r = ray.spawn((), move |()| {
        std::thread::sleep(Duration::from_millis(100));
        finished2.store(true, Ordering::Release);
        vec![7u8; 4096]
    });
    let id = r.id;
    drop(r);

    assert!(!ray.store().contains(id), "drop removes the reserved entry");
    while !finished.load(Ordering::Acquire) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(30)).await;

    assert!(ray.store().contains(id), "producer resurrected the dropped output");
    assert_eq!(ray.store().len(), 1, "no ObjectRef exists to trigger later removal");
    assert!(ray.store().memory_stats().0 >= 4096);

    drop(ray);
    let _ = std::fs::remove_dir_all(spill);
}
