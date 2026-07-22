use crayon::Ray;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_inflight_task_result_does_not_resurrect() {
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

    while !finished.load(Ordering::Acquire) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(30)).await;

    assert!(!ray.store().contains(id));
    assert_eq!(ray.store().len(), 0);
    assert_eq!(ray.store().memory_stats().0, 0);

    drop(ray);
    let _ = std::fs::remove_dir_all(spill);
}
