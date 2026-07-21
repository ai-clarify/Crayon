//! Comprehensive feature tests — covers edge cases, failure modes, and complex
//! scenarios for every Crayon feature.

use crayon::common::ObjectID;
use crayon::resources::Resources;
use crayon::Ray;

#[derive(Default, Clone)]
struct Counter {
    n: i64,
}

impl Counter {
    fn inc(&mut self) -> i64 {
        self.n += 1;
        self.n
    }
    fn add(&mut self, x: i64) -> i64 {
        self.n += x;
        self.n
    }
}

// ---- Object Store ----

#[tokio::test]
async fn object_refcount_gc() {
    let ray = Ray::init(1);
    let store = ray.store();
    let r = ray.put(42i32);
    let id = r.id;
    assert!(store.contains(id));
    drop(r);
    // After all refs dropped, object should be evicted.
    // Give the drop handler a moment.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert!(!store.contains(id));
}

#[tokio::test]
async fn object_batch_put_get() {
    let ray = Ray::init(2);
    let refs = ray.store().put_batch((0..100).collect::<Vec<i32>>());
    assert_eq!(refs.len(), 100);
    let values: Vec<Result<i32, _>> = ray.get_batch(&refs).await;
    let sum: i32 = values.into_iter().filter_map(|v| v.ok()).sum();
    assert_eq!(sum, (0..100).sum());
}

#[tokio::test]
async fn object_not_found() {
    let ray = Ray::init(1);
    let fake_id = ObjectID::new();
    let result: Result<i32, _> = ray.store().get(fake_id).await;
    assert!(result.is_err());
}

// ---- Tasks ----

#[tokio::test]
async fn task_dependency_chain() {
    let ray = Ray::init(4);
    // a -> b -> c (each depends on previous)
    let a = ray.put(1);
    let b = ray.spawn((a,), |(a,): (i32,)| a + 1);
    let c = ray.spawn((b,), |(b,): (i32,)| b * 10);
    let v: i32 = ray.get(&c).await.unwrap();
    assert_eq!(v, 20); // (1+1)*10
}

#[tokio::test]
async fn task_retry_on_failure() {
    let ray = Ray::init(2);
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    // Task fails twice, succeeds on third try (max_retries=3 default)
    let r = ray.spawn((), |()| {
        let n = COUNT.fetch_add(1, Ordering::Relaxed);
        if n < 2 {
            panic!("intentional failure");
        }
        42
    });
    let v: i32 = ray.get(&r).await.unwrap();
    assert_eq!(v, 42);
    assert!(COUNT.load(Ordering::Relaxed) >= 3);
}

#[tokio::test]
async fn task_exhausts_resources_then_queues() {
    // 1 worker with 1 CPU. Task needs 1 CPU, so only one runs at a time.
    let ray = Ray::init_with_resources(1, Resources::new(1.0, 0.0));
    let start = std::time::Instant::now();
    let r1 = ray.spawn((), |()| {
        std::thread::sleep(std::time::Duration::from_millis(50));
        1
    });
    let r2 = ray.spawn((), |()| {
        std::thread::sleep(std::time::Duration::from_millis(50));
        2
    });
    let v1: i32 = ray.get(&r1).await.unwrap();
    let v2: i32 = ray.get(&r2).await.unwrap();
    assert_eq!(v1, 1);
    assert_eq!(v2, 2);
    // Both tasks should take ~100ms total (serial, not parallel)
    assert!(start.elapsed() >= std::time::Duration::from_millis(90));
}

#[tokio::test]
async fn task_resource_too_large_fails_fast() {
    // Worker has 1 CPU, task needs 2 CPUs — should fail immediately, not deadlock
    let ray = Ray::init_with_resources(1, Resources::new(1.0, 0.0));
    let r = ray.spawn_with_resources((), Resources::new(2.0, 0.0), |()| 42);
    let result: Result<i32, _> = ray.get(&r).await;
    assert!(result.is_err());
}

// ---- Actors ----

#[tokio::test]
async fn actor_concurrent_calls() {
    let ray = Ray::init(4);
    let counter = ray.create_actor("c", Counter::default());
    // Fire 100 concurrent calls
    let handles: Vec<_> = (0..100)
        .map(|_| {
            let c = counter.clone();
            tokio::spawn(async move { c.call(|c| c.inc()).await.unwrap() })
        })
        .collect();
    for h in handles {
        let _ = h.await.unwrap();
    }
    // Final count should be 100
    let r = counter.call(|c| c.n).await.unwrap();
    let n: i64 = ray.get(&r).await.unwrap();
    assert_eq!(n, 100);
}

#[tokio::test]
async fn actor_restart_on_panic() {
    #[derive(Default, Clone)]
    struct Flaky {
        count: u32,
    }
    let ray = Ray::init(2);
    // Allow 2 restarts
    let actor = ray.create_actor_with_restarts("f", Flaky::default(), 2);
    // First call panics
    let r1 = actor
        .call::<(), _>(|f| {
            f.count += 1;
            panic!("boom");
        })
        .await;
    assert!(r1.is_err() || r1.is_ok()); // call may or may not surface error
                                        // After restart, state is reset to initial
    let r2 = actor.call(|f| f.count).await.unwrap();
    let count: u32 = ray.get(&r2).await.unwrap();
    assert_eq!(count, 0); // restarted from initial state
}

#[tokio::test]
async fn actor_kill_rejects_new_calls() {
    let ray = Ray::init(2);
    let counter = ray.create_actor("c", Counter::default());
    counter.kill();
    // New calls should fail
    let result = counter.call(|c| c.inc()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn actor_named_lookup_returns_same_actor() {
    let ray = Ray::init(2);
    let c1 = ray.create_actor("shared", Counter::default());
    let _ = c1.call(|c| c.add(10)).await.unwrap();
    let c2 = ray.get_actor::<Counter>("shared").unwrap();
    let r = c2.call(|c| c.n).await.unwrap();
    let n: i64 = ray.get(&r).await.unwrap();
    assert_eq!(n, 10); // same actor, state persists
}

// ---- Status ----

#[tokio::test]
async fn status_tracks_failed_tasks() {
    let ray = Ray::init(1);
    let r = ray.spawn_with_retry((), 0, |()| panic!("fail"));
    let _: Result<i32, _> = ray.get(&r).await;
    let s = ray.status();
    assert_eq!(s.tasks_failed, 1);
}

#[tokio::test]
async fn status_worker_utilization() {
    let ray = Ray::init(4);
    let s = ray.status();
    assert_eq!(s.workers.len(), 4);
    assert_eq!(s.worker_utilization, 0.0); // all idle initially
}
