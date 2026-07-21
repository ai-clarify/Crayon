//! Integration tests for the Crayon runtime.

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
}

#[tokio::test]
async fn put_get_roundtrip() {
    let ray = Ray::init(1);
    let r = ray.put("hello".to_string());
    let v: String = ray.get(&r).await.unwrap();
    assert_eq!(v, "hello");
}

#[tokio::test]
async fn spawn_task() {
    let ray = Ray::init(2);
    let r = ray.spawn((), |()| 6 * 7);
    let v: i32 = ray.get(&r).await.unwrap();
    assert_eq!(v, 42);
}

#[tokio::test]
async fn spawn_with_dependency() {
    let ray = Ray::init(2);
    let a = ray.put(3);
    let b = ray.put(4);
    let r = ray.spawn((a, b), |(a, b): (i32, i32)| a * b);
    let v: i32 = ray.get(&r).await.unwrap();
    assert_eq!(v, 12);
}

#[tokio::test]
async fn fan_out() {
    let ray = Ray::init(4);
    let refs: Vec<_> = (0..10).map(|i| ray.spawn((), move |()| i * i)).collect();
    let mut sum = 0i64;
    for r in &refs {
        sum += ray.get::<i64>(r).await.unwrap();
    }
    assert_eq!(sum, (0..10).map(|i| i * i).sum());
}

#[tokio::test]
async fn actor_stateful() {
    let ray = Ray::init(2);
    let actor = ray.create_actor("c", Counter::default());
    let r1 = actor.call(|c| c.inc()).await.unwrap();
    let r2 = actor.call(|c| c.inc()).await.unwrap();
    let v1: i64 = ray.get(&r1).await.unwrap();
    let v2: i64 = ray.get(&r2).await.unwrap();
    assert_eq!(v1, 1);
    assert_eq!(v2, 2);
}

#[tokio::test]
async fn status_tracks_tasks() {
    let ray = Ray::init(2);
    for _ in 0..5 {
        let r = ray.spawn((), |()| 1);
        let _: i32 = ray.get(&r).await.unwrap();
    }
    let s = ray.status();
    assert_eq!(s.tasks_total, 5);
    assert_eq!(s.tasks_finished, 5);
}

#[tokio::test]
async fn get_waits_for_inflight() {
    let ray = Ray::init(1);
    // Reserve an ID, then get it before put completes.
    let store = ray.store().clone();
    let id = crayon::common::ObjectID::new();
    store.reserve(id);
    let handle = tokio::spawn({
        let store = store.clone();
        async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            store.put_with_id(id, 99i32);
        }
    });
    let v: i32 = store.get(id).await.unwrap();
    assert_eq!(v, 99);
    handle.await.unwrap();
}
