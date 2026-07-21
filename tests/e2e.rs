//! End-to-end multi-node test.
//!
//! Spawns a head node and a worker node in separate tokio tasks, puts an
//! object on the head, and fetches it from the worker. Also tests disk
//! spilling.

use crayon::memory::MemoryManager;
use crayon::node::Node;
use crayon::object_store::ObjectStore;
use crayon::Ray;
use std::sync::Arc;

#[tokio::test]
async fn multi_node_object_transfer() {
    let store1 = ObjectStore::new();
    let store2 = ObjectStore::new();

    // Start head node (no head_addr)
    let head = Node::start("127.0.0.1:0", None, store1.clone())
        .await
        .unwrap();
    let head_addr = head.addr.clone();

    // Start worker node, connect to head
    let worker = Node::start("127.0.0.1:0", Some(&head_addr), store2.clone())
        .await
        .unwrap();

    // Give registration a moment
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Attach nodes to stores for remote fetching
    let store1 = store1.with_remote(head.clone());
    let store2 = store2.with_remote(worker.clone());

    // Put an object on the head node (keep the ref alive so it isn't GC'd)
    let r = store1.put(42i32);

    // Fetch it from the worker node (remote fetch)
    let v: i32 = store2.get(r.id).await.unwrap();
    assert_eq!(v, 42);
}

#[tokio::test]
async fn disk_spilling() {
    let spill_dir = std::env::temp_dir().join("crayon_spill_test");
    let _ = std::fs::remove_dir_all(&spill_dir);
    let mem = MemoryManager::new(100, spill_dir.clone()); // 100 bytes max
    let store = ObjectStore::new().with_memory(mem.clone());

    // Put several objects that exceed memory budget (keep refs alive)
    let mut refs = Vec::new();
    for i in 0..5 {
        let data = vec![i as u8; 50]; // 50 bytes each
        let r = store.put(data);
        refs.push(r);
    }

    // Some objects should have been spilled (memory should be near budget)
    let mem_used = mem.total_bytes();
    assert!(
        mem_used <= 100,
        "expected memory to be at or below budget, got {mem_used}"
    );

    // But we can still retrieve all of them (transparent reload)
    for (i, r) in refs.iter().enumerate() {
        let v: Vec<u8> = store.get(r.id).await.unwrap();
        assert_eq!(v, vec![i as u8; 50]);
    }

    let _ = std::fs::remove_dir_all(&spill_dir);
}

#[tokio::test]
async fn distributed_task_via_ray() {
    let ray = Ray::init(2);

    // Simulate a simple distributed workload: map-reduce style sum of squares
    let refs: Vec<_> = (0..10).map(|i| ray.spawn((), move |()| i * i)).collect();

    let mut total = 0i64;
    for r in &refs {
        total += ray.get::<i64>(r).await.unwrap();
    }
    assert_eq!(total, (0..10).map(|i| i * i).sum::<i64>());
}

#[tokio::test]
async fn actor_as_parameter_server() {
    // Simulates a common RL pattern: an actor holds shared parameters
    #[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
    struct Params {
        weights: Vec<f32>,
        step: u64,
    }

    impl Params {
        fn update(&mut self, grad: Vec<f32>) {
            for (w, g) in self.weights.iter_mut().zip(grad.iter()) {
                *w += g;
            }
            self.step += 1;
        }
        fn get(&self) -> (Vec<f32>, u64) {
            (self.weights.clone(), self.step)
        }
    }

    let ray = Ray::init(2);
    let ps = ray.create_actor("ps", Params { weights: vec![0.0; 4], step: 0 });

    // Worker updates parameters
    let r = ps.call(|p| { p.update(vec![0.1; 4]); p.get() }).await.unwrap();
    let (w, step): (Vec<f32>, u64) = ray.get(&r).await.unwrap();
    assert_eq!(step, 1);
    assert!((w[0] - 0.1).abs() < 1e-6);
}
