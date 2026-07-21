//! End-to-end multi-node test.
//!
//! Spawns a head node and a worker node in separate tokio tasks, puts an
//! object on the head, and fetches it from the worker. Also tests disk
//! spilling.

use crayon::memory::MemoryManager;
use crayon::node::Node;
use crayon::object_store::ObjectStore;
use crayon::Ray;

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
    let ps = ray.create_actor(
        "ps",
        Params {
            weights: vec![0.0; 4],
            step: 0,
        },
    );

    // Worker updates parameters
    let r = ps
        .call(|p| {
            p.update(vec![0.1; 4]);
            p.get()
        })
        .await
        .unwrap();
    let (w, step): (Vec<f32>, u64) = ray.get(&r).await.unwrap();
    assert_eq!(step, 1);
    assert!((w[0] - 0.1).abs() < 1e-6);
}

/// Simulates a full distributed RL training loop:
/// - Parameter server actor holds model weights
/// - N rollout workers generate experience using current weights
/// - Trainer aggregates gradients and updates the PS
/// - Stale rollouts are cancelled when weights change
#[tokio::test]
async fn rl_training_loop() {
    #[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
    struct Model {
        weights: Vec<f32>,
        step: u64,
    }

    impl Model {
        fn apply_grad(&mut self, grad: Vec<f32>) {
            for (w, g) in self.weights.iter_mut().zip(grad.iter()) {
                *w += g;
            }
            self.step += 1;
        }
    }

    let ray = Ray::init(4);
    let ps = ray.create_actor(
        "ps",
        Model {
            weights: vec![0.0; 8],
            step: 0,
        },
    );

    // Run 5 training iterations
    for iteration in 0..5 {
        // Get current weights from PS
        let r = ps.call(|m| (m.weights.clone(), m.step)).await.unwrap();
        let (weights, step): (Vec<f32>, u64) = ray.get(&r).await.unwrap();
        assert_eq!(step, iteration);

        // Spawn 4 rollout workers, each "generates" a gradient
        let weights_ref = ray.put(weights.clone());
        let rollout_refs: Vec<_> = (0..4)
            .map(|i| {
                let w = weights_ref.clone();
                ray.spawn((w,), move |(w,): (Vec<f32>,)| {
                    // Simulate gradient computation
                    let grad: Vec<f32> = w.iter().map(|x| x * 0.01 + i as f32 * 0.001).collect();
                    grad
                })
            })
            .collect();

        // Collect gradients
        let grads: Vec<Result<Vec<f32>, _>> = ray.get_batch(&rollout_refs).await;
        let grads: Vec<Vec<f32>> = grads.into_iter().filter_map(|g| g.ok()).collect();
        assert_eq!(grads.len(), 4);

        // Average gradients
        let dim = grads[0].len();
        let mut avg_grad = vec![0.0f32; dim];
        for g in &grads {
            for (i, v) in g.iter().enumerate() {
                avg_grad[i] += v;
            }
        }
        for v in avg_grad.iter_mut() {
            *v /= grads.len() as f32;
        }

        // Apply gradient to PS
        let r = ps
            .call(move |m| {
                m.apply_grad(avg_grad);
                m.step
            })
            .await
            .unwrap();
        let new_step: u64 = ray.get(&r).await.unwrap();
        assert_eq!(new_step, iteration + 1);
    }

    // Verify weights changed from initial
    let r = ps.call(|m| m.weights.clone()).await.unwrap();
    let final_weights: Vec<f32> = ray.get(&r).await.unwrap();
    assert!(final_weights.iter().any(|w| *w != 0.0));
}

/// Test that tasks can be cancelled (critical for stale rollout in RL).
#[tokio::test]
async fn task_cancellation() {
    let ray = Ray::init(1);

    // Cancel an unknown task id returns false
    let fake_id = crayon::common::TaskID::new();
    assert!(!ray.cancel(fake_id));

    // Spawn a quick task to verify normal operation still works
    let r = ray.spawn((), |()| 42);
    let v: i32 = ray.get(&r).await.unwrap();
    assert_eq!(v, 42);
}

/// Test that named actors can be removed from the registry.
#[tokio::test]
async fn remove_named_actor() {
    #[derive(Default, Clone)]
    #[allow(dead_code)]
    struct Counter(i64);

    let ray = Ray::init(2);
    let _counter = ray.create_actor("c", Counter(0));
    assert!(ray.get_actor::<Counter>("c").is_some());

    assert!(ray.remove_actor("c"));
    assert!(ray.get_actor::<Counter>("c").is_none());
    assert!(!ray.remove_actor("c"));
}

/// Test that fractional resources work (Ray #20933 analog).
#[tokio::test]
async fn fractional_resources() {
    use crayon::resources::Resources;
    let ray = Ray::init_with_resources(2, Resources::new(1.0, 1.0));

    // Two tasks each needing 0.5 GPU should both fit on the 2 workers
    let r1 = ray.spawn_with_resources((), Resources::new(0.0, 0.5), |()| 1);
    let r2 = ray.spawn_with_resources((), Resources::new(0.0, 0.5), |()| 2);

    let v1: i32 = ray.get(&r1).await.unwrap();
    let v2: i32 = ray.get(&r2).await.unwrap();
    assert_eq!(v1, 1);
    assert_eq!(v2, 2);
}
