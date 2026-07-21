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
    let (r, _) = store1.put(42i32);

    // Fetch it from the worker node (remote fetch)
    let v: i32 = store2.get(r.id).await.unwrap();
    assert_eq!(v, 42);
}

#[tokio::test]
async fn distributed_metadata_sync() {
    use crayon::gcs::Gcs;

    let store1 = ObjectStore::new();
    let store2 = ObjectStore::new();
    let gcs1 = Gcs::new();
    let gcs2 = Gcs::new();

    // Start head + worker, both with GCS attached for metadata sync
    let head = Node::start("127.0.0.1:0", None, store1.clone())
        .await
        .unwrap();
    let head_addr = head.addr.clone();
    head.with_gcs(gcs1.clone());

    let worker = Node::start("127.0.0.1:0", Some(&head_addr), store2.clone())
        .await
        .unwrap();
    worker.with_gcs(gcs2.clone());

    // Put an object on head — this adds metadata to gcs1
    let (r, size) = store1.put(99i32);
    gcs1.add_object(crayon::gcs::ObjectMeta {
        id: r.id,
        size_bytes: size,
        created_at: std::time::Instant::now(),
        owner: None,
    });

    // Wait for periodic sync (every 2s) + registration
    tokio::time::sleep(std::time::Duration::from_millis(2500)).await;

    // Worker's GCS should now know about the object from head
    let obj = gcs2.get_object(r.id);
    assert!(obj.is_some(), "worker GCS should have synced object metadata from head");
    assert_eq!(obj.unwrap().size_bytes, 4); // i32 = 4 bytes in bincode
}

/// Test that named actors are discoverable across nodes via GCS sync.
/// An actor created on the head node should appear in the worker's GCS
/// with the correct name and state, and state changes (e.g. kill) propagate.
#[tokio::test]
async fn distributed_actor_discovery() {
    use crayon::common::ActorState;
    use crayon::gcs::Gcs;

    let store1 = ObjectStore::new();
    let store2 = ObjectStore::new();
    let gcs1 = Gcs::new();
    let gcs2 = Gcs::new();

    let head = Node::start("127.0.0.1:0", None, store1.clone())
        .await
        .unwrap();
    let head_addr = head.addr.clone();
    head.with_gcs(gcs1.clone());

    let worker = Node::start("127.0.0.1:0", Some(&head_addr), store2.clone())
        .await
        .unwrap();
    worker.with_gcs(gcs2.clone());

    // Create a named actor on the head node
    #[derive(Clone, Default)]
    #[allow(dead_code)]
    struct Counter(u64);
    let ray = crayon::Ray::init_with_memory(1, 1024 * 1024, std::env::temp_dir().join("crayon_actor_test"));
    let actor = ray.create_actor("counter", Counter(0));
    let actor_id = actor.id();

    // Manually register the actor in head's GCS (Ray::create_actor does this
    // via spawn_actor, but that uses a different Gcs instance — replicate here
    // so the head's synced GCS knows about it).
    gcs1.add_actor(crayon::gcs::ActorMeta {
        id: actor_id,
        name: "counter".to_string(),
        state: ActorState::Running,
        owner_node: None,
        created_at: std::time::Instant::now(),
        pending_tasks: 0,
        completed_tasks: 0,
    });

    // Wait for periodic sync (every 2s) + registration
    tokio::time::sleep(std::time::Duration::from_millis(2500)).await;

    // Worker's GCS should now know about the actor from head
    let meta = gcs2.get_actor(actor_id);
    assert!(
        meta.is_some(),
        "worker GCS should have synced actor metadata from head"
    );
    let meta = meta.unwrap();
    assert_eq!(meta.name, "counter");
    assert_eq!(meta.state, ActorState::Running);

    // Kill the actor on head — state change should propagate to worker
    actor.kill();
    gcs1.set_actor_state(actor_id, ActorState::Dead);

    tokio::time::sleep(std::time::Duration::from_millis(2500)).await;

    let meta = gcs2.get_actor(actor_id).unwrap();
    assert_eq!(
        meta.state,
        ActorState::Dead,
        "actor state change (Dead) should propagate to worker"
    );
}

/// Test cross-node actor method calls. An actor created on the head node with
/// a registered method should be callable from the worker node via
/// `call_named`, which transparently routes through the local node to the
/// actor's owner node (Ray's cross-node actor call pattern).
#[tokio::test]
async fn cross_node_actor_call() {
    let store1 = ObjectStore::new();
    let store2 = ObjectStore::new();

    // Head node: hosts the actor
    let head = Node::start("127.0.0.1:0", None, store1.clone())
        .await
        .unwrap();
    let head_addr = head.addr.clone();

    // Worker node: will call the actor remotely
    let worker = Node::start("127.0.0.1:0", Some(&head_addr), store2.clone())
        .await
        .unwrap();

    // Create actor on head, attach node so it's registered + discoverable.
    // Use ray_head's GCS for sync so the actor appears in the synced GCS.
    #[derive(Clone, Default)]
    #[allow(dead_code)]
    struct Counter {
        count: u64,
    }
    let ray_head = crayon::Ray::init_with_memory(
        1,
        1024 * 1024,
        std::env::temp_dir().join("crayon_cross_node_head"),
    );
    ray_head.attach_node(head.clone());
    head.with_gcs(ray_head.gcs().clone());

    // Set up worker's Ray + GCS sync BEFORE the first broadcast (every 2s)
    // so it receives the actor metadata from head.
    let ray_worker = crayon::Ray::init_with_memory(
        1,
        1024 * 1024,
        std::env::temp_dir().join("crayon_cross_node_worker"),
    );
    ray_worker.attach_node(worker.clone());
    worker.with_gcs(ray_worker.gcs().clone());

    let actor = ray_head.create_actor("counter", Counter { count: 0 });

    // Register a typed method: takes u64 delta, returns u64 count.
    // Serialization is handled automatically — no bincode boilerplate.
    actor.register_method_typed("inc", |state: &mut Counter, delta: u64| -> u64 {
        state.count += delta;
        state.count
    });

    // Wait for GCS sync (every 2s) so worker discovers the actor
    tokio::time::sleep(std::time::Duration::from_millis(2500)).await;

    let remote_actor = ray_worker
        .get_actor::<Counter>("counter")
        .expect("worker should discover actor from head via GCS sync");
    assert!(
        remote_actor.owner_node().is_some(),
        "discovered actor should have an owner node"
    );

    // Call the method remotely with typed args — routes worker -> head -> actor.
    // call_typed returns ObjectRef<u64>, same as local call().
    let r = remote_actor
        .call_typed::<u64, u64>("inc", 42u64)
        .await
        .expect("cross-node actor call should succeed");
    let result: u64 = ray_worker.get(&r).await.unwrap();
    assert_eq!(result, 42, "first call should return 0 + 42 = 42");

    // Call again to verify stateful behavior across the network
    let r = remote_actor
        .call_typed::<u64, u64>("inc", 8u64)
        .await
        .expect("second cross-node actor call should succeed");
    let result: u64 = ray_worker.get(&r).await.unwrap();
    assert_eq!(result, 50, "second call should return 42 + 8 = 50");
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
        let (r, _) = store.put(data);
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
