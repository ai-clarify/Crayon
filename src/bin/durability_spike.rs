//! Durability spike: de-risk whether coordinator durability is a persistence
//! add-on rather than a redesign. Drives the **real** `CoordinatorState` through
//! an in-flight lifecycle, snapshots it with the only durability primitive
//! (`bincode` serialize -> deserialize), and asserts the recovery model on the
//! rehydrated state. Then measures snapshot cost at a full task table.
//!
//! Object payloads are excluded from the snapshot (`#[serde(skip)]` on
//! `ObjectRecord::bytes`): durability persists the control-plane graph, not the
//! worker-local, re-derivable object bytes.
//!
//! Run: `cargo run --release --bin durability_spike [cost_task_count]`

use std::time::Instant;

use crayon::{
    coordinator::{CoordinatorState, TaskState, MAX_TASKS},
    ids::{CoordinatorEpoch, NodeId, WorkerEpoch},
    operation::{Codec, OperationDescriptor, OperationKey},
    protocol::{RegisterWorker, TaskCompletion, WorkerIdentity},
    resources::ResourceSet,
};

fn op() -> OperationDescriptor {
    OperationDescriptor {
        key: OperationKey::new("bench", "op", 1),
        input_codec: Codec::RawBytes,
        output_codec: Codec::RawBytes,
        max_inline_arg_bytes: 1024,
    }
}

fn worker(state: &mut CoordinatorState, slots: u32, cpu: f64) -> WorkerIdentity {
    state
        .register_worker(
            RegisterWorker {
                node_id: NodeId::new(),
                worker_epoch: WorkerEpoch::new(),
                advertise_addr: "127.0.0.1:9000".into(),
                resources: ResourceSet::cpu_gpu(cpu, 0.0).unwrap(),
                slots,
                operations: vec![op()],
            },
            0,
            60_000,
        )
        .expect("register worker")
}

fn snapshot(state: &CoordinatorState) -> Vec<u8> {
    bincode::serialize(state).expect("serialize state")
}
fn restore(bytes: &[u8]) -> CoordinatorState {
    bincode::deserialize(bytes).expect("deserialize state")
}

fn main() {
    // ---- Correctness: recovery model on the rehydrated state ----
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = worker(&mut state, 2, 2.0);
    // One task left in-flight (assigned + started), one still runnable.
    let (in_flight, output) = state
        .submit(op().key, vec![], ResourceSet::cpu_gpu(1.0, 0.0).unwrap(), 2)
        .unwrap();
    let (runnable, _) = state
        .submit(op().key, vec![], ResourceSet::cpu_gpu(1.0, 0.0).unwrap(), 2)
        .unwrap();
    let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state.started(&identity, assignment.fence).unwrap();
    let epoch = state.epoch;
    let revision = state.revision;
    let free_slots = state.workers[&identity.node_id].free_slots;

    // Crash + recover.
    let bytes = snapshot(&state);
    let recovered = restore(&bytes);

    // 1. Exact reconstruction — epoch, revision, states, reservations preserved.
    assert_eq!(recovered.epoch, epoch, "epoch not preserved");
    assert_eq!(recovered.revision, revision, "revision not preserved");
    assert_eq!(recovered.tasks[&in_flight].state, TaskState::Running);
    assert_eq!(recovered.tasks[&runnable].state, TaskState::Runnable);
    assert_eq!(
        recovered.workers[&identity.node_id].free_slots, free_slots,
        "reservation reset on restore"
    );

    // 2. Fencing survives — a worker holding its pre-crash fence completes the
    //    in-flight task on the recovered coordinator; exactly one slot released.
    let mut recovered_a = restore(&bytes);
    let report = TaskCompletion {
        fence: assignment.fence,
        output_id: output,
        codec: Codec::RawBytes,
        size_bytes: 1,
        checksum: crayon::cluster::checksum(&[1]),
        location: "127.0.0.1:9000".into(),
        bytes: Some(vec![1].into()),
    };
    recovered_a
        .complete(&identity, report)
        .expect("fenced completion on recovered state");
    assert_eq!(recovered_a.tasks[&in_flight].state, TaskState::Succeeded);
    assert_eq!(
        recovered_a.workers[&identity.node_id].free_slots,
        free_slots + 1,
        "completion did not release exactly one slot"
    );

    // 3. Clean reconcile — from the same snapshot, with the worker's lease
    //    expired, the retryable in-flight attempt returns to Runnable.
    let mut recovered_b = restore(&bytes);
    let expired = recovered_b.expire_workers(1_000_000).unwrap();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].node, identity.node_id);
    assert_eq!(expired[0].retried, 1);
    assert_eq!(recovered_b.tasks[&in_flight].state, TaskState::Runnable);
    assert!(recovered_b.tasks[&in_flight].assigned.is_none());

    // 4. Snapshot, not command-replay — replaying submit mints a different id.
    let mut fresh = CoordinatorState::new(CoordinatorEpoch::new());
    let _ = worker(&mut fresh, 2, 2.0);
    let (replayed, _) = fresh
        .submit(op().key, vec![], ResourceSet::cpu_gpu(1.0, 0.0).unwrap(), 2)
        .unwrap();
    assert_ne!(
        replayed, in_flight,
        "replayed submit reproduced the same id"
    );

    // ---- Cost: snapshot size + serialize/deserialize latency at scale ----
    let count = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(MAX_TASKS - 1)
        .min(MAX_TASKS - 1);
    let mut big = CoordinatorState::new(CoordinatorEpoch::new());
    let _ = worker(&mut big, 1_000_000, 1024.0);
    for _ in 0..count {
        big.submit(op().key, vec![], ResourceSet::default(), 1)
            .unwrap();
    }
    let t0 = Instant::now();
    let snap = snapshot(&big);
    let serialize = t0.elapsed();
    let t1 = Instant::now();
    let back = restore(&snap);
    let deserialize = t1.elapsed();
    assert_eq!(back.tasks.len(), big.tasks.len(), "round-trip lost tasks");

    println!(
        "recovery asserts: 4/4 passed (reconstruction, fencing, reconcile, snapshot-not-replay)"
    );
    println!(
        "cost @ {} tasks / {} objects: snapshot={:.2} MiB ({} B/task)  serialize={:?}  deserialize={:?}",
        big.tasks.len(),
        big.objects.len(),
        snap.len() as f64 / (1024.0 * 1024.0),
        snap.len() / big.tasks.len().max(1),
        serialize,
        deserialize,
    );
}
