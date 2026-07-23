//! Durability / HA pre-research spike (THROWAWAY).
//!
//! ponytail: this exists only to de-risk the ONE hardest question in the
//! "coordinator durability" feature before anyone commits to building it.
//! Delete after the pre-research decision is made.
//!
//! Highest-risk point under test
//! -----------------------------
//! Crayon's coordinator is in-memory and non-durable: `docs/architecture.md`
//! says "Restarting it creates a new epoch and ends the existing session."
//! The open question is NOT "can we write bytes to disk" — it is whether a
//! restarted coordinator can reconstruct state that keeps **epoch fencing,
//! task terminal states, and resource reservations** exactly consistent, so
//! that in-flight worker attempts complete across the crash WITHOUT being
//! re-executed or double-counted.
//!
//! Two candidate seams, and why the choice is non-obvious:
//!   A. Command log (WAL of RPC mutations), replayed on boot.
//!      FAILS: `submit`/`assign_next` mint fresh random ids (`TaskId::new`,
//!      `ObjectId::new`, `LeaseId::new`) via `Uuid::new_v4`. Replaying the
//!      command produces DIFFERENT ids than the client already observed. The
//!      log would have to also record every generated id — i.e. it degenerates
//!      into a state delta anyway.
//!   B. State snapshot keyed on the monotonic `revision` counter.
//!      WORKS: the entire `CoordinatorState` is already built from
//!      `Serialize` parts; persisting it preserves exact ids, epoch, terminal
//!      states, and reservations. `revision` (bumped by every `changed()`) is
//!      a ready-made log sequence number.
//!
//! This spike drives the REAL `crayon::coordinator::CoordinatorState` through a
//! realistic in-flight lifecycle, simulates a crash by serializing +
//! deserializing (the only durability primitive), and asserts B holds and A
//! diverges. Production would layer an incremental WAL + periodic snapshot on
//! the same seam; recovery correctness is identical because the seam is the
//! serializable state plus the monotonic revision.
//!
//! Run: `cargo run --bin durability_spike`

use crayon::cluster::checksum;
use crayon::coordinator::{CoordinatorState, ObjectState, TaskState};
use crayon::ids::{CoordinatorEpoch, NodeId, WorkerEpoch};
use crayon::operation::{Codec, OperationDescriptor, OperationKey, TaskArg};
use crayon::protocol::{RegisterWorker, TaskCompletion};
use crayon::resources::ResourceSet;

const WORKER_ADDR: &str = "127.0.0.1:9001";

fn copy_op() -> OperationDescriptor {
    OperationDescriptor {
        key: OperationKey::new("test", "copy", 1),
        input_codec: Codec::RawBytes,
        output_codec: Codec::RawBytes,
        max_inline_arg_bytes: 8,
    }
}

/// The only durability primitive: an atomic snapshot of the whole state,
/// tagged with the revision it reflects. Production would fsync this to a log
/// dir; here it is just bytes in memory.
fn snapshot(state: &CoordinatorState) -> (u64, Vec<u8>) {
    (
        state.revision.0,
        bincode::serialize(state).expect("state is serializable"),
    )
}

fn recover(image: &[u8]) -> CoordinatorState {
    bincode::deserialize(image).expect("snapshot round-trips")
}

fn main() {
    // --- Build a realistic in-flight cluster on the REAL state machine. ---
    let epoch = CoordinatorEpoch::new();
    let mut state = CoordinatorState::new(epoch);

    let node = NodeId::new();
    let identity = state
        .register_worker(
            RegisterWorker {
                node_id: node,
                worker_epoch: WorkerEpoch::new(),
                advertise_addr: WORKER_ADDR.into(),
                resources: ResourceSet::cpu_gpu(2.0, 0.0).unwrap(),
                slots: 2,
                operations: vec![copy_op()],
            },
            0,
            60_000,
        )
        .unwrap();

    // Task A and Task B, each cpu=1.0. Drive BOTH into Running so we have two
    // live fences and a fully-committed resource reservation at crash time.
    let (task_a, obj_a) = state
        .submit(
            copy_op().key,
            vec![TaskArg::Inline {
                codec: Codec::RawBytes,
                bytes: vec![1],
            }],
            ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
            2,
        )
        .unwrap();
    let (task_b, _obj_b) = state
        .submit(
            copy_op().key,
            vec![TaskArg::Inline {
                codec: Codec::RawBytes,
                bytes: vec![2],
            }],
            ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
            1,
        )
        .unwrap();

    let assign_a = state.assign_next(node).unwrap().unwrap();
    state.started(&identity, assign_a.fence).unwrap();
    let assign_b = state.assign_next(node).unwrap().unwrap();
    state.started(&identity, assign_b.fence).unwrap();

    // Ground truth just before the crash.
    assert_eq!(state.tasks[&task_a].state, TaskState::Running);
    assert_eq!(state.tasks[&task_b].state, TaskState::Running);
    assert_eq!(state.workers[&node].free_slots, 0);
    assert_eq!(state.workers[&node].available.get("cpu").milli(), 0);
    let (rev_before, image) = snapshot(&state);
    println!(
        "[snapshot] persisted at revision {rev_before}, {} bytes",
        image.len()
    );

    // --- CRASH: the coordinator process dies. In-flight state is gone. ---
    drop(state);
    println!("[crash] coordinator process lost all in-memory state");

    // --- RECOVER from the last durable snapshot. ---
    let mut recovered = recover(&image);
    println!(
        "[recover] rebuilt state from revision {}",
        recovered.revision.0
    );

    // 1. Exact reconstruction: epoch, revision, ids, terminal states, and
    //    reservations survive byte-for-byte.
    assert_eq!(
        recovered.epoch, epoch,
        "epoch must survive (fences depend on it)"
    );
    assert_eq!(recovered.revision.0, rev_before, "revision preserved");
    assert_eq!(recovered.tasks[&task_a].state, TaskState::Running);
    assert_eq!(recovered.tasks[&task_b].state, TaskState::Running);
    assert!(
        recovered.objects.contains_key(&obj_a),
        "object ids preserved"
    );
    assert_eq!(
        recovered.workers[&node].free_slots, 0,
        "reservations not reset to full on recovery"
    );
    assert_eq!(recovered.workers[&node].available.get("cpu").milli(), 0);
    println!("[ok] recovery preserved epoch, revision, ids, task states, reservations");

    // 2. FENCING SURVIVES: the worker still holds `identity` + a pre-crash fence.
    //    It reports completion to the RECOVERED coordinator. This must be
    //    accepted — in-flight work completes across a coordinator restart, which
    //    is exactly what today's design forbids.
    //    (assign_next picks via HashMap order, so use the assignment's own
    //    task/output rather than assuming which task it was.)
    let done_task = assign_a.fence.task_id;
    let done_output = assign_a.output_id;
    let other_task = if done_task == task_a { task_b } else { task_a };
    let report = TaskCompletion {
        fence: assign_a.fence,
        output_id: done_output,
        codec: Codec::RawBytes,
        size_bytes: 1,
        checksum: checksum(&[7]),
        location: WORKER_ADDR.into(),
        bytes: Some(vec![7]),
    };
    recovered
        .complete(&identity, report)
        .expect("pre-crash fence must validate against recovered state");
    assert_eq!(recovered.tasks[&done_task].state, TaskState::Succeeded);
    assert_eq!(
        recovered.objects[&done_output].state,
        ObjectState::Available
    );
    assert_eq!(
        recovered.workers[&node].free_slots, 1,
        "exactly one slot released (no double-count)"
    );
    // The other task is untouched and still running under its own fence.
    assert_eq!(recovered.tasks[&other_task].state, TaskState::Running);
    println!("[ok] pre-crash fence completed an in-flight task on the recovered coordinator");

    // 3. WORKER LOST across the outage (the sharp branch): recover a SECOND
    //    copy from the same snapshot, but this time the worker never reconnects,
    //    so its lease (deadline 60_000ms, set at register time) is already
    //    expired. The existing `expire_workers` sweep must reconcile in-flight
    //    attempts WITHOUT leaking reservations, resurrecting attempts, or
    //    accepting a superseded report.
    let mut recovered2 = recover(&image);
    let expired = recovered2.expire_workers(60_001).unwrap();
    assert_eq!(
        expired,
        vec![node],
        "the stale-lease worker is reaped on recovery"
    );
    // In-flight attempts cleared: nothing stays Assigned/Running. The
    // max_attempts=2 task returns to Runnable (retryable, attempt not lost); the
    // max_attempts=1 task is terminally Failed (exhausted) — neither duplicated.
    let running = recovered2
        .tasks
        .values()
        .filter(|t| matches!(t.state, TaskState::Assigned | TaskState::Running))
        .count();
    let runnable = recovered2
        .tasks
        .values()
        .filter(|t| t.state == TaskState::Runnable)
        .count();
    let failed = recovered2
        .tasks
        .values()
        .filter(|t| matches!(t.state, TaskState::Failed(_)))
        .count();
    assert_eq!(running, 0, "no in-flight attempt survives a lost worker");
    assert_eq!(
        runnable, 1,
        "retryable in-flight attempt returned to Runnable"
    );
    assert_eq!(
        failed, 1,
        "exhausted in-flight attempt is terminally Failed, not resurrected"
    );
    assert!(
        recovered2.tasks.values().all(|t| t.assigned.is_none()),
        "no task holds a stale assignment after the sweep"
    );
    // A completion carrying a pre-crash fence, from a worker whose session is
    // now Dead, is rejected — the superseded holder cannot mutate recovered state.
    let stale = recovered2.complete(
        &identity,
        TaskCompletion {
            fence: assign_a.fence,
            output_id: assign_a.output_id,
            codec: Codec::RawBytes,
            size_bytes: 1,
            checksum: checksum(&[7]),
            location: WORKER_ADDR.into(),
            bytes: Some(vec![7]),
        },
    );
    assert!(
        matches!(stale, Err(crayon::Error::StaleFence)),
        "old-fence report after recovery must be rejected, got {stale:?}"
    );
    println!("[ok] worker-lost recovery reconciled in-flight work and rejected the stale fence");

    // 4. Why NOT a command-replay log: replaying `submit` on a fresh state mints
    //    a different task id, diverging from what the client was told.
    let mut replayed = CoordinatorState::new(epoch);
    replayed
        .register_worker(
            RegisterWorker {
                node_id: node,
                worker_epoch: WorkerEpoch::new(),
                advertise_addr: WORKER_ADDR.into(),
                resources: ResourceSet::cpu_gpu(2.0, 0.0).unwrap(),
                slots: 2,
                operations: vec![copy_op()],
            },
            0,
            60_000,
        )
        .unwrap();
    let (task_a_replayed, _) = replayed
        .submit(
            copy_op().key,
            vec![TaskArg::Inline {
                codec: Codec::RawBytes,
                bytes: vec![1],
            }],
            ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
            2,
        )
        .unwrap();
    assert_ne!(
        task_a, task_a_replayed,
        "command-replay mints fresh ids -> diverges from client-observed id"
    );
    println!(
        "[ok] confirmed command-replay diverges (fresh id) -> snapshot seam is the correct one"
    );

    println!("\nSPIKE PASSED: durability is a persistence-layer add-on on the existing");
    println!("`revision`-tagged serializable state — not a coordinator redesign.");
}
