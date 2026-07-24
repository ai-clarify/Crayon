//! Unit tests for [`super`] — the coordinator state machine. Split out of
//! coordinator.rs to keep the state-machine file focused on the contract.
use super::*;
fn descriptor() -> OperationDescriptor {
    OperationDescriptor {
        key: OperationKey::new("test", "copy", 1),
        input_codec: Codec::RawBytes,
        output_codec: Codec::RawBytes,
        max_inline_arg_bytes: 8,
    }
}
fn register(state: &mut CoordinatorState, node: NodeId) -> WorkerIdentity {
    state
        .register_worker(
            RegisterWorker {
                node_id: node,
                worker_epoch: WorkerEpoch::new(),
                advertise_addr: "127.0.0.1:9001".into(),
                resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                slots: 1,
                operations: vec![descriptor()],
            },
            0,
            100,
        )
        .unwrap()
}

#[test]
fn duplicate_registration_is_idempotent_only_when_exact() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let node = NodeId::new();
    let epoch = WorkerEpoch::new();
    let request = RegisterWorker {
        node_id: node,
        worker_epoch: epoch,
        advertise_addr: "127.0.0.1:9001".into(),
        resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
        slots: 1,
        operations: vec![descriptor()],
    };
    let first = state.register_worker(request.clone(), 0, 100).unwrap();
    let second = state.register_worker(request, 1, 100).unwrap();
    assert_eq!(first.session_id, second.session_id);
    let mut conflict = descriptor();
    conflict.output_codec = Codec::JsonV1;
    assert!(state
        .register_worker(
            RegisterWorker {
                node_id: node,
                worker_epoch: epoch,
                advertise_addr: "127.0.0.1:9001".into(),
                resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                slots: 1,
                operations: vec![conflict]
            },
            2,
            100
        )
        .is_err());
}

#[test]
fn failure_propagates_through_waiting_chain() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (_, first_output) = state
        .submit(
            descriptor().key.clone(),
            vec![TaskArg::Inline {
                codec: Codec::RawBytes,
                bytes: vec![1],
            }],
            ResourceSet::default(),
            1,
        )
        .unwrap();
    let (second, second_output) = state
        .submit(
            descriptor().key.clone(),
            vec![TaskArg::Object(first_output)],
            ResourceSet::default(),
            1,
        )
        .unwrap();
    let (third, _) = state
        .submit(
            descriptor().key.clone(),
            vec![TaskArg::Object(second_output)],
            ResourceSet::default(),
            1,
        )
        .unwrap();
    let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state
        .fail(
            &identity,
            assignment.fence,
            "boom".into(),
            FailureClass::Permanent,
            0,
        )
        .unwrap();
    assert!(matches!(state.tasks[&second].state, TaskState::Failed(_)));
    assert!(matches!(state.tasks[&third].state, TaskState::Failed(_)));
}

#[test]
fn cancel_keeps_resources_until_worker_ack() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (task, _) = state
        .submit(
            descriptor().key.clone(),
            vec![],
            ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
            1,
        )
        .unwrap();
    let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state.cancel(task).unwrap();
    assert_eq!(state.workers[&identity.node_id].free_slots, 0);
    state
        .acknowledge_cancel(&identity, assignment.fence)
        .unwrap();
    assert_eq!(state.workers[&identity.node_id].free_slots, 1);
}

#[test]
fn stale_attempt_cannot_complete_retry() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 2)
        .unwrap();
    let first = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state
        .fail(
            &identity,
            first.fence,
            "retry".into(),
            FailureClass::Transient,
            0,
        )
        .unwrap();
    // Transient retry backs off ~100ms; dispatch with now past the deadline.
    let second = state.assign_next(identity.node_id, 1_000).unwrap().unwrap();
    assert_ne!(first.fence.attempt, second.fence.attempt);
    assert_eq!(
        state.started(&identity, first.fence),
        Err(Error::StaleFence)
    );
}

#[test]
fn rejects_unschedulable_resources() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    register(&mut state, NodeId::new());
    let result = state.submit(
        descriptor().key,
        vec![],
        ResourceSet::cpu_gpu(2.0, 0.0).unwrap(),
        1,
    );
    assert!(matches!(result, Err(Error::InvalidResource(_))));
}

#[test]
fn scheduler_drains_queue_and_leaves_no_stale_index_entries() {
    // Submit -> assign -> complete many single-slot tasks in sequence and
    // confirm the runnable queue and waiting index return to empty. A leak
    // here is exactly the O(n) scan regression the queue was added to kill.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    for i in 0..50u8 {
        let (_, output) = state
            .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
            .unwrap();
        let assignment = state
            .assign_next(identity.node_id, 0)
            .unwrap()
            .expect("a freshly submitted task must be assignable");
        state.started(&identity, assignment.fence).unwrap();
        state
            .complete(
                &identity,
                crate::protocol::TaskCompletion {
                    fence: assignment.fence,
                    output_id: output,
                    codec: Codec::RawBytes,
                    size_bytes: 1,
                    checksum: crate::cluster::checksum(&[i]),
                    location: "127.0.0.1:9001".into(),
                    bytes: Some(vec![i].into()),
                },
            )
            .unwrap();
    }
    assert!(state.runnable.is_empty(), "runnable queue leaked entries");
    assert!(state.waiting.is_empty(), "waiting index leaked entries");
    assert!(state.assign_next(identity.node_id, 0).unwrap().is_none());
}

#[test]
fn cancelled_waiting_task_is_dropped_from_index() {
    // A task blocked on a dependency, then cancelled, must leave the waiting
    // index so reconciliation never revisits a terminal task.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let _ = register(&mut state, NodeId::new());
    let (_, dep_output) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    let (blocked, _) = state
        .submit(
            descriptor().key.clone(),
            vec![TaskArg::Object(dep_output)],
            ResourceSet::default(),
            1,
        )
        .unwrap();
    assert!(state.waiting.contains(&blocked));
    state.cancel(blocked).unwrap();
    assert!(!state.waiting.contains(&blocked));
    assert_eq!(state.tasks[&blocked].state, TaskState::Cancelled);
}

#[test]
fn locality_counters_track_owned_input_placement() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (_, first_output) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    let first = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state
        .complete(&identity, completion(first.fence, first_output))
        .unwrap();
    state
        .submit(
            descriptor().key.clone(),
            vec![TaskArg::Object(first_output)],
            ResourceSet::default(),
            1,
        )
        .unwrap();
    state.assign_next(identity.node_id, 0).unwrap().unwrap();
    assert_eq!(state.sched_assigns, 2);
    assert_eq!(state.sched_owned_input, 1);
    assert_eq!(state.sched_local_hits, 1);
}

#[test]
fn draining_worker_gets_no_assignments() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    state.drain(&identity).unwrap();
    assert!(state.assign_next(identity.node_id, 0).unwrap().is_none());
}

#[test]
fn draining_worker_can_complete_in_flight_task() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (task, output) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state.drain(&identity).unwrap();
    state
        .complete(&identity, completion(assignment.fence, output))
        .unwrap();
    assert_eq!(state.tasks[&task].state, TaskState::Succeeded);
}

#[test]
fn draining_worker_expiry_requeues_task() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new()); // lease deadline 100
    let (task, _) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 2)
        .unwrap();
    state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state.drain(&identity).unwrap();
    // Draining worker exits without reporting; the reaper re-queues.
    let expired = state.expire_workers(1_000).unwrap();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].node, identity.node_id);
    assert_eq!(expired[0].retried, 1);
    assert_eq!(state.tasks[&task].state, TaskState::Runnable);
}

fn completion(fence: TaskFence, output: ObjectId) -> TaskCompletion {
    TaskCompletion {
        fence,
        output_id: output,
        codec: Codec::RawBytes,
        size_bytes: 1,
        checksum: crate::cluster::checksum(&[7]),
        location: "127.0.0.1:9001".into(),
        bytes: Some(vec![7].into()),
    }
}

#[test]
fn release_reserved_output_is_rejected_then_task_completes() {
    // Releasing a live task's still-Reserved output must be refused; otherwise
    // the object is deleted and the task's completion panics on a missing key.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (_, output) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    assert_eq!(
        state.release_object(output),
        Err(Error::ObjectInUse(output))
    );
    state.started(&identity, assignment.fence).unwrap();
    state
        .complete(&identity, completion(assignment.fence, output))
        .unwrap();
    assert_eq!(
        state.tasks[&assignment.fence.task_id].state,
        TaskState::Succeeded
    );
}

#[test]
fn release_reclaims_terminal_task_and_object() {
    // A released output must drop both the object and its now-terminal producer
    // task, so the task table is not a lifetime-capped leak.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (task, output) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    let assignment = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state.started(&identity, assignment.fence).unwrap();
    state
        .complete(&identity, completion(assignment.fence, output))
        .unwrap();
    state.release_object(output).unwrap();
    assert!(!state.tasks.contains_key(&task));
    assert!(!state.objects.contains_key(&output));
}

#[test]
fn expired_worker_is_evicted() {
    // Dead ephemeral workers must be removed, not left as Dead records that
    // grow the map and inflate every submit's schedulability scan.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let node = NodeId::new();
    let _ = register(&mut state, node); // lease_deadline_ms = 0 + 100
    let expired = state.expire_workers(200).unwrap();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].node, node);
    assert!(!state.workers.contains_key(&node));
    assert!(!state.pending_deletes.contains_key(&node));
}

#[test]
fn cancellation_index_tracks_only_pending() {
    // cancellation_for scans this index instead of the whole task table, so it
    // must hold exactly the tasks awaiting a worker cancel ack.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (task, _) = state
        .submit(
            descriptor().key.clone(),
            vec![],
            ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
            1,
        )
        .unwrap();
    state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state.cancel(task).unwrap();
    assert!(state.cancel_requested.contains(&task));
    let fence = state.cancellation_for(&identity).unwrap().unwrap();
    assert_eq!(fence.task_id, task);
    state.acknowledge_cancel(&identity, fence).unwrap();
    assert!(state.cancel_requested.is_empty());
    assert_eq!(state.tasks[&task].state, TaskState::Cancelled);
}

#[test]
fn unknown_task_id_report_does_not_panic() {
    // Worker reports for a task the coordinator never had must map to
    // TaskNotFound, not a HashMap-index panic in the idempotent short-circuit.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let bogus = TaskFence {
        task_id: TaskId::new(),
        attempt: Attempt(1),
        lease_id: LeaseId::new(),
    };
    assert!(matches!(
        state.started(&identity, bogus),
        Err(Error::TaskNotFound(_))
    ));
    assert!(matches!(
        state.fail(&identity, bogus, "x".into(), FailureClass::Transient, 0),
        Err(Error::TaskNotFound(_))
    ));
    assert!(matches!(
        state.acknowledge_cancel(&identity, bogus),
        Err(Error::TaskNotFound(_))
    ));
    assert!(matches!(
        state.complete(&identity, completion(bogus, ObjectId::new())),
        Err(Error::TaskNotFound(_))
    ));
}

#[test]
fn permanent_failure_is_not_retried_despite_remaining_attempts() {
    // A Permanent failure (e.g. an operation panic) is terminal even with
    // attempts left; only Transient failures retry.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (task, _) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
        .unwrap();
    let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state
        .fail(
            &identity,
            a.fence,
            "boom".into(),
            FailureClass::Permanent,
            0,
        )
        .unwrap();
    assert!(matches!(state.tasks[&task].state, TaskState::Failed(_)));
    // A Transient failure with attempts left retries instead.
    let (task2, _) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
        .unwrap();
    let b = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state
        .fail(
            &identity,
            b.fence,
            "flaky".into(),
            FailureClass::Transient,
            0,
        )
        .unwrap();
    assert_eq!(state.tasks[&task2].state, TaskState::Runnable);
}

#[test]
fn transient_retry_defers_dispatch_until_backoff_elapses() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (task, _) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
        .unwrap();
    let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    let outcome = state
        .fail(
            &identity,
            a.fence,
            "flaky".into(),
            FailureClass::Transient,
            1_000,
        )
        .unwrap();
    // First retry (attempt 1) backs off base = 100ms → due at 1100.
    assert!(!outcome); // retried, not permanently failed
    assert_eq!(state.tasks[&task].state, TaskState::Runnable);
    assert_eq!(state.tasks[&task].not_before_ms, 1_100);
    // Not yet due: assign_next skips it and returns nothing.
    assert!(state
        .assign_next(identity.node_id, 1_050)
        .unwrap()
        .is_none());
    assert_eq!(state.tasks[&task].state, TaskState::Runnable);
    // Due: dispatched.
    assert!(state
        .assign_next(identity.node_id, 1_100)
        .unwrap()
        .is_some());
}

#[test]
fn backoff_grows_exponentially_across_attempts() {
    assert_eq!(retry_backoff_ms(1), 100);
    assert_eq!(retry_backoff_ms(2), 200);
    assert_eq!(retry_backoff_ms(3), 400);
    // Capped at RETRY_BACKOFF_CAP_MS, and a huge attempt cannot overflow.
    assert_eq!(retry_backoff_ms(30), RETRY_BACKOFF_CAP_MS);
    assert_eq!(retry_backoff_ms(u32::MAX), RETRY_BACKOFF_CAP_MS);
}

#[test]
fn lease_expiry_retry_also_applies_backoff() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    // Worker lease deadline = now(0) + 100.
    let identity = register(&mut state, NodeId::new());
    let (task, _) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
        .unwrap();
    state.assign_next(identity.node_id, 0).unwrap().unwrap();
    // Expire the lease at now = 200; attempt-1 retry → due at 300.
    let expired = state.expire_workers(200).unwrap();
    assert_eq!(expired.len(), 1);
    assert_eq!(expired[0].retried, 1);
    assert_eq!(state.tasks[&task].state, TaskState::Runnable);
    assert_eq!(state.tasks[&task].not_before_ms, 300);
}

#[test]
fn permanent_failure_leaves_no_backoff_deadline() {
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (task, _) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 3)
        .unwrap();
    let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    let outcome = state
        .fail(
            &identity,
            a.fence,
            "boom".into(),
            FailureClass::Permanent,
            500,
        )
        .unwrap();
    assert!(outcome); // permanently failed
    assert_eq!(state.tasks[&task].not_before_ms, 0);
}

#[test]
fn terminal_task_is_reclaimed_after_ttl_without_client_release() {
    // A fire-and-forget client never calls Release; the server backstop must
    // reclaim the terminal task + output so the table cannot grow to MAX_TASKS.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (task, output) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state.started(&identity, a.fence).unwrap();
    state
        .complete(&identity, completion(a.fence, output))
        .unwrap();
    assert_eq!(state.tasks[&task].state, TaskState::Succeeded);

    // First reap stamps terminal_since; nothing is due yet.
    assert_eq!(state.reap_terminal_tasks(1_000), 0);
    assert!(state.tasks.contains_key(&task));
    // Before the TTL: still retained.
    assert_eq!(
        state.reap_terminal_tasks(1_000 + TERMINAL_TASK_TTL_MS - 1),
        0
    );
    assert!(state.tasks.contains_key(&task));
    // At/after the TTL: reclaimed, output gone.
    assert_eq!(state.reap_terminal_tasks(1_000 + TERMINAL_TASK_TTL_MS), 1);
    assert!(!state.tasks.contains_key(&task));
    assert!(!state.objects.contains_key(&output));
}

#[test]
fn reclaim_spares_a_terminal_output_a_live_task_still_needs() {
    // A terminal producer whose output a still-pending consumer references
    // must NOT be reclaimed — the same precondition client Release enforces.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    let (producer, out) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state.started(&identity, a.fence).unwrap();
    state.complete(&identity, completion(a.fence, out)).unwrap();
    // Downstream consumer depends on the producer's output and is not terminal.
    let (_consumer, _) = state
        .submit(
            descriptor().key.clone(),
            vec![TaskArg::Object(out)],
            ResourceSet::default(),
            1,
        )
        .unwrap();
    // Well past the TTL, the producer stays because its output is in use.
    assert_eq!(state.reap_terminal_tasks(TERMINAL_TASK_TTL_MS * 2), 0);
    assert!(state.tasks.contains_key(&producer));
    assert!(state.objects.contains_key(&out));
}

#[test]
fn assign_prefers_a_task_with_local_inputs() {
    // A worker that already owns an object should get the downstream task that
    // consumes it, even when a non-local task sits ahead in the FIFO queue.
    let mut state = CoordinatorState::new(CoordinatorEpoch::new());
    let identity = register(&mut state, NodeId::new());
    // Produce object O on this worker.
    let (_, output) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    let a = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    state.started(&identity, a.fence).unwrap();
    state
        .complete(&identity, completion(a.fence, output))
        .unwrap();
    // Enqueue a non-local task first, then a local one that consumes O.
    let (non_local, _) = state
        .submit(descriptor().key.clone(), vec![], ResourceSet::default(), 1)
        .unwrap();
    let (local, _) = state
        .submit(
            descriptor().key.clone(),
            vec![TaskArg::Object(output)],
            ResourceSet::default(),
            1,
        )
        .unwrap();
    let picked = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    assert_eq!(picked.fence.task_id, local);
    assert_ne!(picked.fence.task_id, non_local);
    // The non-local task keeps its place and is served next.
    state.started(&identity, picked.fence).unwrap();
    state
        .complete(
            &identity,
            completion(picked.fence, state.tasks[&local].output),
        )
        .unwrap();
    let next = state.assign_next(identity.node_id, 0).unwrap().unwrap();
    assert_eq!(next.fence.task_id, non_local);
}
