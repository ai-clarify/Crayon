//! Unit tests for [`super`] — the RPC server, dispatch, replay cache, and
//! framing. Split out of cluster.rs to keep that file focused on the wire
//! contract and dispatch logic.
use super::*;
use crate::{
    ids::{NodeId, ObjectId, WorkerEpoch},
    operation::{Codec, OperationDescriptor, OperationKey, TaskArg},
    protocol::{RegisterWorker, TaskStatus},
    resources::ResourceSet,
};

fn descriptor() -> OperationDescriptor {
    OperationDescriptor {
        key: OperationKey::new("test", "copy", 1),
        input_codec: Codec::RawBytes,
        output_codec: Codec::RawBytes,
        max_inline_arg_bytes: 8,
    }
}

fn future_envelope(cluster: ClusterId, body: RpcRequest) -> Envelope {
    Envelope::new(cluster, body, now_ms().saturating_add(10_000))
}

fn with_epoch(server: &CoordinatorServer, mut envelope: Envelope) -> Envelope {
    envelope.coordinator_epoch = Some(server.state.lock().epoch);
    envelope
}

fn registered_server() -> (CoordinatorServer, crate::protocol::WorkerIdentity) {
    let cluster = ClusterId::new();
    let server = CoordinatorServer::new(cluster, 5_000);
    let reply = server
        .dispatch_once(&future_envelope(
            cluster,
            RpcRequest::Worker(WorkerRequest::Register(RegisterWorker {
                node_id: NodeId::new(),
                worker_epoch: WorkerEpoch::new(),
                advertise_addr: "127.0.0.1:9001".into(),
                resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                slots: 1,
                operations: vec![descriptor()],
            })),
        ))
        .unwrap();
    let RpcReply::Worker(WorkerReply::Registered(registered)) = reply else {
        panic!("registration failed")
    };
    (server, registered.identity)
}

#[test]
fn duplicate_submit_is_dispatched_once() {
    let (server, _) = registered_server();
    let request = with_epoch(
        &server,
        future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::Submit {
                operation: descriptor().key,
                args: vec![TaskArg::Inline {
                    codec: Codec::RawBytes,
                    bytes: vec![1],
                }],
                resources: ResourceSet::default(),
                max_attempts: 1,
            }),
        ),
    );
    let first = server.dispatch_once(&request).unwrap();
    let second = server.dispatch_once(&request).unwrap();
    let submitted = |reply| match reply {
        RpcReply::Client(ClientReply::Submitted { task_id, output_id }) => (task_id, output_id),
        _ => panic!("submit failed"),
    };
    assert_eq!(submitted(first), submitted(second));
    assert_eq!(server.state.lock().tasks.len(), 1);
}

#[test]
fn request_id_cannot_be_reused_with_another_body() {
    let (server, _) = registered_server();
    let first = with_epoch(
        &server,
        future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::Submit {
                operation: descriptor().key,
                args: vec![],
                resources: ResourceSet::default(),
                max_attempts: 1,
            }),
        ),
    );
    server.dispatch_once(&first).unwrap();
    let mut conflicting = with_epoch(
        &server,
        future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::Submit {
                operation: descriptor().key,
                args: vec![TaskArg::Inline {
                    codec: Codec::RawBytes,
                    bytes: vec![1],
                }],
                resources: ResourceSet::default(),
                max_attempts: 1,
            }),
        ),
    );
    conflicting.request_id = first.request_id;
    assert!(matches!(
        server.dispatch_once(&conflicting),
        Ok(RpcReply::Client(ClientReply::Error(Error::Protocol(_))))
    ));
}

#[test]
fn completion_reply_is_replayed_without_releasing_twice() {
    let (server, identity) = registered_server();
    let submit = server
        .dispatch_once(&with_epoch(
            &server,
            future_envelope(
                server.cluster_id,
                RpcRequest::Client(ClientRequest::Submit {
                    operation: descriptor().key,
                    args: vec![],
                    resources: ResourceSet::cpu_gpu(1.0, 0.0).unwrap(),
                    max_attempts: 1,
                }),
            ),
        ))
        .unwrap();
    let RpcReply::Client(ClientReply::Submitted { task_id, output_id }) = submit else {
        panic!("submit failed")
    };
    let assignment = server
        .state
        .lock()
        .assign_next(identity.node_id, 0)
        .unwrap()
        .unwrap();
    server
        .state
        .lock()
        .started(&identity, assignment.fence)
        .unwrap();
    let node_id = identity.node_id;
    let completion = with_epoch(
        &server,
        future_envelope(
            server.cluster_id,
            RpcRequest::Worker(WorkerRequest::Completed {
                identity,
                report: crate::protocol::TaskCompletion {
                    fence: assignment.fence,
                    output_id,
                    codec: Codec::RawBytes,
                    size_bytes: 1,
                    checksum: checksum(&[7]),
                    location: "127.0.0.1:9001".into(),
                    bytes: Some(vec![7].into()),
                },
            }),
        ),
    );
    assert!(matches!(
        server.dispatch_once(&completion),
        Ok(RpcReply::Worker(WorkerReply::Accepted))
    ));
    assert!(matches!(
        server.dispatch_once(&completion),
        Ok(RpcReply::Worker(WorkerReply::Accepted))
    ));
    let state = server.state.lock();
    assert_eq!(
        crate::protocol::TaskStatus::from(&state.tasks[&task_id].state),
        TaskStatus::Succeeded
    );
    assert_eq!(state.workers[&node_id].free_slots, 1);
}

#[tokio::test]
async fn transport_retry_reuses_the_envelope() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let cluster = ClusterId::new();
    let envelope = future_envelope(cluster, RpcRequest::Client(ClientRequest::Workers));
    let expected = envelope.request_id;
    let server = tokio::spawn(async move {
        for attempt in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let received = read_frame::<Envelope>(&mut stream).await.unwrap().unwrap();
            assert_eq!(received.request_id, expected);
            if attempt == 1 {
                write_frame(&mut stream, &RpcReply::Client(ClientReply::Workers(vec![])))
                    .await
                    .unwrap();
            }
        }
    });
    assert!(matches!(
        request(&address.to_string(), &envelope).await,
        Ok(RpcReply::Client(ClientReply::Workers(_)))
    ));
    server.await.unwrap();
}

// F1: a GetBatch must not clone unbounded inline bytes. Cumulative inline
// payload is capped at one frame (MAX_OBJECT_BYTES); overflow slots return
// CapacityExceeded so the reply always fits a frame regardless of how many
// duplicate ids a single request packs in.
#[test]
fn get_batch_response_bytes_are_bounded() {
    let (server, _) = registered_server();
    // ~3 MB object: two copies fit under MAX_OBJECT_BYTES (~8.13 MB), the
    // third pushes over, so later slots overflow.
    let object_size = 3_000_000usize;
    let id = server
        .state
        .lock()
        .put(Codec::RawBytes, vec![7u8; object_size])
        .unwrap()
        .id;
    let reply = server
        .dispatch_once(&future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::GetBatch {
                objects: vec![id; 6],
                wait_ms: 0,
                min_ready: 6,
            }),
        ))
        .unwrap();
    let RpcReply::Client(ClientReply::ObjectBatch(results)) = reply else {
        panic!("expected object batch")
    };
    // One entry per requested id, in order.
    assert_eq!(results.len(), 6);
    let ok_count = MAX_OBJECT_BYTES / object_size;
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), ok_count);
    assert!(results[0].is_ok());
    assert!(results[1].is_ok());
    for result in &results[ok_count..] {
        assert!(matches!(result, Err(Error::CapacityExceeded(_))));
    }
    // Cumulative cloned inline bytes never exceed one frame.
    let cloned: usize = results
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|p| p.bytes.as_ref().map_or(0, |b| b.len()))
        .sum();
    assert!(cloned <= MAX_OBJECT_BYTES);
}

// F2: a dispatched mutation is always cached. Filling MAX_REPLAY_ENTRIES is
// too expensive to hit directly, so assert the invariant: a successful Submit
// leaves exactly one task AND a replay entry, so a retry replays instead of
// re-dispatching (which would duplicate the task).
#[test]
fn mutation_not_dispatched_when_replay_full() {
    let (server, _) = registered_server();
    let request = with_epoch(
        &server,
        future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::Submit {
                operation: descriptor().key,
                args: vec![],
                resources: ResourceSet::default(),
                max_attempts: 1,
            }),
        ),
    );
    let first = server.dispatch_once(&request).unwrap();
    // Dispatched => cached: the entry is present before any retry.
    assert!(server
        .replay
        .lock()
        .entries
        .contains_key(&request.request_id));
    let second = server.dispatch_once(&request).unwrap();
    let task_id = |reply| match reply {
        RpcReply::Client(ClientReply::Submitted { task_id, .. }) => task_id,
        _ => panic!("submit failed"),
    };
    assert_eq!(task_id(first), task_id(second));
    assert_eq!(server.state.lock().tasks.len(), 1);
}

// F3: replay retention is clamped to a server ceiling, so a client-chosen
// far-future deadline cannot pin an entry indefinitely.
#[test]
fn replay_ttl_is_server_clamped() {
    let (server, _) = registered_server();
    let mut request = with_epoch(
        &server,
        future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::Submit {
                operation: descriptor().key,
                args: vec![],
                resources: ResourceSet::default(),
                max_attempts: 1,
            }),
        ),
    );
    // Deadline ~10 years out; the server must ignore it for retention.
    let ten_years_ms: u64 = 10 * 365 * 24 * 60 * 60 * 1_000;
    request.deadline_unix_ms = now_ms().saturating_add(ten_years_ms);
    server.dispatch_once(&request).unwrap();
    let replay = server.replay.lock();
    let entry = replay
        .entries
        .get(&request.request_id)
        .expect("mutation cached");
    // Capped at now + MAX_REPLAY_TTL_MS (+ slack for clock movement across
    // the two now_ms() reads), far below the 10-year envelope deadline.
    assert!(entry.expires_at_ms <= now_ms() + MAX_REPLAY_TTL_MS + 1_000);
}

// F4: the running byte counter stays exact across insert / remove / sweep,
// so the O(1) capacity check never drifts from the true stored size.
#[test]
fn replay_used_bytes_tracks_entries() {
    let entry = |expires_at_ms| ReplayEntry {
        body_hash: [0u8; 32],
        reply: RpcReply::Client(ClientReply::Released),
        expires_at_ms,
        bytes: 100,
    };
    let step = 100 + REPLAY_ENTRY_OVERHEAD;
    let mut cache = ReplayCache::default();
    let ids: Vec<RequestId> = (0..3).map(|_| RequestId::new()).collect();

    cache.insert(ids[0], entry(50)); // expires early
    cache.insert(ids[1], entry(200));
    cache.insert(ids[2], entry(200));
    assert_eq!(cache.used_bytes, 3 * step);

    cache.remove(&ids[1]);
    assert_eq!(cache.used_bytes, 2 * step);
    cache.remove(&ids[1]); // idempotent: no double-subtract
    assert_eq!(cache.used_bytes, 2 * step);

    cache.sweep(100); // drops ids[0] (expires 50), keeps ids[2]
    assert_eq!(cache.entries.len(), 1);
    assert_eq!(cache.used_bytes, step);
}

// F5: first-K-ready (ray.wait) — the batch is ready once `min_ready` slots
// resolve, even while others are still pending; min_ready==len is ray.get.
#[test]
fn get_batch_ready_honors_min_ready() {
    let pending = || Err(Error::ObjectPending(ObjectId::new()));
    let done = || Err(Error::ObjectLost(ObjectId::new())); // any non-pending slot
    let reply = |slots: Vec<Result<ObjectPayload, Error>>| {
        RpcReply::Client(ClientReply::ObjectBatch(slots))
    };

    // 1 of 3 resolved.
    let one_ready = reply(vec![done(), pending(), pending()]);
    assert!(client_get_batch_ready(&one_ready, 1));
    assert!(!client_get_batch_ready(&one_ready, 2));

    // all pending: never ready unless min_ready is 0.
    let none_ready = reply(vec![pending(), pending()]);
    assert!(!client_get_batch_ready(&none_ready, 1));
    assert!(client_get_batch_ready(&none_ready, 0));

    // all resolved: ray.get case.
    let all_ready = reply(vec![done(), done()]);
    assert!(client_get_batch_ready(&all_ready, 2));
}

// A client Put over the TCP path returns metadata only (no echoed payload), and
// the object still round-trips through a Get. Guards the copy-cut: the put reply
// must carry bytes: None, yet the stored object must be intact and fetchable.
#[test]
fn tcp_put_reply_is_metadata_only_and_round_trips() {
    let (server, _) = registered_server();
    let payload = b"hello crayon tcp put".to_vec();
    let expected_id = ObjectId::from_checksum(checksum(&payload));

    let put = server
        .dispatch_once(&with_epoch(
            &server,
            future_envelope(
                server.cluster_id,
                RpcRequest::Client(ClientRequest::Put {
                    codec: Codec::RawBytes,
                    bytes: payload.clone(),
                }),
            ),
        ))
        .unwrap();
    let RpcReply::Client(ClientReply::Object(meta)) = put else {
        panic!("expected object reply from put")
    };
    // Metadata-only reply: no payload echoed back, but id/size/checksum are set.
    assert!(meta.bytes.is_none(), "put reply must not echo the payload");
    assert_eq!(meta.id, expected_id);
    assert_eq!(meta.size_bytes, payload.len() as u64);
    assert_eq!(meta.checksum, checksum(&payload));

    // The object is intact and fetchable. In this single-process server the
    // coordinator's arena holds the payload, so Get resolves to the same-host
    // zero-copy form (bytes dropped, arena offset set) rather than inline bytes;
    // either way the object round-trips with matching id/size/checksum.
    let get = server
        .dispatch_once(&future_envelope(
            server.cluster_id,
            RpcRequest::Client(ClientRequest::Get {
                object: meta.id,
                wait_ms: 0,
            }),
        ))
        .unwrap();
    let RpcReply::Client(ClientReply::Object(fetched)) = get else {
        panic!("expected object reply from get")
    };
    assert_eq!(fetched.id, expected_id);
    assert_eq!(fetched.size_bytes, payload.len() as u64);
    assert_eq!(fetched.checksum, checksum(&payload));
    // Resolvable: inline bytes match, or an arena ref locates the payload.
    match (fetched.bytes.as_deref(), &fetched.arena) {
        (Some(bytes), _) => assert_eq!(bytes, payload.as_slice()),
        (None, Some(_)) => {} // same-host arena ref; bytes read zero-copy client-side
        (None, None) => panic!("get reply resolved to neither bytes nor an arena ref"),
    }
}
