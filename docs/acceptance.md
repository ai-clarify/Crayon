# Acceptance

## Pull-request gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo check --all-targets
cargo test
```

## Real-process feature matrix

`tests/cluster_process.rs` starts independent coordinator, worker, and client OS
processes on dynamic loopback ports. Every child is killed and reaped on success
or panic.

| Feature | Process test | Production path |
|---|---|---|
| Remote execution and typed readiness | `remote_execution_and_typed_readiness` | coordinator registration, worker polling, public client submit/result |
| Multi-stage DAG and worker-local transfer | `multi_stage_dag_fetches_worker_local_output` | reserved dependency, waiting/runnable transition, locate + `GetLocal`, checksum validation |
| Capability and fixed-point resource routing | `capability_and_resource_routing_are_enforced` | operation catalog, worker resources, unschedulable rejection |
| Cooperative running cancellation | `running_cancellation_releases_capacity_after_ack` | `CancelRequested`, worker poll/ack, resource release, cancelled output |
| Worker lease loss, retry, and object loss | `worker_loss_retries_and_loses_owned_objects` | lease reaper, attempt increment, alternate worker, owner loss |
| Frame bound and service recovery | `protocol_deadline_and_frame_limits_are_bounded` | pre-allocation frame limit, bounded connection handling, post-failure liveness |

These are the product acceptance gates. Unit tests remain focused checks for
pure state transitions and arithmetic; they are not substitutes for feature
acceptance.

RPC retries reuse the original request ID. The coordinator replays the first
response for that ID until its request deadline, so response loss cannot repeat a
mutation within one live coordinator epoch. Coordinator restart clears this
in-memory history, and task execution remains at-least-once across worker attempts.

## Manual smoke

```bash
cargo build --bin crayon-cluster
target/debug/crayon-cluster coordinator 127.0.0.1:7000 &
target/debug/crayon-cluster worker 127.0.0.1:7000 127.0.0.1:7001 &
target/debug/crayon-cluster submit 127.0.0.1:7000 20 22
```

Expected output: `42`.

## Benchmark

The benchmark is a release-only measurement, not a functional PR gate. A
semantic smoke run (`--warmups 1 --samples 3`) must pass on supported platforms
before a release. Full results are generated on the target machine and attached
to the GitHub Release; raw samples are not committed.

## Not yet claimed

Coordinator restart persistence, stale-report replay injection, chunked objects,
connection saturation, TLS/authentication, and hard preemption remain outside
the current release contract.
