# Architecture

## Processes

```text
Client ── RPC ──> Coordinator ── assignments/cancel ──> Worker
  │                    │                                 │
  │                    └── metadata only                 ├── operation registry
  └──────── exact object location ──────────────────────>├── local object store
                                                        └── object RPC
```

The coordinator is the only control-plane authority. Workers initiate control
requests and expose concrete addresses for object transfer.

## Identity and fencing

- A coordinator creates a new epoch on each start.
- A worker has a stable node ID, a process epoch, and a coordinator session ID.
- An assignment has a task ID, attempt number, and lease ID.
- Completion, failure, and cancellation acknowledgements must match the complete
  fence. Stale reports cannot mutate terminal state.

## Task states

```text
Waiting -> Runnable -> Assigned -> Running -> Succeeded
                                |          -> Failed
                                -> CancelRequested -> Cancelled
```

Worker loss returns retryable tasks to `Runnable` with a new attempt. Exhausted
attempts become `Failed`. External side effects are therefore at-least-once.

## Objects

Client puts are temporarily held by the coordinator. Task outputs stay in the
producing worker's immutable local store. The coordinator records only codec,
size, BLAKE3 digest, owner, and location.

A requester obtains metadata from the coordinator and then fetches worker-local
bytes directly. It verifies ID, length, and digest before deserialization.

## Resources

Resources are fixed-point values where 1000 milli-units equal one unit. The
coordinator reserves resources atomically with assignment and restores them only
on accepted completion, failure, worker expiry, or cancellation acknowledgement.

## Failure boundaries

The coordinator is intentionally not durable or highly available. Restarting it
creates a new epoch and ends the existing session. Workers and clients must
reconnect to the new session. Application state requires explicit checkpoints.
