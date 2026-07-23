# Crayon v0.5.0

Data-plane and RL-pipeline release: a shared-memory object arena lifts the object
size cap and makes same-host transfer zero-copy, a Python client lands, and a
real-LLM RL pipeline ships end to end. See [CHANGELOG.md](CHANGELOG.md) for the
full list.

## Compatibility notice

Wire protocol **major version 4** (up from 3): the shared-memory arena added
`ArenaReserve`/`ArenaCommit` request variants and connection pooling changed the
transport, shifting request encodings — wire-incompatible with 3. Coordinator,
worker, and client processes must be upgraded together; a mixed-version cluster
is rejected at the envelope check.

## Highlights

- **Shared-memory object arena.** A plasma-style host-local mmap arena: a client
  reserves a slot, writes its bytes into the mapping, and commits, so a same-host
  `put`/`get` never crosses the 8 MiB RPC frame cap — objects scale to gigabytes.
  Readers map the arena once and read any object as a slice; cross-host peers fall
  back to the frame-bounded network path. Large transfers use parallel BLAKE3 and
  parallel memcpy.
- **Python client (`crayon` on PyPI).** pyo3 bindings over `ClusterClient`,
  installable with `pip install crayon`.
- **Real-LLM RL pipeline.** An `llm-actor` rollout op backed by a resident Python
  LLM sidecar (weights synced from the arena once per policy version), a Rust
  `llm-judge`, a Python driver, and a micro-batched REINFORCE learner.
- **Connection pooling.** Multiple frames per TCP connection + client-side
  connection pooling cut per-RPC connect cost.

## Reliability

- Orphaned `/dev/shm` arenas left by a SIGKILLed coordinator are reaped on the
  next coordinator start (pid-tagged tokens; only ever removes a dead-process
  file), so the arena backing does not leak across restarts.

## Install

```bash
cargo install crayon-rs --version 0.5.0   # Rust runtime + CLI
pip install crayon                         # Python client
```

## Limits

- Task execution is at-least-once; no exactly-once external side effects.
- No coordinator persistence/HA (a `durability_spike` proves it is a cheap
  add-on, deliberately unbuilt — see `docs/feature-prescreen.md`),
  TLS/authentication, hard preemption, or distributed reference counting.
