//! The Global Control Store (GCS) — Crayon's analog of Ray's GCS.
//!
//! Ray's GCS is a Redis-backed metadata service that records every object,
//! actor, task, and node in the cluster. It is the source of truth for the
//! control plane. Here we keep it in-process as a set of guarded maps; a
//! distributed build would swap the backend for Redis/etcd without changing
//! the API.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use crate::common::{ActorID, ActorState, ObjectID, TaskID, TaskState};

#[derive(Debug, Clone)]
pub struct ObjectMeta {
    pub id: ObjectID,
    pub size_bytes: usize,
    pub created_at: Instant,
    pub owner: Option<TaskID>,
    pub owner_node: Option<crate::node::NodeID>,
}

#[derive(Debug, Clone)]
pub struct ActorMeta {
    pub id: ActorID,
    pub name: String,
    pub state: ActorState,
    /// The node that hosts this actor (for cross-node discovery/calls).
    pub owner_node: Option<crate::node::NodeID>,
    pub created_at: Instant,
    pub pending_tasks: usize,
    pub completed_tasks: usize,
}

#[derive(Debug, Clone)]
pub struct TaskMeta {
    pub id: TaskID,
    pub state: TaskState,
    pub created_at: Instant,
    pub finished_at: Option<Instant>,
    pub output: Option<ObjectID>,
}

#[derive(Debug, Clone)]
pub struct WorkerMeta {
    pub id: usize,
    pub busy: bool,
    pub tasks_done: usize,
}

#[derive(Default)]
struct GcsInner {
    objects: HashMap<ObjectID, ObjectMeta>,
    actors: HashMap<ActorID, ActorMeta>,
    tasks: HashMap<TaskID, TaskMeta>,
    workers: HashMap<usize, WorkerMeta>,
}

/// In-process GCS. Cheap to clone (Arc).
#[derive(Clone, Default)]
pub struct Gcs {
    inner: Arc<Mutex<GcsInner>>,
}

impl Gcs {
    pub fn new() -> Self {
        Gcs {
            inner: Arc::new(Mutex::new(GcsInner::default())),
        }
    }

    // ---- objects ----
    pub fn add_object(&self, meta: ObjectMeta) {
        self.inner.lock().objects.insert(meta.id, meta);
    }

    pub fn get_object(&self, id: ObjectID) -> Option<ObjectMeta> {
        self.inner.lock().objects.get(&id).cloned()
    }

    pub fn remove_object(&self, id: ObjectID) {
        self.inner.lock().objects.remove(&id);
    }

    pub fn objects(&self) -> Vec<ObjectMeta> {
        self.inner.lock().objects.values().cloned().collect()
    }

    // ---- actors ----
    pub fn add_actor(&self, meta: ActorMeta) {
        self.inner.lock().actors.insert(meta.id, meta);
    }

    pub fn get_actor(&self, id: ActorID) -> Option<ActorMeta> {
        self.inner.lock().actors.get(&id).cloned()
    }

    /// Look up an actor by name. Avoids cloning all actor metas (unlike
    /// `actors().iter().find(...)`) — scans the map under the lock and returns
    /// only the match. Used by `Ray::get_actor` for remote actor discovery.
    pub fn get_actor_by_name(&self, name: &str) -> Option<ActorMeta> {
        self.inner
            .lock()
            .actors
            .values()
            .find(|a| a.name == name)
            .cloned()
    }

    pub fn set_actor_state(&self, id: ActorID, state: ActorState) {
        if let Some(a) = self.inner.lock().actors.get_mut(&id) {
            a.state = state;
        }
    }

    pub fn record_actor_task(&self, id: ActorID, completed: bool) {
        if let Some(a) = self.inner.lock().actors.get_mut(&id) {
            if completed {
                a.completed_tasks += 1;
                a.pending_tasks = a.pending_tasks.saturating_sub(1);
            } else {
                a.pending_tasks += 1;
            }
        }
    }

    pub fn actors(&self) -> Vec<ActorMeta> {
        self.inner.lock().actors.values().cloned().collect()
    }

    // ---- tasks ----
    pub fn add_task(&self, meta: TaskMeta) {
        let mut inner = self.inner.lock();
        inner.tasks.insert(meta.id, meta);
        // Prevent unbounded growth (#52081 analog): if too many tasks
        // accumulate, prune completed ones. Keeps pending/running tasks.
        const MAX_TASKS: usize = 100_000;
        if inner.tasks.len() > MAX_TASKS {
            inner
                .tasks
                .retain(|_, t| t.state == TaskState::Pending || t.state == TaskState::Running);
        }
    }

    pub fn set_task_state(&self, id: TaskID, state: TaskState, output: Option<ObjectID>) {
        if let Some(t) = self.inner.lock().tasks.get_mut(&id) {
            t.state = state;
            if matches!(
                state,
                TaskState::Finished | TaskState::Failed | TaskState::Cancelled
            ) {
                t.finished_at = Some(Instant::now());
            }
            if let Some(o) = output {
                t.output = Some(o);
            }
        }
    }

    pub fn tasks(&self) -> Vec<TaskMeta> {
        self.inner.lock().tasks.values().cloned().collect()
    }

    // ---- workers ----
    pub fn add_worker(&self, id: usize) {
        self.inner.lock().workers.insert(
            id,
            WorkerMeta {
                id,
                busy: false,
                tasks_done: 0,
            },
        );
    }

    pub fn set_worker_busy(&self, id: usize, busy: bool) {
        if let Some(w) = self.inner.lock().workers.get_mut(&id) {
            w.busy = busy;
            if !busy {
                w.tasks_done += 1;
            }
        }
    }

    pub fn workers(&self) -> Vec<WorkerMeta> {
        self.inner.lock().workers.values().cloned().collect()
    }
}
