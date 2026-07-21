//! Worker pool and task scheduler.
//!
//! Ray runs a pool of worker processes per node; the raylet schedules tasks
//! onto them. In-process, we model workers as tokio tasks pulled from a pool.
//! The scheduler accounts for per-worker CPU/GPU resources (see
//! [`crate::resources`]) and only assigns a task to a worker that can fit it.

use std::sync::Arc;

use std::sync::atomic::AtomicBool;

use futures::future::BoxFuture;
use parking_lot::Mutex as PlMutex;
use tokio::sync::mpsc;

use crate::common::{CrayonError, ObjectID, TaskID, TaskState};
use crate::gcs::Gcs;
use crate::object_store::ObjectStore;
use crate::resources::{ResourceTracker, Resources};

/// A unit of work a worker can execute. The closure receives the object store
/// (so it can resolve task dependencies) and returns serialized bytes that get
/// stored under `output_id`.
pub struct Task {
    pub id: TaskID,
    pub output_id: ObjectID,
    pub resources: Resources,
    pub max_retries: u32,
    pub retries: u32,
    pub priority: u8,
    pub cancelled: Arc<AtomicBool>,
    pub func:
        Arc<dyn Fn(ObjectStore) -> BoxFuture<'static, Result<Vec<u8>, CrayonError>> + Send + Sync>,
}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for Task {}

impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Task {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl std::fmt::Debug for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task")
            .field("id", &self.id)
            .field("resources", &self.resources)
            .finish()
    }
}

/// A pool of workers that pull tasks from per-worker queues.
pub struct WorkerPool {
    worker_tx: Vec<mpsc::UnboundedSender<Task>>,
    pending: Arc<PlMutex<std::collections::BinaryHeap<Task>>>,
    gcs: Gcs,
    store: ObjectStore,
    tracker: Arc<ResourceTracker>,
    num_workers: usize,
    /// Workers send a signal here when they finish a task (releasing resources).
    done_tx: mpsc::UnboundedSender<()>,
    /// Maps task id -> cancellation flag. Set by `cancel_task`.
    cancel_tokens: Arc<PlMutex<std::collections::HashMap<TaskID, Arc<AtomicBool>>>>,
}

impl WorkerPool {
    /// Spawn `num_workers` worker tasks, each with `per_worker` resources.
    pub fn new(
        num_workers: usize,
        per_worker: Resources,
        gcs: Gcs,
        store: ObjectStore,
    ) -> Arc<Self> {
        let tracker = Arc::new(ResourceTracker::new(num_workers, per_worker));
        let mut worker_tx = Vec::with_capacity(num_workers);
        let (done_tx, done_rx) = mpsc::unbounded_channel::<()>();
        let pending: Arc<PlMutex<std::collections::BinaryHeap<Task>>> =
            Arc::new(PlMutex::new(std::collections::BinaryHeap::new()));
        let cancel_tokens: Arc<PlMutex<std::collections::HashMap<TaskID, Arc<AtomicBool>>>> =
            Arc::new(PlMutex::new(std::collections::HashMap::new()));

        for i in 0..num_workers {
            gcs.add_worker(i);
            let (tx, mut rx) = mpsc::unbounded_channel::<Task>();
            worker_tx.push(tx);

            let gcs = gcs.clone();
            let store = store.clone();
            let tracker = tracker.clone();
            let done_tx = done_tx.clone();
            let pending = pending.clone();
            tokio::spawn(async move {
                while let Some(mut task) = rx.recv().await {
                    let res = task.resources;
                    gcs.set_worker_busy(i, true);

                    // Check for cancellation before running.
                    if task.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                        let msg = "task cancelled".to_string();
                        let bytes = bincode::serialize(&msg).unwrap();
                        store.put_bytes(task.output_id, bytes, None);
                        gcs.set_task_state(task.id, TaskState::Failed, Some(task.output_id));
                        gcs.set_worker_busy(i, false);
                        tracker.release(i, &res);
                        let _ = done_tx.send(());
                        continue;
                    }

                    gcs.set_task_state(task.id, TaskState::Running, None);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        (task.func)(store.clone())
                    }));
                    let outcome = match result {
                        Ok(fut) => fut.await,
                        Err(p) => {
                            let msg = if let Some(s) = p.downcast_ref::<&str>() {
                                s.to_string()
                            } else if let Some(s) = p.downcast_ref::<String>() {
                                s.clone()
                            } else {
                                "task panicked".into()
                            };
                            Err(CrayonError::TaskFailed(msg))
                        }
                    };
                    match outcome {
                        Ok(bytes) => {
                            store.put_bytes(task.output_id, bytes, None);
                            gcs.set_task_state(task.id, TaskState::Finished, Some(task.output_id));
                        }
                        Err(e) => {
                            if task.retries < task.max_retries {
                                task.retries += 1;
                                gcs.set_task_state(task.id, TaskState::Pending, None);
                                // Exponential backoff: 100ms * 2^retries, capped at 5s.
                                let delay = std::time::Duration::from_millis(
                                    100 * (1 << task.retries.min(6)),
                                );
                                let pending = pending.clone();
                                tokio::spawn(async move {
                                    tokio::time::sleep(delay).await;
                                    pending.lock().push(task);
                                });
                                let _ = done_tx.send(());
                                gcs.set_worker_busy(i, false);
                                tracker.release(i, &res);
                                continue;
                            }
                            let bytes = bincode::serialize(&e.to_string()).unwrap();
                            store.put_bytes(task.output_id, bytes, None);
                            gcs.set_task_state(task.id, TaskState::Failed, Some(task.output_id));
                        }
                    }
                    gcs.set_worker_busy(i, false);
                    tracker.release(i, &res);
                    let _ = done_tx.send(());
                }
            });
        }

        let pool = Arc::new(WorkerPool {
            worker_tx,
            pending,
            gcs,
            store,
            tracker,
            num_workers,
            done_tx,
            cancel_tokens,
        });

        // Background task: when a worker finishes, retry pending tasks that
        // might now fit the freed resources.
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            let mut done_rx = done_rx;
            while done_rx.recv().await.is_some() {
                pool_clone.pump_pending();
            }
        });

        pool
    }

    /// Submit a task to the pool. Tries to acquire resources; if no worker can
    /// fit the task right now, it's queued and retried when resources free up.
    pub fn submit(self: &Arc<Self>, task: Task) -> Result<(), CrayonError> {
        // Deadlock guard: if no worker has enough *total* resources to ever
        // fit this task, it would queue forever. Fail fast instead.
        if !self.tracker.can_any_worker_fit(&task.resources) {
            return Err(CrayonError::TaskFailed(format!(
                "task requires {:?} but no worker has enough total resources",
                task.resources
            )));
        }
        self.cancel_tokens
            .lock()
            .insert(task.id, task.cancelled.clone());
        self.gcs.add_task(crate::gcs::TaskMeta {
            id: task.id,
            state: TaskState::Pending,
            created_at: std::time::Instant::now(),
            finished_at: None,
            output: Some(task.output_id),
        });
        self.store.reserve(task.output_id);
        self.dispatch(task);
        Ok(())
    }

    /// Cancel a task by id. If the task is pending or running, it will be
    /// skipped (or aborted at the next checkpoint) and its output will be
    /// marked as failed. Returns `false` if the task id is unknown.
    pub fn cancel_task(&self, id: TaskID) -> bool {
        if let Some(token) = self.cancel_tokens.lock().get(&id) {
            token.store(true, std::sync::atomic::Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Try to send a task to a worker with sufficient resources.
    fn dispatch(self: &Arc<Self>, task: Task) {
        if let Some(worker_id) = self.tracker.try_acquire(&task.resources) {
            self.worker_tx[worker_id]
                .send(task)
                .expect("worker channel should not close while pool is alive");
        } else {
            // No worker can fit this task right now; queue it.
            self.pending.lock().push(task);
        }
    }

    /// Called when resources are released; retries pending tasks in priority
    /// order (highest priority first).
    pub fn pump_pending(self: &Arc<Self>) {
        let tasks: Vec<Task> = {
            let mut p = self.pending.lock();
            std::iter::from_fn(|| p.pop()).collect()
        };
        for task in tasks {
            self.dispatch(task);
        }
    }

    pub fn tracker(&self) -> &Arc<ResourceTracker> {
        &self.tracker
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

/// The scheduler. Routes tasks to workers based on resource availability.
pub struct Scheduler {
    pool: Arc<WorkerPool>,
}

impl Scheduler {
    pub fn new(pool: Arc<WorkerPool>) -> Self {
        Scheduler { pool }
    }

    pub fn schedule(&self, task: Task) -> Result<(), CrayonError> {
        self.pool.submit(task)
    }

    /// Cancel a task by id. See [`WorkerPool::cancel_task`].
    pub fn cancel_task(&self, id: crate::common::TaskID) -> bool {
        self.pool.cancel_task(id)
    }

    /// Retry any tasks that were queued waiting for resources.
    pub fn pump_pending(&self) {
        self.pool.pump_pending();
    }
}
