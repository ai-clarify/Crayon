//! Worker pool and task scheduler.

use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use tokio::sync::{mpsc, Notify};

use crate::common::{CrayonError, ObjectID, TaskID, TaskState};
use crate::gcs::Gcs;
use crate::object_store::{ObjectStore, ProducerLease};
use crate::resources::{ResourceTracker, Resources};

pub type TaskFunc =
    Arc<dyn Fn(ObjectStore) -> BoxFuture<'static, Result<Vec<u8>, CrayonError>> + Send + Sync>;

pub(crate) struct TaskControl {
    cancelled: AtomicBool,
    wake: Notify,
    output_id: ObjectID,
    lease: Arc<ProducerLease>,
}

impl TaskControl {
    fn cancel(&self, id: TaskID) {
        self.cancelled.store(true, Ordering::Release);
        self.lease.fail(CrayonError::TaskCancelled(id));
        self.wake.notify_waiters();
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

pub struct Task {
    pub id: TaskID,
    pub output_id: ObjectID,
    pub resources: Resources,
    pub max_retries: u32,
    pub retries: u32,
    pub priority: u8,
    pub(crate) control: Arc<TaskControl>,
    pub func: TaskFunc,
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

pub struct WorkerPool {
    worker_tx: Vec<mpsc::UnboundedSender<Task>>,
    pending: Arc<Mutex<BinaryHeap<Task>>>,
    gcs: Gcs,
    tracker: Arc<ResourceTracker>,
    num_workers: usize,
    _done_tx: mpsc::UnboundedSender<()>,
    controls: Arc<Mutex<HashMap<TaskID, Arc<TaskControl>>>>,
}

impl WorkerPool {
    pub fn new(
        num_workers: usize,
        per_worker: Resources,
        gcs: Gcs,
        store: ObjectStore,
    ) -> Arc<Self> {
        let tracker = Arc::new(ResourceTracker::new(num_workers, per_worker));
        let mut worker_tx = Vec::with_capacity(num_workers);
        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<()>();
        let pending = Arc::new(Mutex::new(BinaryHeap::new()));
        let controls = Arc::new(Mutex::new(HashMap::new()));

        for worker_id in 0..num_workers {
            gcs.add_worker(worker_id);
            let (tx, mut rx) = mpsc::unbounded_channel::<Task>();
            worker_tx.push(tx);
            let gcs = gcs.clone();
            let store = store.clone();
            let tracker = tracker.clone();
            let done_tx = done_tx.clone();
            let pending = pending.clone();
            let controls = controls.clone();
            tokio::spawn(async move {
                while let Some(mut task) = rx.recv().await {
                    let resources = task.resources;
                    gcs.set_worker_busy(worker_id, true);
                    if task.control.is_cancelled() {
                        terminal(&gcs, &controls, task.id, TaskState::Cancelled);
                        finish_worker(&gcs, &tracker, &done_tx, worker_id, &resources);
                        continue;
                    }

                    gcs.set_task_state(task.id, TaskState::Running, None);
                    crate::device::set_current_device(tracker.gpu_device(worker_id));
                    use futures::FutureExt;
                    let future = std::panic::AssertUnwindSafe((task.func)(store.clone()));
                    let outcome = match future.catch_unwind().await {
                        Ok(result) => result,
                        Err(panic) => Err(crate::common::panic_to_error(panic, "task panicked")),
                    };

                    if task.control.is_cancelled() {
                        terminal(&gcs, &controls, task.id, TaskState::Cancelled);
                    } else {
                        match outcome {
                            Ok(bytes) => {
                                task.control.lease.publish(bytes, None);
                                terminal(&gcs, &controls, task.id, TaskState::Finished);
                            }
                            Err(error) if task.retries < task.max_retries => {
                                task.retries += 1;
                                gcs.set_task_state(task.id, TaskState::Pending, None);
                                let delay = std::time::Duration::from_millis(
                                    100 * (1 << task.retries.min(6)),
                                );
                                let pending = pending.clone();
                                let done = done_tx.clone();
                                tokio::spawn(async move {
                                    tokio::select! {
                                        _ = tokio::time::sleep(delay) => {
                                            if !task.control.is_cancelled() {
                                                pending.lock().push(task);
                                                let _ = done.send(());
                                            }
                                        }
                                        _ = task.control.wake.notified() => {}
                                    }
                                });
                            }
                            Err(error) => {
                                task.control.lease.fail(error);
                                terminal(&gcs, &controls, task.id, TaskState::Failed);
                            }
                        }
                    }
                    finish_worker(&gcs, &tracker, &done_tx, worker_id, &resources);
                }
            });
        }

        let pool = Arc::new(Self {
            worker_tx,
            pending,
            gcs,
            tracker,
            num_workers,
            _done_tx: done_tx,
            controls,
        });
        let pump = pool.clone();
        tokio::spawn(async move {
            while done_rx.recv().await.is_some() {
                pump.pump_pending();
            }
        });
        pool
    }

    pub(crate) fn new_control(output_id: ObjectID, lease: Arc<ProducerLease>) -> Arc<TaskControl> {
        Arc::new(TaskControl {
            cancelled: AtomicBool::new(false),
            wake: Notify::new(),
            output_id,
            lease,
        })
    }

    pub fn submit(self: &Arc<Self>, task: Task) -> Result<(), CrayonError> {
        if !self.tracker.can_any_worker_fit(&task.resources) {
            return Err(CrayonError::TaskFailed(format!(
                "task requires {:?} but no worker has enough total resources",
                task.resources
            )));
        }
        self.controls.lock().insert(task.id, task.control.clone());
        self.gcs.add_task(crate::gcs::TaskMeta {
            id: task.id,
            state: TaskState::Pending,
            created_at: std::time::Instant::now(),
            finished_at: None,
            output: Some(task.output_id),
        });
        self.dispatch(task);
        Ok(())
    }

    pub fn cancel_task(&self, id: TaskID) -> bool {
        let Some(control) = self.controls.lock().get(&id).cloned() else {
            return false;
        };
        control.cancel(id);
        self.remove_pending(id);
        self.gcs
            .set_task_state(id, TaskState::Cancelled, Some(control.output_id));
        true
    }

    fn remove_pending(&self, id: TaskID) {
        let mut pending = self.pending.lock();
        let mut tasks: Vec<_> = pending.drain().collect();
        tasks.retain(|task| task.id != id);
        pending.extend(tasks);
    }

    fn dispatch(self: &Arc<Self>, task: Task) {
        if task.control.is_cancelled() {
            terminal(&self.gcs, &self.controls, task.id, TaskState::Cancelled);
            return;
        }
        if let Some(worker_id) = self.tracker.try_acquire(&task.resources) {
            if self.worker_tx[worker_id].send(task).is_err() {
                self.tracker.release(worker_id, &Resources::default_task());
            }
        } else {
            self.pending.lock().push(task);
        }
    }

    pub fn pump_pending(self: &Arc<Self>) {
        let tasks: Vec<_> = std::iter::from_fn(|| self.pending.lock().pop()).collect();
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

fn terminal(
    gcs: &Gcs,
    controls: &Mutex<HashMap<TaskID, Arc<TaskControl>>>,
    id: TaskID,
    state: TaskState,
) {
    gcs.set_task_state(id, state, None);
    controls.lock().remove(&id);
}

fn finish_worker(
    gcs: &Gcs,
    tracker: &ResourceTracker,
    done: &mpsc::UnboundedSender<()>,
    worker_id: usize,
    resources: &Resources,
) {
    gcs.set_worker_busy(worker_id, false);
    tracker.release(worker_id, resources);
    let _ = done.send(());
}

pub struct Scheduler {
    pool: Arc<WorkerPool>,
}

impl Scheduler {
    pub fn new(pool: Arc<WorkerPool>) -> Self {
        Self { pool }
    }

    pub fn schedule(&self, task: Task) -> Result<(), CrayonError> {
        self.pool.submit(task)
    }

    pub fn cancel_task(&self, id: TaskID) -> bool {
        self.pool.cancel_task(id)
    }

    pub fn pump_pending(&self) {
        self.pool.pump_pending();
    }
}
