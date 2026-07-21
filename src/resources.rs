//! Resource tracking for workers and tasks — Crayon's analog of Ray's
//! resource model (CPU/GPU accounting).
//!
//! Ray allows tasks and actors to declare resource requirements
//! (`num_cpus`, `num_gpus`). The raylet only schedules a task onto a worker
//! that has enough free resources, and accounts for them while the task runs.

use parking_lot::Mutex;

/// A set of compute resources. Uses `f64` so tasks can request fractional
/// resources (e.g. `0.5` CPU), matching Ray's model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resources {
    pub cpu: f64,
    pub gpu: f64,
}

impl Resources {
    pub fn new(cpu: f64, gpu: f64) -> Self {
        Resources { cpu, gpu }
    }

    /// Default task resource request: 1 CPU, 0 GPU (matches Ray's default).
    pub fn default_task() -> Self {
        Resources { cpu: 1.0, gpu: 0.0 }
    }

    /// Can `self` cover a request for `needed` resources?
    pub fn can_fit(&self, needed: &Resources) -> bool {
        self.cpu >= needed.cpu && self.gpu >= needed.gpu
    }

    pub fn subtract(&mut self, other: &Resources) {
        self.cpu -= other.cpu;
        self.gpu -= other.gpu;
    }

    pub fn add(&mut self, other: &Resources) {
        self.cpu += other.cpu;
        self.gpu += other.gpu;
    }
}

/// Tracks available resources for a set of workers. Thread-safe.
pub struct ResourceTracker {
    inner: Mutex<Vec<WorkerResources>>,
}

struct WorkerResources {
    available: Resources,
    total: Resources,
}

impl ResourceTracker {
    pub fn new(num_workers: usize, per_worker: Resources) -> Self {
        let workers = (0..num_workers)
            .map(|_| WorkerResources {
                available: per_worker,
                total: per_worker,
            })
            .collect();
        ResourceTracker {
            inner: Mutex::new(workers),
        }
    }

    /// Try to reserve `needed` resources on some worker. Returns the worker id
    /// if one had enough free resources, else `None`.
    ///
    /// If the task needs no GPU, prefers workers with no GPU total to avoid
    /// wasting expensive GPU nodes on CPU-only work (#47866 analog).
    pub fn try_acquire(&self, needed: &Resources) -> Option<usize> {
        let mut workers = self.inner.lock();
        // First pass: if no GPU needed, prefer CPU-only workers.
        if needed.gpu == 0.0 {
            for (i, w) in workers.iter_mut().enumerate() {
                if w.total.gpu == 0.0 && w.available.can_fit(needed) {
                    w.available.subtract(needed);
                    return Some(i);
                }
            }
        }
        // Second pass: any worker that fits.
        for (i, w) in workers.iter_mut().enumerate() {
            if w.available.can_fit(needed) {
                w.available.subtract(needed);
                return Some(i);
            }
        }
        None
    }

    /// Release previously-acquired resources back to a worker.
    pub fn release(&self, worker_id: usize, resources: &Resources) {
        let mut workers = self.inner.lock();
        if let Some(w) = workers.get_mut(worker_id) {
            w.available.add(resources);
        }
    }

    /// Snapshot of available resources per worker (for status reporting).
    pub fn snapshot(&self) -> Vec<Resources> {
        self.inner.lock().iter().map(|w| w.available).collect()
    }

    /// Returns true if at least one worker has enough *total* resources to
    /// ever fit this request. If false, the task can never be scheduled and
    /// would deadlock — the caller should fail fast instead of queueing.
    pub fn can_any_worker_fit(&self, needed: &Resources) -> bool {
        self.inner.lock().iter().any(|w| w.total.can_fit(needed))
    }
}
