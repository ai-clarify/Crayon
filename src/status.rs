//! System status reporting — Crayon's analog of `ray status` / the dashboard.
//!
//! Aggregates metadata from the GCS into a human-readable snapshot: object
//! count, actor states, task throughput, worker utilization.

use serde::Serialize;

use crate::gcs::Gcs;
use crate::object_store::ObjectStore;

#[derive(Debug, Serialize)]
pub struct SystemStatus {
    pub objects: usize,
    pub actors: Vec<ActorStatus>,
    pub tasks_total: usize,
    pub tasks_finished: usize,
    pub tasks_failed: usize,
    pub tasks_cancelled: usize,
    pub tasks_pending: usize,
    pub tasks_running: usize,
    pub workers: Vec<WorkerStatus>,
    pub worker_utilization: f64,
    /// Bytes currently held in the in-memory object store.
    pub memory_used_bytes: usize,
    /// Memory budget; objects are spilled to disk beyond this. 0 = unlimited.
    pub memory_limit_bytes: usize,
    pub spill_failures: usize,
}

#[derive(Debug, Serialize)]
pub struct ActorStatus {
    pub id: String,
    pub name: String,
    pub state: String,
    pub pending: usize,
    pub completed: usize,
}

#[derive(Debug, Serialize)]
pub struct WorkerStatus {
    pub id: usize,
    pub busy: bool,
    pub tasks_done: usize,
}

impl SystemStatus {
    pub fn snapshot(gcs: &Gcs, store: &ObjectStore) -> Self {
        let objects = store.len();
        let (memory_used_bytes, memory_limit_bytes) = store.memory_stats();
        let spill_failures = store.spill_failures();
        let actors = gcs
            .actors()
            .into_iter()
            .map(|a| ActorStatus {
                id: format!("{}", a.id),
                name: a.name,
                state: format!("{:?}", a.state),
                pending: a.pending_tasks,
                completed: a.completed_tasks,
            })
            .collect();

        let tasks = gcs.tasks();
        let tasks_total = tasks.len();
        let mut tasks_finished = 0;
        let mut tasks_failed = 0;
        let mut tasks_cancelled = 0;
        let mut tasks_pending = 0;
        let mut tasks_running = 0;
        for t in &tasks {
            match t.state {
                crate::common::TaskState::Finished => tasks_finished += 1,
                crate::common::TaskState::Failed => tasks_failed += 1,
                crate::common::TaskState::Cancelled => tasks_cancelled += 1,
                crate::common::TaskState::Pending => tasks_pending += 1,
                crate::common::TaskState::Running => tasks_running += 1,
            }
        }

        let workers: Vec<WorkerStatus> = gcs
            .workers()
            .into_iter()
            .map(|w| WorkerStatus {
                id: w.id,
                busy: w.busy,
                tasks_done: w.tasks_done,
            })
            .collect();

        let busy = workers.iter().filter(|w| w.busy).count();
        let worker_utilization = if workers.is_empty() {
            0.0
        } else {
            busy as f64 / workers.len() as f64
        };

        SystemStatus {
            objects,
            actors,
            tasks_total,
            tasks_finished,
            tasks_failed,
            tasks_cancelled,
            tasks_pending,
            tasks_running,
            workers,
            worker_utilization,
            memory_used_bytes,
            memory_limit_bytes,
            spill_failures,
        }
    }

    /// Pretty-print the status to a string (like `ray status`).
    pub fn pretty(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Crayon Status ===\n");
        s.push_str(&format!("Objects: {}\n", self.objects));
        s.push_str(&format!(
            "Tasks: {} total, {} finished, {} failed, {} cancelled, {} running, {} pending\n",
            self.tasks_total,
            self.tasks_finished,
            self.tasks_failed,
            self.tasks_cancelled,
            self.tasks_running,
            self.tasks_pending
        ));
        s.push_str(&format!(
            "Workers: {} (utilization {:.0}%)\n",
            self.workers.len(),
            self.worker_utilization * 100.0
        ));
        if self.memory_limit_bytes > 0 {
            let pct = self.memory_used_bytes as f64 / self.memory_limit_bytes as f64 * 100.0;
            s.push_str(&format!(
                "Memory: {} / {} ({:.0}%)\n",
                format_bytes(self.memory_used_bytes),
                format_bytes(self.memory_limit_bytes),
                pct
            ));
        }
        if !self.actors.is_empty() {
            s.push_str("Actors:\n");
            for a in &self.actors {
                s.push_str(&format!(
                    "  {} [{}] state={} pending={} done={}\n",
                    a.name, a.id, a.state, a.pending, a.completed
                ));
            }
        }
        s
    }
}

/// Human-readable byte size, e.g. "1.5 GiB".
fn format_bytes(b: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{:.1} {}", v, UNITS[i])
}
