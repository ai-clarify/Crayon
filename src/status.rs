//! System status reporting — Crayon's analog of `ray status` / the dashboard.
//!
//! Aggregates metadata from the GCS into a human-readable snapshot: object
//! count, actor states, task throughput, worker utilization.

use serde::Serialize;

use crate::gcs::Gcs;

#[derive(Debug, Serialize)]
pub struct SystemStatus {
    pub objects: usize,
    pub actors: Vec<ActorStatus>,
    pub tasks_total: usize,
    pub tasks_finished: usize,
    pub tasks_failed: usize,
    pub tasks_pending: usize,
    pub tasks_running: usize,
    pub workers: Vec<WorkerStatus>,
    pub worker_utilization: f64,
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
    pub fn snapshot(gcs: &Gcs) -> Self {
        let objects = gcs.objects().len();
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
        let tasks_finished = tasks
            .iter()
            .filter(|t| matches!(t.state, crate::common::TaskState::Finished))
            .count();
        let tasks_failed = tasks
            .iter()
            .filter(|t| matches!(t.state, crate::common::TaskState::Failed))
            .count();
        let tasks_pending = tasks
            .iter()
            .filter(|t| matches!(t.state, crate::common::TaskState::Pending))
            .count();
        let tasks_running = tasks
            .iter()
            .filter(|t| matches!(t.state, crate::common::TaskState::Running))
            .count();

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
            tasks_pending,
            tasks_running,
            workers,
            worker_utilization,
        }
    }

    /// Pretty-print the status to a string (like `ray status`).
    pub fn pretty(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Crayon Status ===\n");
        s.push_str(&format!("Objects: {}\n", self.objects));
        s.push_str(&format!(
            "Tasks: {} total, {} finished, {} failed, {} running, {} pending\n",
            self.tasks_total,
            self.tasks_finished,
            self.tasks_failed,
            self.tasks_running,
            self.tasks_pending
        ));
        s.push_str(&format!(
            "Workers: {} (utilization {:.0}%)\n",
            self.workers.len(),
            self.worker_utilization * 100.0
        ));
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
