//! Crayon — a Rust reimplementation of Ray's core.
//!
//! Public API:
//! - [`Ray::init`]   — start the runtime
//! - [`Ray::put`]    — store an object, get an [`ObjectRef`]
//! - [`Ray::get`]    — fetch an object by reference
//! - [`Ray::spawn`]  — run a closure as a remote task
//! - [`Ray::create_actor`] — spawn a stateful actor
//! - [`Ray::status`] — snapshot system status
//!
//! ```no_run
//! use crayon::Ray;
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let ray = Ray::init(4);
//! let r = ray.put(42);
//! let v: i32 = ray.get(&r).await?;
//! assert_eq!(v, 42);
//! # Ok(())
//! # }
//! ```

pub mod actor;
pub mod args;
pub mod common;
pub mod gcs;
pub mod memory;
pub mod node;
pub mod object_store;
pub mod resources;
pub mod scheduler;
pub mod status;

use std::collections::HashMap;
use std::sync::Arc;

use crate::actor::{spawn_actor, ActorHandle, ActorHandleInner};
use crate::common::{CrayonError, ObjectID, ObjectRef};
use crate::gcs::Gcs;
use crate::object_store::ObjectStore;
use crate::resources::Resources;
use crate::scheduler::{Scheduler, Task, WorkerPool};
use crate::status::SystemStatus;

/// The Crayon runtime handle. Cheap to clone (Arc).
#[derive(Clone)]
pub struct Ray {
    inner: Arc<RayInner>,
}

struct RayInner {
    store: ObjectStore,
    gcs: Gcs,
    scheduler: Scheduler,
    named_actors: parking_lot::Mutex<HashMap<String, Arc<ActorHandleInner>>>,
}

impl Ray {
    /// Initialize the runtime with `num_workers` worker tasks, each with
    /// 1 CPU and 0 GPU (Ray's default per-worker resources).
    pub fn init(num_workers: usize) -> Self {
        Self::init_with_resources(num_workers, Resources::new(1.0, 0.0))
    }

    /// Initialize the runtime with custom per-worker resources.
    pub fn init_with_resources(num_workers: usize, per_worker: Resources) -> Self {
        let store = ObjectStore::new();
        let gcs = Gcs::new();
        let pool = WorkerPool::new(num_workers, per_worker, gcs.clone(), store.clone());
        let scheduler = Scheduler::new(pool);
        Ray {
            inner: Arc::new(RayInner {
                store,
                gcs,
                scheduler,
                named_actors: parking_lot::Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Initialize with a memory manager for disk spilling.
    pub fn init_with_memory(
        num_workers: usize,
        max_memory_bytes: usize,
        spill_dir: std::path::PathBuf,
    ) -> Self {
        let mem = crate::memory::MemoryManager::new(max_memory_bytes, spill_dir);
        let store = ObjectStore::new().with_memory(mem);
        let gcs = Gcs::new();
        let pool = WorkerPool::new(
            num_workers,
            Resources::new(1.0, 0.0),
            gcs.clone(),
            store.clone(),
        );
        let scheduler = Scheduler::new(pool);
        Ray {
            inner: Arc::new(RayInner {
                store,
                gcs,
                scheduler,
                named_actors: parking_lot::Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Store an object in the object store. Returns a typed, refcounted
    /// reference. When the last clone is dropped, the object is evicted.
    pub fn put<T: serde::Serialize + Send + 'static>(&self, value: T) -> ObjectRef<T> {
        let r = self.inner.store.put(value);
        self.inner.gcs.add_object(crate::gcs::ObjectMeta {
            id: r.id,
            size_bytes: 0, // ponytail: track real size when serializing
            created_at: std::time::Instant::now(),
            owner: None,
        });
        r
    }

    /// Fetch an object by reference. Waits if the object is in-flight.
    pub async fn get<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        r: &ObjectRef<T>,
    ) -> Result<T, CrayonError> {
        self.inner.store.get(r.id).await
    }

    /// Fetch many objects concurrently. Returns results in input order.
    /// Critical for RL: pulling a batch of rollout samples in one fan-out.
    pub async fn get_batch<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        refs: &[ObjectRef<T>],
    ) -> Vec<Result<T, CrayonError>> {
        let ids: Vec<_> = refs.iter().map(|r| r.id).collect();
        self.inner.store.get_batch(&ids).await
    }

    /// Run a closure as a remote task on a worker, using default resources
    /// (1 CPU, 0 GPU). `args` is a tuple of
    /// [`ResolveArg`](crate::args::ResolveArg)s; any [`ObjectRef`] arguments
    /// are automatically fetched before the closure runs (Ray's automatic
    /// dependency resolution). Returns a reference to the result.
    ///
    /// ```ignore
    /// let a = ray.put(1);
    /// let b = ray.put(2);
    /// let r = ray.spawn((a, b), |(a, b): (i32, i32)| a + b);
    /// ```
    pub fn spawn<F, T, Args>(&self, args: Args, func: F) -> ObjectRef<T>
    where
        Args: crate::args::ResolveArgs + Send + Sync + Clone + 'static,
        F: Fn(Args::Output) -> T + Send + Sync + 'static,
        T: serde::Serialize + Send + 'static,
    {
        self.spawn_with_resources(args, Resources::default_task(), func)
    }

    /// Like [`Ray::spawn`] but with explicit resource requirements.
    pub fn spawn_with_resources<F, T, Args>(
        &self,
        args: Args,
        resources: Resources,
        func: F,
    ) -> ObjectRef<T>
    where
        Args: crate::args::ResolveArgs + Send + Sync + Clone + 'static,
        F: Fn(Args::Output) -> T + Send + Sync + 'static,
        T: serde::Serialize + Send + 'static,
    {
        self.spawn_inner(args, resources, 3, func)
    }

    /// Like [`Ray::spawn`] but with a custom max retry count (0 = no retry).
    pub fn spawn_with_retry<F, T, Args>(
        &self,
        args: Args,
        max_retries: u32,
        func: F,
    ) -> ObjectRef<T>
    where
        Args: crate::args::ResolveArgs + Send + Sync + Clone + 'static,
        F: Fn(Args::Output) -> T + Send + Sync + 'static,
        T: serde::Serialize + Send + 'static,
    {
        self.spawn_inner(args, Resources::default_task(), max_retries, func)
    }

    fn spawn_inner<F, T, Args>(
        &self,
        args: Args,
        resources: Resources,
        max_retries: u32,
        func: F,
    ) -> ObjectRef<T>
    where
        Args: crate::args::ResolveArgs + Send + Sync + Clone + 'static,
        F: Fn(Args::Output) -> T + Send + Sync + 'static,
        T: serde::Serialize + Send + 'static,
    {
        let id = crate::common::TaskID::new();
        let output_id = ObjectID::new();
        let args = Arc::new(args);
        let func = Arc::new(func);
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = Task {
            id,
            output_id,
            resources,
            max_retries,
            retries: 0,
            priority: 0,
            cancelled,
            func: Arc::new(move |store: ObjectStore| {
                let args = args.clone();
                let func = func.clone();
                Box::pin(async move {
                    let resolved = (*args).clone().resolve(&store).await?;
                    let result = func(resolved);
                    bincode::serialize(&result).map_err(|e| CrayonError::Serialize(e.to_string()))
                })
            }),
        };
        let r = self.inner.store.reserve_ref(output_id);
        if let Err(e) = self.inner.scheduler.schedule(task) {
            // Scheduling failed (e.g. task resources exceed any worker's total).
            // Store the error so the caller gets it on `get` instead of a
            // timeout.
            self.inner.store.put_error(output_id, e);
            self.inner
                .gcs
                .set_task_state(id, crate::common::TaskState::Failed, Some(output_id));
        }
        r
    }

    /// Spawn a stateful actor with the given initial state. If `name` is
    /// non-empty, the actor is registered and can later be retrieved with
    /// [`Ray::get_actor`]. The actor does not restart on panic by default.
    pub fn create_actor<S: Send + Clone + 'static>(&self, name: &str, state: S) -> ActorHandle<S> {
        self.create_actor_with_restarts(name, state, 0)
    }

    /// Like [`Ray::create_actor`] but with a custom `max_restarts` count.
    /// On panic, the actor is reset to its initial state and continues
    /// (up to `max_restarts` times).
    pub fn create_actor_with_restarts<S: Send + Clone + 'static>(
        &self,
        name: &str,
        state: S,
        max_restarts: u32,
    ) -> ActorHandle<S> {
        let handle = spawn_actor(
            name,
            state,
            max_restarts,
            self.inner.gcs.clone(),
            self.inner.store.clone(),
        );
        if !name.is_empty() {
            self.inner
                .named_actors
                .lock()
                .insert(name.to_string(), handle.inner.clone());
        }
        handle
    }

    /// Look up a previously registered actor by name. The caller must specify
    /// the actor's state type `S`; a mismatch will surface as a panic on the
    /// first `call` (the actor task downcasts its state).
    pub fn get_actor<S: Send + 'static>(&self, name: &str) -> Option<ActorHandle<S>> {
        self.inner
            .named_actors
            .lock()
            .get(name)
            .cloned()
            .map(ActorHandle::from_inner)
    }

    /// Remove a named actor from the registry and kill it. Returns `true` if
    /// the actor existed. This allows long-running training jobs to clean up
    /// actors that are no longer needed (Ray #24711 analog).
    pub fn remove_actor(&self, name: &str) -> bool {
        if let Some(inner) = self.inner.named_actors.lock().remove(name) {
            inner.shutdown();
            inner
                .gcs()
                .set_actor_state(inner.id, crate::common::ActorState::Dead);
            true
        } else {
            false
        }
    }

    /// Snapshot the current system status.
    pub fn status(&self) -> SystemStatus {
        SystemStatus::snapshot(&self.inner.gcs)
    }

    /// Cancel a task by id. The task will be skipped if pending, or its result
    /// discarded if already running. Returns `false` if the id is unknown.
    ///
    /// Critical for RL: stale rollouts from an outdated policy must be killed
    /// to free resources for the new policy.
    pub fn cancel(&self, task_id: crate::common::TaskID) -> bool {
        self.inner.scheduler.cancel_task(task_id)
    }

    /// Access the object store directly (advanced).
    pub fn store(&self) -> &ObjectStore {
        &self.inner.store
    }

    /// Access the GCS directly (advanced).
    pub fn gcs(&self) -> &Gcs {
        &self.inner.gcs
    }
}
