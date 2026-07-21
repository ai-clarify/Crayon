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
pub mod affinity;
pub mod args;
pub mod common;
pub mod device;
pub mod gcs;
pub mod memory;
pub mod node;
pub mod object_store;
pub mod resources;
pub mod scheduler;
pub mod status;
pub mod versioning;

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
    /// Optional node for distributed actor discovery + cross-node calls.
    node: parking_lot::Mutex<Option<Arc<crate::node::Node>>>,
}

impl Ray {
    /// Initialize the runtime with `num_workers` worker tasks, each with
    /// 1 CPU and 0 GPU (Ray's default per-worker resources).
    ///
    /// Memory management (LRU eviction + disk spilling) is enabled by default
    /// with a limit of 50% of available RAM. This prevents the OOM kills that
    /// plague Ray's plasma store — objects are transparently spilled to a temp
    /// directory when the budget is exceeded. Use [`Ray::init_with_memory`]
    /// to customize the limit or spill directory.
    pub fn init(num_workers: usize) -> Self {
        let max_mem = default_memory_budget();
        let spill_dir = std::env::temp_dir().join(format!("crayon-spill-{}", std::process::id()));
        Self::init_with_memory(num_workers, max_mem, spill_dir)
    }

    /// Initialize the runtime with custom per-worker resources.
    /// Memory management uses the same defaults as [`Ray::init`].
    pub fn init_with_resources(num_workers: usize, per_worker: Resources) -> Self {
        let max_mem = default_memory_budget();
        let spill_dir = std::env::temp_dir().join(format!("crayon-spill-{}", std::process::id()));
        Self::init_with_memory_and_resources(num_workers, max_mem, spill_dir, per_worker)
    }

    /// Initialize with a memory manager for disk spilling.
    pub fn init_with_memory(
        num_workers: usize,
        max_memory_bytes: usize,
        spill_dir: std::path::PathBuf,
    ) -> Self {
        Self::init_with_memory_and_resources(
            num_workers,
            max_memory_bytes,
            spill_dir,
            Resources::new(1.0, 0.0),
        )
    }

    /// Initialize with both memory management and custom per-worker resources.
    pub fn init_with_memory_and_resources(
        num_workers: usize,
        max_memory_bytes: usize,
        spill_dir: std::path::PathBuf,
        per_worker: Resources,
    ) -> Self {
        let mem = crate::memory::MemoryManager::new(max_memory_bytes, spill_dir);
        let store = ObjectStore::new().with_memory(mem);
        let gcs = Gcs::new();
        let pool = WorkerPool::new(num_workers, per_worker, gcs.clone(), store.clone());
        let scheduler = Scheduler::new(pool);
        Ray {
            inner: Arc::new(RayInner {
                store,
                gcs,
                scheduler,
                named_actors: parking_lot::Mutex::new(HashMap::new()),
                node: parking_lot::Mutex::new(None),
            }),
        }
    }

    /// Attach a [`Node`] for distributed actor discovery and cross-node calls.
    /// Must be called before creating actors that should be reachable from
    /// other nodes.
    pub fn attach_node(&self, node: Arc<crate::node::Node>) {
        *self.inner.node.lock() = Some(node);
    }

    /// Store an object in the object store. Returns a typed, refcounted
    /// reference. When the last clone is dropped, the object is evicted.
    pub fn put<T: serde::Serialize + Send + 'static>(&self, value: T) -> ObjectRef<T> {
        let (r, size) = self.inner.store.put(value);
        self.inner.gcs.add_object(crate::gcs::ObjectMeta {
            id: r.id,
            size_bytes: size,
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
        // If a node is attached, the actor lives on this node — register it so
        // incoming ActorCall messages can be routed to it.
        let node = self.inner.node.lock().clone();
        let owner_node = node.as_ref().map(|n| n.id);
        let handle = spawn_actor(
            name,
            state,
            max_restarts,
            self.inner.gcs.clone(),
            self.inner.store.clone(),
            owner_node,
            node.clone(),
        );
        if !name.is_empty() {
            self.inner
                .named_actors
                .lock()
                .insert(name.to_string(), handle.inner.clone());
        }
        if let Some(node) = node {
            node.register_actor(handle.inner.clone());
        }
        handle
    }

    /// Look up a previously registered actor by name. If the actor is not
    /// local, searches the GCS (which is synced from peers) and returns a
    /// remote proxy handle — [`ActorHandle::call_named`] will transparently
    /// forward calls to the actor's owner node.
    ///
    /// The caller must specify the actor's state type `S`; a mismatch will
    /// surface as a panic on the first `call` (the actor task downcasts its
    /// state).
    pub fn get_actor<S: Send + 'static>(&self, name: &str) -> Option<ActorHandle<S>> {
        // 1. Local actor registry.
        if let Some(inner) = self.inner.named_actors.lock().get(name).cloned() {
            return Some(ActorHandle::from_inner(inner));
        }
        // 2. Remote actor discovered via GCS sync. Build a proxy handle whose
        //    `call_named` routes through the local node to the owner node.
        let node = self.inner.node.lock().clone()?;
        let meta = self.inner.gcs.get_actor_by_name(name)?;
        let owner = meta.owner_node?; // local actors are checked above
        let inner = crate::actor::ActorHandleInner::remote_proxy(
            meta.id,
            owner,
            node,
            self.inner.store.clone(),
            self.inner.gcs.clone(),
        );
        Some(ActorHandle::from_inner(inner))
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
        SystemStatus::snapshot(&self.inner.gcs, &self.inner.store)
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

    /// Create a versioned artifact store for RL checkpoints, rollouts, etc.
    /// `keep_last` bounds versions per name (0 = unlimited). Old versions are
    /// evicted automatically — the other half of the OOM story.
    pub fn versioned_store(&self, keep_last: usize) -> crate::versioning::VersionedStore {
        crate::versioning::VersionedStore::new(self.inner.store.clone(), keep_last)
    }

    /// Access the GCS directly (advanced).
    pub fn gcs(&self) -> &Gcs {
        &self.inner.gcs
    }
}

/// Pick a sensible default memory budget: 50% of available RAM, clamped to
/// `[256 MiB, 32 GiB]`. Falls back to 2 GiB if detection fails.
///
/// This is what makes Crayon OOM-safe out of the box — unlike Ray's plasma
/// store, which grows until the OS OOM-kills the process.
fn default_memory_budget() -> usize {
    let total = detect_total_memory_bytes().unwrap_or(2 * 1024 * 1024 * 1024);
    let budget = total / 2;
    budget.clamp(256 * 1024 * 1024, 32 * 1024 * 1024 * 1024)
}

/// Best-effort total system memory detection, no external crates.
fn detect_total_memory_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let kb: usize = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())?;
                    return Some(kb * 1024);
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let out = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        return s.trim().parse::<usize>().ok();
    }
    #[allow(unreachable_code)]
    None
}
