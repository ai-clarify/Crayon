//! The object store — Crayon's analog of Ray's Plasma store.
//!
//! Objects are stored as `Vec<u8>` (bincode-serialized) so they can be shipped
//! across nodes. A [`MemoryManager`](crate::memory::MemoryManager) tracks
//! per-object sizes and evicts LRU objects to disk when memory exceeds a
//! threshold. Evicted objects are transparently reloaded on next access.
//!
//! If a [`Node`](crate::node::Node) is attached, `get` will fetch objects
//! from remote peers when they're not found locally.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;
use tokio::sync::Notify;

use crate::common::{CrayonError, ObjectID, ObjectRef};
use crate::memory::MemoryManager;

/// Boxed future returned by [`RemoteFetcher::fetch_remote`].
pub type RemoteFetchFuture = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>>
            + Send,
    >,
>;

/// Trait for fetching objects from remote nodes. Implemented by [`Node`].
pub trait RemoteFetcher: Send + Sync {
    fn fetch_remote(&self, id: ObjectID) -> RemoteFetchFuture;
}

struct Entry {
    /// Stored as `bytes::Bytes` so clones are zero-copy (refcount bump only).
    /// Critical for `get_batch` and spill paths that would otherwise copy
    /// large RL objects (model weights, trajectories) multiple times.
    value: Option<bytes::Bytes>,
    /// If set, the task that produces this object failed. `get` returns this
    /// error instead of waiting forever or returning garbage bytes.
    error: Option<CrayonError>,
    notify: Arc<Notify>,
    refcount: Arc<AtomicUsize>,
    owner_node: Option<crate::node::NodeID>,
    spilled: bool,
    /// True while the object is being asynchronously written to disk.
    /// Readers must wait for it to finish (or fail) before reading from disk.
    spilling: bool,
    /// Size in bytes, for GCS metadata and memory accounting.
    size_bytes: usize,
}

pub(crate) struct StoreInner {
    map: Mutex<HashMap<ObjectID, Arc<Mutex<Entry>>>>,
    mem: Mutex<Option<Arc<MemoryManager>>>,
    remote: Mutex<Option<Arc<dyn RemoteFetcher>>>,
}

#[derive(Clone)]
pub struct ObjectStore {
    inner: Arc<StoreInner>,
}

impl ObjectStore {
    pub fn new() -> Self {
        ObjectStore {
            inner: Arc::new(StoreInner {
                map: Mutex::new(HashMap::new()),
                mem: Mutex::new(None),
                remote: Mutex::new(None),
            }),
        }
    }

    /// Attach a memory manager for disk spilling.
    pub fn with_memory(self, mem: Arc<MemoryManager>) -> Self {
        *self.inner.mem.lock() = Some(mem);
        self
    }

    /// Attach a remote fetcher for cross-node object retrieval.
    pub fn with_remote(self, remote: Arc<dyn RemoteFetcher>) -> Self {
        *self.inner.remote.lock() = Some(remote);
        self
    }

    pub(crate) fn downgrade(&self) -> Weak<StoreInner> {
        Arc::downgrade(&self.inner)
    }

    pub fn put<T: serde::Serialize + Send + 'static>(&self, value: T) -> (ObjectRef<T>, usize) {
        let id = ObjectID::new();
        let bytes = bincode::serialize(&value).expect("bincode serialize should not fail");
        let size = bytes.len();
        self.put_bytes(id, bytes, None);
        let entry = self.entry(id, None);
        let e = entry.lock();
        e.refcount.fetch_add(1, Ordering::Relaxed);
        let refcount = e.refcount.clone();
        drop(e);
        (ObjectRef::new(id, refcount, self.downgrade()), size)
    }

    /// Store many objects at once. Returns one `ObjectRef` per input. Amortizes
    /// lock acquisition — critical for high-throughput workloads like RL
    /// experience replay, where thousands of samples are pushed per step.
    pub fn put_batch<T: serde::Serialize + Send + 'static>(
        &self,
        values: Vec<T>,
    ) -> Vec<ObjectRef<T>> {
        values
            .into_iter()
            .map(|v| self.put(v).0)
            .collect()
    }

    pub fn put_with_id<T: serde::Serialize + Send + 'static>(&self, id: ObjectID, value: T) {
        let bytes = bincode::serialize(&value).expect("bincode serialize should not fail");
        self.put_bytes(id, bytes, None);
    }

    pub fn put_bytes(&self, id: ObjectID, bytes: Vec<u8>, owner_node: Option<crate::node::NodeID>) {
        let size = bytes.len();
        let bytes = bytes::Bytes::from(bytes);
        let entry = self.entry(id, owner_node);
        {
            let mut e = entry.lock();
            e.value = Some(bytes);
            e.size_bytes = size;
            e.spilled = false;
            e.notify.notify_waiters();
        }
        // Memory accounting + eviction. Scope the lock so it's released before
        // spill_to_disk runs (which also needs the mem lock).
        let evicted = {
            let mem = self.inner.mem.lock().clone();
            match mem {
                Some(m) => m.record_put(id, size),
                None => Vec::new(),
            }
        };
        for vid in evicted {
            self.spill_to_disk(vid);
        }
    }

    /// Mark an object as failed. Subsequent `get` calls return the error
    /// instead of waiting for a value that will never arrive.
    pub fn put_error(&self, id: ObjectID, error: CrayonError) {
        let entry = self.entry(id, None);
        let mut e = entry.lock();
        e.error = Some(error);
        e.notify.notify_waiters();
    }

    pub fn reserve_ref<T>(&self, id: ObjectID) -> ObjectRef<T> {
        let entry = self.entry(id, None);
        let e = entry.lock();
        e.refcount.fetch_add(1, Ordering::Relaxed);
        let refcount = e.refcount.clone();
        drop(e);
        ObjectRef::new(id, refcount, self.downgrade())
    }

    pub fn reserve(&self, id: ObjectID) {
        let _ = self.entry(id, None);
    }

    pub async fn get<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        id: ObjectID,
    ) -> Result<T, CrayonError> {
        let bytes = self.get_bytes(id).await?;
        bincode::deserialize(&bytes).map_err(|e| CrayonError::Serialize(e.to_string()))
    }

    /// Fetch many objects concurrently. Returns results in input order. For RL
    /// training this is the hot path: pulling a batch of rollout samples should
    /// be one concurrent fan-out, not N sequential RPCs.
    pub async fn get_batch<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        ids: &[ObjectID],
    ) -> Vec<Result<T, CrayonError>> {
        let store = self.clone();
        let futs: Vec<_> = ids
            .iter()
            .map(|id| {
                let store = store.clone();
                async move { store.get::<T>(*id).await }
            })
            .collect();
        futures::future::join_all(futs).await
    }

    pub async fn get_bytes(&self, id: ObjectID) -> Result<bytes::Bytes, CrayonError> {
        self.get_bytes_timeout(id, std::time::Duration::from_secs(30))
            .await
    }

    /// Like `get_bytes` but with a custom timeout for waiting on in-flight objects.
    pub async fn get_bytes_timeout(
        &self,
        id: ObjectID,
        timeout: std::time::Duration,
    ) -> Result<bytes::Bytes, CrayonError> {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut tried_remote = false;

        loop {
            // Check if the task producing this object failed.
            if let Some(err) = self.try_get_error(&id) {
                return Err(err);
            }

            // Fast path: in memory (zero-copy clone)
            if let Some(bytes) = self.try_get_in_memory(&id) {
                self.touch(&id);
                return Ok(bytes);
            }

            // Check if spilled to disk
            if self.is_spilled(&id) {
                match self.load_from_disk(&id) {
                    Ok(bytes) => {
                        let b = bytes::Bytes::from(bytes);
                        self.put_bytes(id, b.to_vec(), None);
                        return Ok(b);
                    }
                    Err(_) => {
                        // Spilled but file is gone; fall through to remote fetch
                    }
                }
            }

            // If currently spilling to disk, wait for it to finish.
            if self.is_spilling(&id) {
                let entry = self.entry(id, None);
                let notified = { entry.lock().notify.clone() };
                match tokio::time::timeout_at(deadline, notified.notified()).await {
                    Ok(_) => continue,
                    Err(_) => return Err(CrayonError::Timeout(id)),
                }
            }

            // Try remote fetch (once)
            if !tried_remote {
                tried_remote = true;
                let remote = self.inner.remote.lock().clone();
                if let Some(remote) = remote {
                    if let Ok(bytes) = remote.fetch_remote(id).await {
                        let b = bytes::Bytes::from(bytes);
                        self.put_bytes(id, b.to_vec(), None);
                        return Ok(b);
                    }
                }
            }

            // If the entry was pre-reserved (in-flight task output), wait for it.
            let existed = self.inner.map.lock().contains_key(&id);
            if !existed {
                return Err(CrayonError::ObjectNotFound(id));
            }

            let entry = self.entry(id, None);
            let notified = { entry.lock().notify.clone() };
            match tokio::time::timeout_at(deadline, notified.notified()).await {
                Ok(_) => continue,
                Err(_) => return Err(CrayonError::Timeout(id)),
            }
        }
    }

    fn try_get_in_memory(&self, id: &ObjectID) -> Option<bytes::Bytes> {
        self.inner
            .map
            .lock()
            .get(id)
            .and_then(|e| e.lock().value.clone())
    }

    fn try_get_error(&self, id: &ObjectID) -> Option<CrayonError> {
        self.inner
            .map
            .lock()
            .get(id)
            .and_then(|e| e.lock().error.clone())
    }

    fn is_spilled(&self, id: &ObjectID) -> bool {
        self.inner
            .map
            .lock()
            .get(id)
            .map(|e| e.lock().spilled)
            .unwrap_or(false)
    }

    fn touch(&self, id: &ObjectID) {
        if let Some(mem) = self.inner.mem.lock().clone() {
            mem.touch(*id);
        }
    }

    fn is_spilling(&self, id: &ObjectID) -> bool {
        self.inner
            .map
            .lock()
            .get(id)
            .map(|e| e.lock().spilling)
            .unwrap_or(false)
    }

    /// Asynchronously spill an object to disk. Takes the bytes out of memory
    /// immediately (so the memory budget is freed), then writes to disk on a
    /// background task. Readers see `spilling=true` and wait for completion.
    fn spill_to_disk(&self, id: ObjectID) {
        let bytes = {
            let map = self.inner.map.lock();
            if let Some(entry) = map.get(&id) {
                let mut e = entry.lock();
                e.spilling = true;
                e.value.take()
            } else {
                None
            }
        };
        let Some(bytes) = bytes else {
            return;
        };
        let mem = self.inner.mem.lock().clone();
        let store = self.inner.clone();

        let do_spill = move || {
            let result = if let Some(mem) = mem {
                let path = mem.spill_path(id);
                std::fs::write(&path, &bytes)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "no memory manager",
                ))
            };
            let map = store.map.lock();
            if let Some(entry) = map.get(&id) {
                let mut e = entry.lock();
                e.spilling = false;
                match result {
                    Ok(_) => e.spilled = true,
                    Err(_) => {
                        tracing::warn!("failed to spill object {id} to disk; keeping in memory");
                        e.value = Some(bytes);
                    }
                }
                e.notify.notify_waiters();
            }
        };

        // Spawn on the tokio runtime if available, else run synchronously.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn_blocking(do_spill);
            }
            Err(_) => do_spill(),
        }
    }

    fn load_from_disk(&self, id: &ObjectID) -> Result<Vec<u8>, std::io::Error> {
        if let Some(mem) = self.inner.mem.lock().clone() {
            let path = mem.spill_path(*id);
            let bytes = std::fs::read(&path)?;
            // Clean up the spill file
            let _ = std::fs::remove_file(&path);
            Ok(bytes)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no memory manager",
            ))
        }
    }

    pub fn contains(&self, id: ObjectID) -> bool {
        self.inner
            .map
            .lock()
            .get(&id)
            .map(|e| {
                let e = e.lock();
                e.value.is_some() || e.spilled || e.spilling
            })
            .unwrap_or(false)
    }

    pub fn owner_node(&self, id: ObjectID) -> Option<crate::node::NodeID> {
        self.inner
            .map
            .lock()
            .get(&id)
            .and_then(|e| e.lock().owner_node)
    }

    pub fn delete(&self, id: ObjectID) {
        self.inner.map.lock().remove(&id);
        if let Some(mem) = self.inner.mem.lock().clone() {
            mem.remove(id);
        }
    }

    pub fn len(&self) -> usize {
        self.inner
            .map
            .lock()
            .values()
            .filter(|e| {
                let e = e.lock();
                e.value.is_some() || e.spilled || e.spilling
            })
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return (used_bytes, limit_bytes) for the memory manager, if attached.
    /// `limit_bytes` is 0 when no memory manager is configured.
    pub fn memory_stats(&self) -> (usize, usize) {
        if let Some(mem) = self.inner.mem.lock().clone() {
            (mem.total_bytes(), mem.max_memory_bytes())
        } else {
            (0, 0)
        }
    }

    /// Size in bytes of a stored object, or 0 if not found / not yet materialized.
    pub fn object_size(&self, id: ObjectID) -> usize {
        self.inner
            .map
            .lock()
            .get(&id)
            .map(|e| e.lock().size_bytes)
            .unwrap_or(0)
    }

    fn entry(&self, id: ObjectID, owner_node: Option<crate::node::NodeID>) -> Arc<Mutex<Entry>> {
        let mut map = self.inner.map.lock();
        map.entry(id)
            .or_insert_with(|| {
                Arc::new(Mutex::new(Entry {
                    value: None,
                    error: None,
                    notify: Arc::new(Notify::new()),
                    refcount: Arc::new(AtomicUsize::new(0)),
                    owner_node,
                    spilled: false,
                    spilling: false,
                    size_bytes: 0,
                }))
            })
            .clone()
    }
}

impl Default for ObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn on_ref_dropped(store: &Weak<StoreInner>, id: ObjectID) {
    if let Some(store) = store.upgrade() {
        store.map.lock().remove(&id);
        if let Some(mem) = store.mem.lock().clone() {
            mem.remove(id);
        }
    }
}
