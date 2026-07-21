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

/// Trait for fetching objects from remote nodes. Implemented by [`Node`].
pub trait RemoteFetcher: Send + Sync {
    fn fetch_remote(
        &self,
        id: ObjectID,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>> + Send>>;
}

struct Entry {
    value: Option<Vec<u8>>,
    notify: Arc<Notify>,
    refcount: Arc<AtomicUsize>,
    owner_node: Option<crate::node::NodeID>,
    spilled: bool,
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

    pub fn put<T: serde::Serialize + Send + 'static>(&self, value: T) -> ObjectRef<T> {
        let id = ObjectID::new();
        let bytes = bincode::serialize(&value).expect("bincode serialize should not fail");
        self.put_bytes(id, bytes, None);
        let entry = self.entry(id, None);
        let e = entry.lock();
        e.refcount.fetch_add(1, Ordering::Relaxed);
        let refcount = e.refcount.clone();
        drop(e);
        ObjectRef::new(id, refcount, self.downgrade())
    }

    pub fn put_with_id<T: serde::Serialize + Send + 'static>(&self, id: ObjectID, value: T) {
        let bytes = bincode::serialize(&value).expect("bincode serialize should not fail");
        self.put_bytes(id, bytes, None);
    }

    pub fn put_bytes(&self, id: ObjectID, bytes: Vec<u8>, owner_node: Option<crate::node::NodeID>) {
        let size = bytes.len();
        let entry = self.entry(id, owner_node);
        {
            let mut e = entry.lock();
            e.value = Some(bytes);
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

    pub async fn get_bytes(&self, id: ObjectID) -> Result<Vec<u8>, CrayonError> {
        self.get_bytes_timeout(id, std::time::Duration::from_secs(30))
            .await
    }

    /// Like `get_bytes` but with a custom timeout for waiting on in-flight objects.
    pub async fn get_bytes_timeout(
        &self,
        id: ObjectID,
        timeout: std::time::Duration,
    ) -> Result<Vec<u8>, CrayonError> {
        // Fast path: in memory
        if let Some(bytes) = self.try_get_in_memory(&id) {
            self.touch(&id);
            return Ok(bytes);
        }

        // Check if spilled to disk
        if self.is_spilled(&id) {
            match self.load_from_disk(&id) {
                Ok(bytes) => {
                    self.put_bytes(id, bytes.clone(), None);
                    return Ok(bytes);
                }
                Err(_) => {
                    // Spilled but file is gone; fall through to remote fetch
                }
            }
        }

        // Try remote fetch
        let remote = self.inner.remote.lock().clone();
        if let Some(remote) = remote {
            if let Ok(bytes) = remote.fetch_remote(id).await {
                self.put_bytes(id, bytes.clone(), None);
                return Ok(bytes);
            }
        }

        // If the entry was pre-reserved (in-flight task output), wait for it.
        let existed = self.inner.map.lock().contains_key(&id);
        if !existed {
            return Err(CrayonError::ObjectNotFound(id));
        }

        let entry = self.entry(id, None);
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = {
                let e = entry.lock();
                if let Some(bytes) = e.value.as_ref() {
                    return Ok(bytes.clone());
                }
                e.notify.clone()
            };
            match tokio::time::timeout_at(deadline, notified.notified()).await {
                Ok(_) => continue,
                Err(_) => return Err(CrayonError::Timeout(id)),
            }
        }
    }

    fn try_get_in_memory(&self, id: &ObjectID) -> Option<Vec<u8>> {
        self.inner
            .map
            .lock()
            .get(id)
            .and_then(|e| e.lock().value.clone())
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

    fn spill_to_disk(&self, id: ObjectID) {
        // Take the bytes out of the entry first, releasing locks before disk I/O
        let bytes = {
            let map = self.inner.map.lock();
            if let Some(entry) = map.get(&id) {
                let mut e = entry.lock();
                e.value.take()
            } else {
                None
            }
        };
        if let Some(bytes) = bytes {
            if let Some(mem) = self.inner.mem.lock().clone() {
                let path = mem.spill_path(id);
                if std::fs::write(&path, &bytes).is_ok() {
                    // Mark as spilled
                    let map = self.inner.map.lock();
                    if let Some(entry) = map.get(&id) {
                        entry.lock().spilled = true;
                    }
                } else {
                    // Put it back if spill fails
                    let map = self.inner.map.lock();
                    if let Some(entry) = map.get(&id) {
                        entry.lock().value = Some(bytes);
                    }
                }
            }
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
                e.value.is_some() || e.spilled
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
                e.value.is_some() || e.spilled
            })
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn entry(&self, id: ObjectID, owner_node: Option<crate::node::NodeID>) -> Arc<Mutex<Entry>> {
        let mut map = self.inner.map.lock();
        map.entry(id)
            .or_insert_with(|| {
                Arc::new(Mutex::new(Entry {
                    value: None,
                    notify: Arc::new(Notify::new()),
                    refcount: Arc::new(AtomicUsize::new(0)),
                    owner_node,
                    spilled: false,
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
