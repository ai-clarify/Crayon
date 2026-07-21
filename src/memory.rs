//! Memory and disk management — Crayon's analog of Ray's plasma store
//! memory accounting + object spilling.
//!
//! When the in-memory object store exceeds `max_memory_bytes`, the
//! [`MemoryManager`] evicts least-recently-used (LRU) objects to a spill
//! directory on disk. Evicted objects are transparently reloaded on next
//! access. This lets the store hold more objects than fit in RAM, which is
//! critical for RL workloads that generate large amounts of experience data.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::common::ObjectID;

/// Tracks per-object memory usage and handles LRU eviction to disk.
pub struct MemoryManager {
    inner: Mutex<MemoryInner>,
}

struct MemoryInner {
    /// Object ID → size in bytes.
    sizes: HashMap<ObjectID, usize>,
    /// LRU list, most-recently-used at the back.
    lru: Vec<ObjectID>,
    /// Total bytes currently tracked.
    total_bytes: usize,
    /// Spill directory on disk.
    spill_dir: PathBuf,
    /// Maximum in-memory bytes before spilling.
    max_memory_bytes: usize,
}

impl MemoryManager {
    pub fn new(max_memory_bytes: usize, spill_dir: PathBuf) -> Arc<Self> {
        std::fs::create_dir_all(&spill_dir).ok();
        Arc::new(MemoryManager {
            inner: Mutex::new(MemoryInner {
                sizes: HashMap::new(),
                lru: Vec::new(),
                total_bytes: 0,
                spill_dir,
                max_memory_bytes,
            }),
        })
    }

    /// Record that an object was stored (or updated). Returns a list of object
    /// IDs that were evicted to disk to make room.
    pub fn record_put(&self, id: ObjectID, size: usize) -> Vec<ObjectID> {
        let mut inner = self.inner.lock();
        if let Some(old) = inner.sizes.insert(id, size) {
            inner.total_bytes -= old;
        }
        inner.total_bytes += size;
        // Touch: move to back (most recent)
        inner.lru.retain(|x| *x != id);
        inner.lru.push(id);

        // Evict if over budget
        let mut evicted = Vec::new();
        while inner.total_bytes > inner.max_memory_bytes && inner.lru.len() > 1 {
            let victim = inner.lru.remove(0); // LRU is at front
            if let Some(sz) = inner.sizes.remove(&victim) {
                inner.total_bytes -= sz;
                evicted.push(victim);
            }
        }
        evicted
    }

    /// Record that an object was accessed (move to back of LRU).
    pub fn touch(&self, id: ObjectID) {
        let mut inner = self.inner.lock();
        inner.lru.retain(|x| *x != id);
        inner.lru.push(id);
    }

    /// Record that an object was removed. Also deletes its spill file if any.
    pub fn remove(&self, id: ObjectID) {
        let mut inner = self.inner.lock();
        if let Some(sz) = inner.sizes.remove(&id) {
            inner.total_bytes -= sz;
        }
        inner.lru.retain(|x| *x != id);
        // Clean up spill file to prevent disk leaks (#53261 analog).
        let path = inner.spill_dir.join(format!("{}.bin", id));
        std::fs::remove_file(&path).ok();
    }

    pub fn total_bytes(&self) -> usize {
        self.inner.lock().total_bytes
    }

    pub fn spill_path(&self, id: ObjectID) -> PathBuf {
        self.inner.lock().spill_dir.join(format!("{}.bin", id))
    }

    pub fn num_objects(&self) -> usize {
        self.inner.lock().sizes.len()
    }
}
