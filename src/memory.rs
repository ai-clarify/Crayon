//! Resident-memory accounting and LRU spill selection.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::common::ObjectID;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Eviction {
    pub id: ObjectID,
    pub generation: u64,
    pub size: usize,
}

pub struct MemoryManager {
    inner: Mutex<MemoryInner>,
}

struct MemoryInner {
    sizes: HashMap<ObjectID, (u64, usize)>,
    lru: Vec<(ObjectID, u64)>,
    total_bytes: usize,
    spill_dir: PathBuf,
    max_memory_bytes: usize,
    spill_failures: usize,
}

impl MemoryManager {
    pub fn new(max_memory_bytes: usize, spill_dir: PathBuf) -> Arc<Self> {
        std::fs::create_dir_all(&spill_dir).ok();
        Arc::new(Self {
            inner: Mutex::new(MemoryInner {
                sizes: HashMap::new(),
                lru: Vec::new(),
                total_bytes: 0,
                spill_dir,
                max_memory_bytes,
                spill_failures: 0,
            }),
        })
    }

    pub(crate) fn record_put(&self, id: ObjectID, generation: u64, size: usize) -> Vec<Eviction> {
        let mut inner = self.inner.lock();
        if let Some((_, old)) = inner.sizes.insert(id, (generation, size)) {
            inner.total_bytes = inner.total_bytes.saturating_sub(old);
        }
        inner.total_bytes += size;
        inner.lru.retain(|(item, _)| *item != id);
        inner.lru.push((id, generation));
        select_evictions(&mut inner)
    }

    /// Restore resident accounting after a failed spill. The object is kept out
    /// of the LRU until its next access so one bad path cannot create a retry loop.
    pub(crate) fn restore_after_spill_failure(&self, id: ObjectID, generation: u64, size: usize) {
        let mut inner = self.inner.lock();
        if !matches!(inner.sizes.get(&id), Some((g, _)) if *g == generation) {
            inner.sizes.insert(id, (generation, size));
            inner.total_bytes += size;
        }
        inner.lru.retain(|(item, _)| *item != id);
        inner.spill_failures += 1;
    }

    pub(crate) fn touch(&self, id: ObjectID, generation: u64) {
        let mut inner = self.inner.lock();
        if matches!(inner.sizes.get(&id), Some((g, _)) if *g == generation) {
            inner.lru.retain(|(item, _)| *item != id);
            inner.lru.push((id, generation));
        }
    }

    pub(crate) fn remove(&self, id: ObjectID, generation: u64) {
        let mut inner = self.inner.lock();
        if matches!(inner.sizes.get(&id), Some((g, _)) if *g == generation) {
            if let Some((_, size)) = inner.sizes.remove(&id) {
                inner.total_bytes = inner.total_bytes.saturating_sub(size);
            }
        }
        inner
            .lru
            .retain(|(item, item_generation)| *item != id || *item_generation != generation);
        let path = spill_path(&inner.spill_dir, id, generation);
        std::fs::remove_file(path).ok();
    }

    pub fn total_bytes(&self) -> usize {
        self.inner.lock().total_bytes
    }

    pub fn max_memory_bytes(&self) -> usize {
        self.inner.lock().max_memory_bytes
    }

    pub fn usage_ratio(&self) -> f64 {
        let inner = self.inner.lock();
        if inner.max_memory_bytes == 0 {
            0.0
        } else {
            inner.total_bytes as f64 / inner.max_memory_bytes as f64
        }
    }

    pub(crate) fn spill_path(&self, id: ObjectID, generation: u64) -> PathBuf {
        spill_path(&self.inner.lock().spill_dir, id, generation)
    }

    pub fn num_objects(&self) -> usize {
        self.inner.lock().sizes.len()
    }

    pub fn spill_failures(&self) -> usize {
        self.inner.lock().spill_failures
    }
}

fn select_evictions(inner: &mut MemoryInner) -> Vec<Eviction> {
    let mut evicted = Vec::new();
    while inner.total_bytes > inner.max_memory_bytes {
        let Some((id, generation)) = inner.lru.first().copied() else {
            break;
        };
        inner.lru.remove(0);
        if let Some((current_generation, size)) = inner.sizes.get(&id).copied() {
            if current_generation == generation {
                inner.sizes.remove(&id);
                inner.total_bytes = inner.total_bytes.saturating_sub(size);
                evicted.push(Eviction {
                    id,
                    generation,
                    size,
                });
            }
        }
    }
    evicted
}

fn spill_path(dir: &std::path::Path, id: ObjectID, generation: u64) -> PathBuf {
    dir.join(format!("{id}.{generation}.bin"))
}
