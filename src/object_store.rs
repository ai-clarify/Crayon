//! In-process object store with generation-safe references and disk spilling.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;
use tokio::sync::watch;

use crate::common::{CrayonError, ObjectID, ObjectRef, TaskID};
use crate::gcs::{Gcs, ObjectMeta};
use crate::memory::{Eviction, MemoryManager};

pub type RemoteFetchFuture = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>>
            + Send,
    >,
>;

pub trait RemoteFetcher: Send + Sync {
    fn fetch_remote(&self, id: ObjectID) -> RemoteFetchFuture;
}

#[derive(Clone)]
enum EntryState {
    Pending,
    Bytes(bytes::Bytes),
    Local(Arc<dyn std::any::Any + Send + Sync>),
    Spilling,
    Spilled,
    Failed(CrayonError),
}

struct Entry {
    generation: u64,
    state: EntryState,
    changed: watch::Sender<u64>,
    version: u64,
    refcount: Arc<AtomicUsize>,
    producers: usize,
    pins: usize,
    owner_node: Option<crate::node::NodeID>,
    size_bytes: usize,
}

pub(crate) struct StoreInner {
    map: Mutex<HashMap<ObjectID, Arc<Mutex<Entry>>>>,
    next_generation: AtomicU64,
    mem: Mutex<Option<Arc<MemoryManager>>>,
    remote: Mutex<Option<Arc<dyn RemoteFetcher>>>,
    gcs: Mutex<Option<Gcs>>,
}

#[derive(Clone)]
pub struct ObjectStore {
    inner: Arc<StoreInner>,
}

/// A producer may complete only the exact object generation it reserved.
pub(crate) struct ProducerLease {
    store: Weak<StoreInner>,
    id: ObjectID,
    generation: u64,
    finished: AtomicBool,
}

impl ProducerLease {
    pub(crate) fn publish(&self, bytes: Vec<u8>, owner_node: Option<crate::node::NodeID>) -> bool {
        if self.finished.swap(true, Ordering::AcqRel) {
            return false;
        }
        finish_producer(
            &self.store,
            self.id,
            self.generation,
            Some(Ok(bytes)),
            owner_node,
        )
    }

    pub(crate) fn fail(&self, error: CrayonError) -> bool {
        if self.finished.swap(true, Ordering::AcqRel) {
            return false;
        }
        finish_producer(
            &self.store,
            self.id,
            self.generation,
            Some(Err(error)),
            None,
        )
    }
}

impl Drop for ProducerLease {
    fn drop(&mut self) {
        if !self.finished.swap(true, Ordering::AcqRel) {
            finish_producer(&self.store, self.id, self.generation, None, None);
        }
    }
}

impl ObjectStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(StoreInner {
                map: Mutex::new(HashMap::new()),
                next_generation: AtomicU64::new(1),
                mem: Mutex::new(None),
                remote: Mutex::new(None),
                gcs: Mutex::new(None),
            }),
        }
    }

    pub fn with_memory(self, mem: Arc<MemoryManager>) -> Self {
        *self.inner.mem.lock() = Some(mem);
        self
    }

    pub fn with_remote(self, remote: Arc<dyn RemoteFetcher>) -> Self {
        *self.inner.remote.lock() = Some(remote);
        self
    }

    pub(crate) fn with_gcs(self, gcs: Gcs) -> Self {
        *self.inner.gcs.lock() = Some(gcs);
        self
    }

    pub(crate) fn downgrade(&self) -> Weak<StoreInner> {
        Arc::downgrade(&self.inner)
    }

    pub fn put<T: serde::Serialize + Send + 'static>(&self, value: T) -> (ObjectRef<T>, usize) {
        let id = ObjectID::new();
        let bytes = bincode::serialize(&value).expect("bincode serialize should not fail");
        let size = bytes.len();
        let generation = self.replace_entry(
            id,
            None,
            EntryState::Bytes(bytes::Bytes::from(bytes)),
            size,
            0,
            0,
        );
        (self.reserve_ref_for_generation(id, generation, None), size)
    }

    pub fn put_batch<T: serde::Serialize + Send + 'static>(
        &self,
        values: Vec<T>,
    ) -> Vec<ObjectRef<T>> {
        values.into_iter().map(|value| self.put(value).0).collect()
    }

    pub fn put_with_id<T: serde::Serialize + Send + 'static>(&self, id: ObjectID, value: T) {
        let bytes = bincode::serialize(&value).expect("bincode serialize should not fail");
        self.put_bytes(id, bytes, None);
    }

    pub fn put_bytes_ref(&self, bytes: Vec<u8>) -> ObjectRef<Vec<u8>> {
        let id = ObjectID::new();
        self.put_bytes(id, bytes, None);
        self.reserve_ref(id)
    }

    /// Replace an object explicitly. Async producers should use a producer lease.
    pub fn put_bytes(&self, id: ObjectID, bytes: Vec<u8>, owner_node: Option<crate::node::NodeID>) {
        let size = bytes.len();
        self.replace_entry(
            id,
            owner_node,
            EntryState::Bytes(bytes::Bytes::from(bytes)),
            size,
            0,
            0,
        );
    }

    pub fn put_error(&self, id: ObjectID, error: CrayonError) {
        self.replace_entry(id, None, EntryState::Failed(error), 0, 0, 0);
    }

    /// Reserve a consumer reference and a lease for an asynchronous producer.
    pub(crate) fn reserve_output<T>(
        &self,
        id: ObjectID,
        task_id: Option<TaskID>,
    ) -> (ObjectRef<T>, Arc<ProducerLease>) {
        let generation = self.replace_entry(id, None, EntryState::Pending, 0, 1, 0);
        let reference = self.reserve_ref_for_generation(id, generation, task_id);
        let lease = Arc::new(ProducerLease {
            store: self.downgrade(),
            id,
            generation,
            finished: AtomicBool::new(false),
        });
        (reference, lease)
    }

    pub fn reserve_ref<T>(&self, id: ObjectID) -> ObjectRef<T> {
        let generation = self.ensure_entry(id, None);
        self.reserve_ref_for_generation(id, generation, None)
    }

    pub fn reserve(&self, id: ObjectID) {
        self.ensure_entry(id, None);
    }

    pub(crate) fn pin(&self, id: ObjectID) -> u64 {
        let generation = self.ensure_entry(id, None);
        if let Some(entry) = self.inner.map.lock().get(&id).cloned() {
            let mut entry = entry.lock();
            if entry.generation == generation {
                entry.pins += 1;
            }
        }
        generation
    }

    pub(crate) fn unpin(&self, id: ObjectID, generation: u64) {
        let mut remove = false;
        if let Some(entry) = self.inner.map.lock().get(&id).cloned() {
            let mut entry = entry.lock();
            if entry.generation == generation {
                entry.pins = entry.pins.saturating_sub(1);
                remove = entry.pins == 0
                    && entry.producers == 0
                    && entry.refcount.load(Ordering::Acquire) == 0;
            }
        }
        if remove {
            remove_generation(&self.inner, id, generation);
        }
    }

    pub async fn get<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        id: ObjectID,
    ) -> Result<T, CrayonError> {
        let bytes = self.get_bytes(id).await?;
        bincode::deserialize(&bytes).map_err(|e| CrayonError::Serialize(e.to_string()))
    }

    pub async fn get_batch<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        ids: &[ObjectID],
    ) -> Vec<Result<T, CrayonError>> {
        futures::future::join_all(ids.iter().map(|id| self.get::<T>(*id))).await
    }

    pub async fn get_bytes(&self, id: ObjectID) -> Result<bytes::Bytes, CrayonError> {
        self.get_bytes_timeout(id, std::time::Duration::from_secs(300))
            .await
    }

    pub async fn get_bytes_timeout(
        &self,
        id: ObjectID,
        timeout: std::time::Duration,
    ) -> Result<bytes::Bytes, CrayonError> {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut tried_remote = false;

        loop {
            let snapshot = self.snapshot(id);
            if let Some((generation, state, mut changed)) = snapshot {
                match state {
                    EntryState::Bytes(bytes) => {
                        self.touch(id, generation);
                        return Ok(bytes);
                    }
                    EntryState::Failed(error) => return Err(error),
                    EntryState::Spilled => {
                        if let Some(bytes) = self.load_from_disk(id, generation)? {
                            return Ok(bytes);
                        }
                    }
                    EntryState::Local(_) => return Err(CrayonError::TypeMismatch),
                    EntryState::Pending | EntryState::Spilling => {}
                }

                if !tried_remote && !matches!(state, EntryState::Pending | EntryState::Spilling) {
                    tried_remote = true;
                    if let Some(bytes) = self.fetch_remote(id).await {
                        return Ok(bytes);
                    }
                }

                match tokio::time::timeout_at(deadline, changed.changed()).await {
                    Ok(Ok(())) | Ok(Err(_)) => continue,
                    Err(_) => return Err(CrayonError::Timeout(id)),
                }
            }

            if !tried_remote {
                if let Some(bytes) = self.fetch_remote(id).await {
                    return Ok(bytes);
                }
            }
            return Err(CrayonError::ObjectNotFound(id));
        }
    }

    async fn fetch_remote(&self, id: ObjectID) -> Option<bytes::Bytes> {
        let remote = self.inner.remote.lock().clone()?;
        let bytes = remote.fetch_remote(id).await.ok()?;
        let value = bytes::Bytes::from(bytes);
        self.put_bytes(id, value.to_vec(), None);
        Some(value)
    }

    fn snapshot(&self, id: ObjectID) -> Option<(u64, EntryState, watch::Receiver<u64>)> {
        let entry = self.inner.map.lock().get(&id).cloned()?;
        let entry = entry.lock();
        Some((
            entry.generation,
            entry.state.clone(),
            entry.changed.subscribe(),
        ))
    }

    fn touch(&self, id: ObjectID, generation: u64) {
        if let Some(mem) = self.inner.mem.lock().clone() {
            mem.touch(id, generation);
        }
    }

    fn spill_to_disk(&self, eviction: Eviction) {
        let bytes = {
            let Some(entry) = self.inner.map.lock().get(&eviction.id).cloned() else {
                return;
            };
            let mut entry = entry.lock();
            if entry.generation != eviction.generation {
                return;
            }
            match std::mem::replace(&mut entry.state, EntryState::Spilling) {
                EntryState::Bytes(bytes) => {
                    notify(&mut entry);
                    bytes
                }
                state => {
                    entry.state = state;
                    return;
                }
            }
        };

        let mem = self.inner.mem.lock().clone();
        let store = self.inner.clone();
        let run = move || {
            let result = mem.as_ref().map_or_else(
                || {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "no memory manager",
                    ))
                },
                |mem| std::fs::write(mem.spill_path(eviction.id, eviction.generation), &bytes),
            );
            let current = store.map.lock().get(&eviction.id).cloned();
            let Some(entry) = current else {
                if let Some(mem) = mem {
                    std::fs::remove_file(mem.spill_path(eviction.id, eviction.generation)).ok();
                }
                return;
            };
            let mut entry = entry.lock();
            if entry.generation != eviction.generation {
                if let Some(mem) = mem {
                    std::fs::remove_file(mem.spill_path(eviction.id, eviction.generation)).ok();
                }
                return;
            }
            match result {
                Ok(()) => entry.state = EntryState::Spilled,
                Err(error) => {
                    tracing::warn!("failed to spill object {}: {error}", eviction.id);
                    entry.state = EntryState::Bytes(bytes);
                    if let Some(mem) = mem {
                        mem.restore_after_spill_failure(
                            eviction.id,
                            eviction.generation,
                            eviction.size,
                        );
                    }
                }
            }
            notify(&mut entry);
        };
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn_blocking(run);
            }
            Err(_) => run(),
        }
    }

    fn load_from_disk(
        &self,
        id: ObjectID,
        generation: u64,
    ) -> Result<Option<bytes::Bytes>, CrayonError> {
        let Some(mem) = self.inner.mem.lock().clone() else {
            return Ok(None);
        };
        let path = mem.spill_path(id, generation);
        let bytes = std::fs::read(&path).map_err(|e| CrayonError::TaskFailed(e.to_string()))?;
        let value = bytes::Bytes::from(bytes);
        let current = self.inner.map.lock().get(&id).cloned();
        let Some(entry) = current else {
            std::fs::remove_file(path).ok();
            return Ok(None);
        };
        let mut entry = entry.lock();
        if entry.generation != generation || !matches!(entry.state, EntryState::Spilled) {
            return Ok(None);
        }
        entry.state = EntryState::Bytes(value.clone());
        entry.size_bytes = value.len();
        notify(&mut entry);
        std::fs::remove_file(path).ok();
        drop(entry);
        let evictions = mem.record_put(id, generation, value.len());
        for eviction in evictions {
            self.spill_to_disk(eviction);
        }
        Ok(Some(value))
    }

    pub fn contains(&self, id: ObjectID) -> bool {
        self.inner.map.lock().get(&id).is_some_and(|entry| {
            matches!(
                entry.lock().state,
                EntryState::Bytes(_)
                    | EntryState::Local(_)
                    | EntryState::Spilling
                    | EntryState::Spilled
            )
        })
    }

    pub fn owner_node(&self, id: ObjectID) -> Option<crate::node::NodeID> {
        self.inner
            .map
            .lock()
            .get(&id)
            .and_then(|entry| entry.lock().owner_node)
    }

    pub fn delete(&self, id: ObjectID) {
        let generation = self
            .inner
            .map
            .lock()
            .get(&id)
            .map(|entry| entry.lock().generation);
        if let Some(generation) = generation {
            remove_generation(&self.inner, id, generation);
        }
    }

    pub fn len(&self) -> usize {
        self.inner
            .map
            .lock()
            .values()
            .filter(|entry| {
                matches!(
                    entry.lock().state,
                    EntryState::Bytes(_)
                        | EntryState::Local(_)
                        | EntryState::Spilling
                        | EntryState::Spilled
                )
            })
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn memory_stats(&self) -> (usize, usize) {
        self.inner
            .mem
            .lock()
            .clone()
            .map_or((0, 0), |mem| (mem.total_bytes(), mem.max_memory_bytes()))
    }

    pub fn spill_failures(&self) -> usize {
        self.inner
            .mem
            .lock()
            .clone()
            .map_or(0, |mem| mem.spill_failures())
    }

    pub fn object_size(&self, id: ObjectID) -> usize {
        self.inner
            .map
            .lock()
            .get(&id)
            .map_or(0, |entry| entry.lock().size_bytes)
    }

    pub fn put_local<T: Send + Sync + 'static>(&self, id: ObjectID, value: T) {
        self.replace_entry(id, None, EntryState::Local(Arc::new(value)), 0, 0, 0);
    }

    pub fn get_local<T: Send + Sync + 'static>(&self, id: ObjectID) -> Result<Arc<T>, CrayonError> {
        let entry = self
            .inner
            .map
            .lock()
            .get(&id)
            .cloned()
            .ok_or(CrayonError::ObjectNotFound(id))?;
        let state = entry.lock().state.clone();
        match state {
            EntryState::Local(value) => {
                value.downcast::<T>().map_err(|_| CrayonError::TypeMismatch)
            }
            _ => Err(CrayonError::ObjectNotFound(id)),
        }
    }

    pub fn is_local(&self, id: ObjectID) -> bool {
        self.inner
            .map
            .lock()
            .get(&id)
            .is_some_and(|entry| matches!(entry.lock().state, EntryState::Local(_)))
    }

    fn ensure_entry(&self, id: ObjectID, owner_node: Option<crate::node::NodeID>) -> u64 {
        if let Some(entry) = self.inner.map.lock().get(&id).cloned() {
            return entry.lock().generation;
        }
        self.replace_entry(id, owner_node, EntryState::Pending, 0, 0, 0)
    }

    fn reserve_ref_for_generation<T>(
        &self,
        id: ObjectID,
        generation: u64,
        task_id: Option<TaskID>,
    ) -> ObjectRef<T> {
        let entry = self
            .inner
            .map
            .lock()
            .get(&id)
            .cloned()
            .expect("reserved entry exists");
        let entry = entry.lock();
        assert_eq!(entry.generation, generation);
        entry.refcount.fetch_add(1, Ordering::Relaxed);
        ObjectRef::new(
            id,
            generation,
            entry.refcount.clone(),
            self.downgrade(),
            task_id,
        )
    }

    fn replace_entry(
        &self,
        id: ObjectID,
        owner_node: Option<crate::node::NodeID>,
        state: EntryState,
        size: usize,
        producers: usize,
        pins: usize,
    ) -> u64 {
        let generation = self.inner.next_generation.fetch_add(1, Ordering::Relaxed);
        let (changed, _) = watch::channel(0);
        let entry = Arc::new(Mutex::new(Entry {
            generation,
            state,
            changed,
            version: 0,
            refcount: Arc::new(AtomicUsize::new(0)),
            producers,
            pins,
            owner_node,
            size_bytes: size,
        }));
        let old = self.inner.map.lock().insert(id, entry);
        if let Some(old) = old {
            let old_generation = old.lock().generation;
            self.cleanup_artifacts(id, old_generation);
        }
        if size > 0 {
            self.publish_metadata(id, size, owner_node);
            let resident = {
                let map = self.inner.map.lock();
                matches!(
                    map.get(&id).map(|entry| entry.lock().state.clone()),
                    Some(EntryState::Bytes(_))
                )
            };
            if resident {
                let mem = self.inner.mem.lock().clone();
                if let Some(mem) = mem {
                    let evictions = mem.record_put(id, generation, size);
                    for eviction in evictions {
                        self.spill_to_disk(eviction);
                    }
                }
            }
        }
        generation
    }

    fn publish_metadata(&self, id: ObjectID, size: usize, owner_node: Option<crate::node::NodeID>) {
        if let Some(gcs) = self.inner.gcs.lock().clone() {
            gcs.add_object(ObjectMeta {
                id,
                size_bytes: size,
                created_at: std::time::Instant::now(),
                owner: None,
                owner_node,
            });
        }
    }

    fn cleanup_artifacts(&self, id: ObjectID, generation: u64) {
        if let Some(mem) = self.inner.mem.lock().clone() {
            mem.remove(id, generation);
        }
    }
}

impl Default for ObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

fn notify(entry: &mut Entry) {
    entry.version = entry.version.wrapping_add(1);
    entry.changed.send_replace(entry.version);
}

fn finish_producer(
    store: &Weak<StoreInner>,
    id: ObjectID,
    generation: u64,
    result: Option<Result<Vec<u8>, CrayonError>>,
    owner_node: Option<crate::node::NodeID>,
) -> bool {
    let Some(store) = store.upgrade() else {
        return false;
    };
    let Some(entry) = store.map.lock().get(&id).cloned() else {
        return false;
    };
    let mut entry = entry.lock();
    if entry.generation != generation || entry.producers == 0 {
        return false;
    }
    entry.producers -= 1;
    let abandoned = entry.refcount.load(Ordering::Acquire) == 0 && entry.pins == 0;
    if abandoned {
        drop(entry);
        remove_generation(&store, id, generation);
        return false;
    }
    match result {
        Some(Ok(bytes)) => {
            let size = bytes.len();
            entry.state = EntryState::Bytes(bytes::Bytes::from(bytes));
            entry.owner_node = owner_node;
            entry.size_bytes = size;
            notify(&mut entry);
            drop(entry);
            if let Some(gcs) = store.gcs.lock().clone() {
                gcs.add_object(ObjectMeta {
                    id,
                    size_bytes: size,
                    created_at: std::time::Instant::now(),
                    owner: None,
                    owner_node,
                });
            }
            if let Some(mem) = store.mem.lock().clone() {
                let object_store = ObjectStore {
                    inner: store.clone(),
                };
                for eviction in mem.record_put(id, generation, size) {
                    object_store.spill_to_disk(eviction);
                }
            }
        }
        Some(Err(error)) => {
            entry.state = EntryState::Failed(error);
            notify(&mut entry);
        }
        None => {
            if entry.producers == 0 {
                entry.state = EntryState::Failed(CrayonError::ObjectNotFound(id));
                notify(&mut entry);
            }
        }
    }
    true
}

pub(crate) fn on_ref_dropped(
    store: &Weak<StoreInner>,
    id: ObjectID,
    generation: u64,
    refcount: &Arc<AtomicUsize>,
) {
    let Some(store) = store.upgrade() else {
        return;
    };
    let remove = store.map.lock().get(&id).is_some_and(|entry| {
        let entry = entry.lock();
        entry.generation == generation
            && Arc::ptr_eq(&entry.refcount, refcount)
            && entry.producers == 0
            && entry.pins == 0
            && entry.refcount.load(Ordering::Acquire) == 0
    });
    if remove {
        remove_generation(&store, id, generation);
    }
}

fn remove_generation(store: &Arc<StoreInner>, id: ObjectID, generation: u64) {
    let removed = {
        let mut map = store.map.lock();
        if map
            .get(&id)
            .is_some_and(|entry| entry.lock().generation == generation)
        {
            map.remove(&id)
        } else {
            None
        }
    };
    if removed.is_none() {
        return;
    }
    if let Some(mem) = store.mem.lock().clone() {
        mem.remove(id, generation);
    }
    if let Some(gcs) = store.gcs.lock().clone() {
        gcs.remove_object(id);
    }
}
