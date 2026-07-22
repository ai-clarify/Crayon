//! Versioned artifact storage for RL training.
//!
//! RL training produces a stream of intermediate products: policy checkpoints,
//! rollout batches, optimizer states, and metrics. These need to be versioned
//! so you can:
//! - Roll back to an earlier policy if training diverges
//! - Reproduce a result from a specific checkpoint
//! - Compare metrics across training steps
//! - Garbage-collect old versions to control memory/disk usage
//!
//! [`VersionedStore`] wraps the object store and tags each artifact with a
//! name, a monotonically increasing version number, and a timestamp. It keeps
//! a configurable history per name and evicts the oldest versions beyond that
//! window — so memory stays bounded even on long training runs (the other half
//! of the OOM story, alongside LRU spilling).
//!
//! ```ignore
//! let vs = VersionedStore::new(ray.store().clone(), 10); // keep last 10
//! vs.put("policy", &params, step)?;
//! let latest = vs.get::<Vec<f32>>("policy")?;
//! let old = vs.get_at::<Vec<f32>>("policy", step - 5)?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::common::ObjectID;
use crate::object_store::ObjectStore;

/// Metadata for one version of a named artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMeta {
    pub name: String,
    pub version: u64,
    pub created_at: u64, // unix seconds
    pub object_id: ObjectID,
    #[serde(skip)]
    pub generation: u64,
    pub size_bytes: usize,
}

/// Tracks versioned artifacts. Each `name` has a list of versions, newest last.
struct VersionedInner {
    /// name -> ordered versions (oldest first).
    versions: HashMap<String, Vec<VersionMeta>>,
    /// Max versions to keep per name. 0 = unlimited.
    keep_last: usize,
}

#[derive(Clone)]
pub struct VersionedStore {
    store: ObjectStore,
    inner: Arc<Mutex<VersionedInner>>,
}

impl VersionedStore {
    /// Create a versioned store. `keep_last` bounds how many versions to retain
    /// per name; older versions are evicted (and their objects freed) when the
    /// window is exceeded. Use 0 for unlimited.
    pub fn new(store: ObjectStore, keep_last: usize) -> Self {
        VersionedStore {
            store,
            inner: Arc::new(Mutex::new(VersionedInner {
                versions: HashMap::new(),
                keep_last,
            })),
        }
    }

    /// Store a new version of `name`. The version number is auto-incremented
    /// per name. Returns the metadata for the stored version.
    pub fn put<T: serde::Serialize + Send + 'static>(&self, name: &str, value: T) -> VersionMeta {
        let v = self.next_version(name);
        self.put_with_version(name, value, v)
    }

    /// Store a value with an explicit version number (e.g. the training step).
    pub fn put_with_version<T: serde::Serialize + Send + 'static>(
        &self,
        name: &str,
        value: T,
        version: u64,
    ) -> VersionMeta {
        let bytes = bincode::serialize(&value).expect("bincode serialize should not fail");
        self.put_serialized(name, bytes, version)
    }

    pub fn put_bytes_with_version(&self, name: &str, bytes: Vec<u8>, version: u64) -> VersionMeta {
        self.put_serialized(name, bytes, version)
    }

    pub async fn get_bytes(&self, name: &str) -> Option<bytes::Bytes> {
        let meta = self.inner.lock().versions.get(name)?.last()?.clone();
        self.store.get_bytes(meta.object_id).await.ok()
    }

    pub async fn get_bytes_at(&self, name: &str, version: u64) -> Option<bytes::Bytes> {
        let meta = self
            .inner
            .lock()
            .versions
            .get(name)?
            .iter()
            .find(|meta| meta.version == version)?
            .clone();
        self.store.get_bytes(meta.object_id).await.ok()
    }

    fn put_serialized(&self, name: &str, bytes: Vec<u8>, version: u64) -> VersionMeta {
        let id = crate::common::ObjectID::new();
        let size = bytes.len();
        self.store.put_bytes(id, bytes, None);
        let generation = self.store.pin(id);
        let meta = VersionMeta {
            name: name.to_string(),
            version,
            created_at: now_secs(),
            object_id: id,
            generation,
            size_bytes: size,
        };

        let mut inner = self.inner.lock();
        let keep_last = inner.keep_last;
        let entry = inner.versions.entry(name.to_string()).or_default();
        // If this version already exists, replace it (remove old object).
        if let Some(pos) = entry.iter().position(|m| m.version == version) {
            let old = entry.remove(pos);
            self.store.unpin(old.object_id, old.generation);
        }
        entry.push(meta.clone());

        // Evict old versions beyond the window.
        if keep_last > 0 && entry.len() > keep_last {
            let overflow = entry.len() - keep_last;
            for _ in 0..overflow {
                let old = entry.remove(0);
                self.store.unpin(old.object_id, old.generation);
            }
        }
        meta
    }

    /// Get the latest version of `name`.
    pub async fn get<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        name: &str,
    ) -> Option<T> {
        let meta = {
            let inner = self.inner.lock();
            inner.versions.get(name)?.last()?.clone()
        };
        // ObjectStore::get already handles bincode deserialization.
        self.store.get::<T>(meta.object_id).await.ok()
    }

    /// Get a specific version of `name`.
    pub async fn get_at<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        name: &str,
        version: u64,
    ) -> Option<T> {
        let meta = {
            let inner = self.inner.lock();
            inner
                .versions
                .get(name)?
                .iter()
                .find(|m| m.version == version)?
                .clone()
        };
        self.store.get::<T>(meta.object_id).await.ok()
    }

    /// List all versions of `name`, oldest first.
    pub fn history(&self, name: &str) -> Vec<VersionMeta> {
        self.inner
            .lock()
            .versions
            .get(name)
            .cloned()
            .unwrap_or_default()
    }

    /// Latest version number for `name`, or 0 if none.
    pub fn latest_version(&self, name: &str) -> u64 {
        self.inner
            .lock()
            .versions
            .get(name)
            .and_then(|v| v.last().map(|m| m.version))
            .unwrap_or(0)
    }

    /// Delete all versions of `name`.
    pub fn remove(&self, name: &str) {
        let mut inner = self.inner.lock();
        if let Some(versions) = inner.versions.remove(name) {
            for m in versions {
                self.store.unpin(m.object_id, m.generation);
            }
        }
    }

    fn next_version(&self, name: &str) -> u64 {
        self.latest_version(name) + 1
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
