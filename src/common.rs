//! Core types shared across the Crayon runtime.
//!
//! Mirrors Ray's identity primitives: ObjectID, ActorID, TaskID, and the
//! ObjectRef handle that users pass around.

use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Monotonic counter used to mint unique IDs within a process.
static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Unique identifier for an object in the object store.
///
/// Ray uses a 28-byte ObjectID (task ID + index). We mint a 16-byte UUID
/// prefixed with a process-local counter for human readability.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ObjectID([u8; 16]);

impl ObjectID {
    pub fn new() -> Self {
        let _ = next_id();
        ObjectID(*Uuid::new_v4().as_bytes())
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl Default for ObjectID {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ObjectID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Obj({})", &hex(&self.0)[..8])
    }
}

impl fmt::Display for ObjectID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex(&self.0))
    }
}

/// Unique identifier for an actor.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ActorID([u8; 16]);

impl ActorID {
    pub fn new() -> Self {
        let _ = next_id();
        ActorID(*Uuid::new_v4().as_bytes())
    }
}

impl Default for ActorID {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ActorID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Actor({})", &hex(&self.0)[..8])
    }
}

impl fmt::Display for ActorID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex(&self.0))
    }
}

/// Unique identifier for a task.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct TaskID([u8; 16]);

impl TaskID {
    pub fn new() -> Self {
        let _ = next_id();
        TaskID(*Uuid::new_v4().as_bytes())
    }
}

impl Default for TaskID {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TaskID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Task({})", &hex(&self.0)[..8])
    }
}

impl fmt::Display for TaskID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex(&self.0))
    }
}

/// A typed reference to an object stored in the object store.
///
/// This is the Rust analog of Ray's `ObjectRef`. It is cheap to clone and
/// can be passed across task/actor boundaries. The actual data lives in the
/// object store and is fetched on `get`.
///
/// Reference counting: each clone increments the refcount; each drop
/// decrements it. When the last ref is dropped, the object is evicted from
/// the store (mirrors Plasma's reference counting).
pub struct ObjectRef<T> {
    pub id: ObjectID,
    pub(crate) refcount: Arc<AtomicUsize>,
    pub(crate) store: Weak<crate::object_store::StoreInner>,
    _marker: PhantomData<T>,
}

impl<T> ObjectRef<T> {
    pub(crate) fn new(
        id: ObjectID,
        refcount: Arc<AtomicUsize>,
        store: Weak<crate::object_store::StoreInner>,
    ) -> Self {
        ObjectRef {
            id,
            refcount,
            store,
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for ObjectRef<T> {
    fn clone(&self) -> Self {
        self.refcount.fetch_add(1, Ordering::Relaxed);
        ObjectRef {
            id: self.id,
            refcount: self.refcount.clone(),
            store: self.store.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for ObjectRef<T> {
    fn drop(&mut self) {
        if self.refcount.fetch_sub(1, Ordering::Release) == 1 {
            // Last reference gone — evict from the store.
            crate::object_store::on_ref_dropped(&self.store, self.id);
        }
    }
}

impl<T> fmt::Debug for ObjectRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ref({:?})", self.id)
    }
}

impl<T> PartialEq for ObjectRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for ObjectRef<T> {}

impl<T> std::hash::Hash for ObjectRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Lifecycle state of an actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorState {
    Pending,
    Running,
    Idle,
    Dead,
}

/// State of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    Pending,
    Running,
    Finished,
    Failed,
}

/// Errors returned by the Crayon runtime.
#[derive(Debug, Clone)]
pub enum CrayonError {
    ObjectNotFound(ObjectID),
    ActorNotFound(ActorID),
    TypeMismatch,
    Serialize(String),
    TaskFailed(String),
    ActorDead(ActorID),
    Timeout(ObjectID),
}

impl fmt::Display for CrayonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CrayonError::ObjectNotFound(id) => write!(f, "object not found: {id}"),
            CrayonError::ActorNotFound(id) => write!(f, "actor not found: {id}"),
            CrayonError::TypeMismatch => write!(f, "object type mismatch"),
            CrayonError::Serialize(e) => write!(f, "serialization error: {e}"),
            CrayonError::TaskFailed(e) => write!(f, "task failed: {e}"),
            CrayonError::ActorDead(id) => write!(f, "actor dead: {id}"),
            CrayonError::Timeout(id) => write!(f, "timeout waiting for object: {id}"),
        }
    }
}

impl std::error::Error for CrayonError {}

/// Convert a caught panic payload into a [`CrayonError::TaskFailed`].
///
/// Shared by the scheduler and actor runtime — both catch panics via
/// `catch_unwind` and need to extract a human-readable message.
pub fn panic_to_error(p: Box<dyn std::any::Any + Send>, default_msg: &str) -> CrayonError {
    let msg = if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        default_msg.into()
    };
    CrayonError::TaskFailed(msg)
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
