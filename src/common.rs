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

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

macro_rules! id_type {
    ($name:ident, $label:literal) => {
        #[derive(Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
        pub struct $name([u8; 16]);

        impl $name {
            pub fn new() -> Self {
                let _ = next_id();
                Self(*Uuid::new_v4().as_bytes())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($label, "({})"), &hex(&self.0)[..8])
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", hex(&self.0))
            }
        }
    };
}

id_type!(ObjectID, "Obj");
id_type!(ActorID, "Actor");
id_type!(TaskID, "Task");

impl ObjectID {
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// A typed reference to an object stored in the object store.
pub struct ObjectRef<T> {
    pub id: ObjectID,
    pub(crate) generation: u64,
    pub(crate) refcount: Arc<AtomicUsize>,
    pub(crate) store: Weak<crate::object_store::StoreInner>,
    pub(crate) task_id: Option<TaskID>,
    _marker: PhantomData<T>,
}

impl<T> ObjectRef<T> {
    pub(crate) fn new(
        id: ObjectID,
        generation: u64,
        refcount: Arc<AtomicUsize>,
        store: Weak<crate::object_store::StoreInner>,
        task_id: Option<TaskID>,
    ) -> Self {
        Self {
            id,
            generation,
            refcount,
            store,
            task_id,
            _marker: PhantomData,
        }
    }

    /// The producing task, if this reference came from `Ray::spawn`.
    pub fn task_id(&self) -> Option<TaskID> {
        self.task_id
    }
}

impl<T> Clone for ObjectRef<T> {
    fn clone(&self) -> Self {
        self.refcount.fetch_add(1, Ordering::Relaxed);
        Self {
            id: self.id,
            generation: self.generation,
            refcount: self.refcount.clone(),
            store: self.store.clone(),
            task_id: self.task_id,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for ObjectRef<T> {
    fn drop(&mut self) {
        if self.refcount.fetch_sub(1, Ordering::AcqRel) == 1 {
            crate::object_store::on_ref_dropped(
                &self.store,
                self.id,
                self.generation,
                &self.refcount,
            );
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
        self.id == other.id && self.generation == other.generation
    }
}
impl<T> Eq for ObjectRef<T> {}
impl<T> std::hash::Hash for ObjectRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.generation.hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorState {
    Pending,
    Running,
    Idle,
    Dead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    Pending,
    Running,
    Finished,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum CrayonError {
    ObjectNotFound(ObjectID),
    ActorNotFound(ActorID),
    TypeMismatch,
    Serialize(String),
    TaskFailed(String),
    TaskCancelled(TaskID),
    ActorDead(ActorID),
    ActorMethodNotFound(ActorID, String),
    Timeout(ObjectID),
}

impl fmt::Display for CrayonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObjectNotFound(id) => write!(f, "object not found: {id}"),
            Self::ActorNotFound(id) => write!(f, "actor not found: {id}"),
            Self::TypeMismatch => write!(f, "object type mismatch"),
            Self::Serialize(e) => write!(f, "serialization error: {e}"),
            Self::TaskFailed(e) => write!(f, "task failed: {e}"),
            Self::TaskCancelled(id) => write!(f, "task cancelled: {id}"),
            Self::ActorDead(id) => write!(f, "actor dead: {id}"),
            Self::ActorMethodNotFound(id, name) => {
                write!(f, "actor {id} has no registered method '{name}'")
            }
            Self::Timeout(id) => write!(f, "timeout waiting for object: {id}"),
        }
    }
}

impl std::error::Error for CrayonError {}

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
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}
