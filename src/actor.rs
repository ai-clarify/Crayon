//! The actor runtime — Crayon's analog of Ray actors.
//!
//! A Ray actor is a stateful process that runs methods serially. Here an actor
//! is a tokio task that owns its state and processes method calls from a
//! mpsc mailbox one at a time. Method calls are closures that borrow the actor
//! state mutably and return a boxed result, which is sent back to the caller.
//!
//! This mirrors Ray's concurrency model: each actor has a single execution
//! thread (by default) so its state is never accessed concurrently.

use std::any::Any;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

use crate::common::{ActorID, ActorState, CrayonError, ObjectID};
use crate::gcs::Gcs;
use crate::object_store::ObjectStore;

pub(crate) struct Call {
    func: Box<dyn FnOnce(&mut dyn Any) -> Vec<u8> + Send>,
    reply: oneshot::Sender<Vec<u8>>,
}

/// Type-erased actor handle internals. Stored in the named-actor registry so
/// that `get_actor` can look an actor up by name without knowing its state
/// type `S` at registration time.
pub struct ActorHandleInner {
    pub id: ActorID,
    tx: mpsc::UnboundedSender<Call>,
    store: ObjectStore,
    gcs: Gcs,
}

impl ActorHandleInner {
    pub(crate) fn send_call(&self, call: Call) -> Result<(), CrayonError> {
        self.tx.send(call).map_err(|_| CrayonError::ActorDead(self.id))
    }

    pub(crate) fn store(&self) -> &ObjectStore {
        &self.store
    }

    pub(crate) fn gcs(&self) -> &Gcs {
        &self.gcs
    }
}

/// A handle to a running actor. Cheap to clone.
pub struct ActorHandle<S: Send + 'static> {
    pub inner: Arc<ActorHandleInner>,
    _state: std::marker::PhantomData<S>,
}

impl<S: Send + 'static> Clone for ActorHandle<S> {
    fn clone(&self) -> Self {
        ActorHandle {
            inner: self.inner.clone(),
            _state: std::marker::PhantomData,
        }
    }
}

impl<S: Send + 'static> ActorHandle<S> {
    /// Reconstruct a typed handle from a type-erased inner (used by the
    /// named-actor registry).
    pub(crate) fn from_inner(inner: Arc<ActorHandleInner>) -> Self {
        ActorHandle {
            inner,
            _state: std::marker::PhantomData,
        }
    }

    pub fn id(&self) -> ActorID {
        self.inner.id
    }

    /// Call a method on the actor. The closure runs on the actor's task,
    /// borrowing its state mutably. Returns a refcounted [`ObjectRef`] for the
    /// result.
    pub async fn call<T, F>(&self, func: F) -> Result<crate::common::ObjectRef<T>, CrayonError>
    where
        T: serde::Serialize + Send + 'static,
        F: FnOnce(&mut S) -> T + Send + 'static,
    {
        let output_id = ObjectID::new();
        let r = self.inner.store().reserve_ref(output_id);
        self.inner.gcs().record_actor_task(self.inner.id, false);

        let erased: Box<dyn FnOnce(&mut dyn Any) -> Vec<u8> + Send> = Box::new(move |state: &mut dyn Any| {
            let s = state
                .downcast_mut::<S>()
                .expect("actor state type mismatch");
            bincode::serialize(&func(s)).expect("serialize actor result")
        });

        let (reply, rx) = oneshot::channel();
        self.inner.send_call(Call { func: erased, reply })?;

        let store = self.inner.store().clone();
        let gcs = self.inner.gcs().clone();
        let id = self.inner.id;
        tokio::spawn(async move {
            match rx.await {
                Ok(bytes) => {
                    store.put_bytes(output_id, bytes, None);
                    gcs.record_actor_task(id, true);
                }
                Err(_) => {
                    // ponytail: surface a proper error to the caller.
                }
            }
        });

        Ok(r)
    }

    /// Kill the actor (drops the mailbox). In-flight calls will error.
    pub fn kill(&self) {
        self.inner.gcs().set_actor_state(self.inner.id, ActorState::Dead);
    }
}

/// Spawn a new actor with the given initial state.
pub fn spawn_actor<S: Send + 'static>(
    name: &str,
    state: S,
    gcs: Gcs,
    store: ObjectStore,
) -> ActorHandle<S> {
    let id = ActorID::new();
    gcs.add_actor(crate::gcs::ActorMeta {
        id,
        name: name.to_string(),
        state: ActorState::Running,
        created_at: Instant::now(),
        pending_tasks: 0,
        completed_tasks: 0,
    });

    let (tx, mut rx) = mpsc::unbounded_channel::<Call>();
    let gcs_clone = gcs.clone();

    tokio::spawn(async move {
        let mut state: Box<dyn Any + Send> = Box::new(state);
        while let Some(call) = rx.recv().await {
            gcs_clone.set_actor_state(id, ActorState::Running);
            let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (call.func)(state.as_mut())
            })) {
                Ok(v) => v,
                Err(p) => {
                    let msg = if let Some(s) = p.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = p.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "actor method panicked".into()
                    };
                    bincode::serialize(&msg).unwrap()
                }
            };
            let _ = call.reply.send(result);
        }
        gcs_clone.set_actor_state(id, ActorState::Dead);
    });

    ActorHandle {
        inner: Arc::new(ActorHandleInner { id, tx, store, gcs }),
        _state: std::marker::PhantomData,
    }
}
