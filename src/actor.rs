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

/// Default mailbox size for actors. When full, `call` awaits — this is the
/// backpressure that prevents a slow actor (e.g. a parameter server) from
/// being flooded by fast producers (e.g. rollout workers).
const ACTOR_MAILBOX_SIZE: usize = 256;

/// Type-erased actor handle internals. Stored in the named-actor registry so
/// that `get_actor` can look an actor up by name without knowing its state
/// type `S` at registration time.
pub struct ActorHandleInner {
    pub id: ActorID,
    tx: mpsc::Sender<Call>,
    store: ObjectStore,
    gcs: Gcs,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl ActorHandleInner {
    pub(crate) async fn send_call(&self, call: Call) -> Result<(), CrayonError> {
        if self.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(CrayonError::ActorDead(self.id));
        }
        self.tx
            .send(call)
            .await
            .map_err(|_| CrayonError::ActorDead(self.id))
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
        self.inner.send_call(Call { func: erased, reply }).await?;

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

    /// Kill the actor: signals the actor task to shut down and rejects new
    /// calls. In-flight calls will complete or error.
    pub fn kill(&self) {
        self.inner
            .shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.inner.gcs().set_actor_state(self.inner.id, ActorState::Dead);
    }
}

/// Spawn a new actor with the given initial state. If `max_restarts > 0`,
/// the actor will be restarted from its initial state on panic (Ray's
/// `max_restarts`). Each restart resets the state to the original value.
pub fn spawn_actor<S: Send + Clone + 'static>(
    name: &str,
    state: S,
    max_restarts: u32,
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

    let (tx, mut rx) = mpsc::channel::<Call>(ACTOR_MAILBOX_SIZE);
    let gcs_clone = gcs.clone();
    let initial_state = state.clone();
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();

    tokio::spawn(async move {
        let mut restarts = 0;
        let mut state: Box<dyn Any + Send> = Box::new(state);
        'restart: loop {
            while let Some(call) = rx.recv().await {
                if shutdown_clone.load(std::sync::atomic::Ordering::Relaxed) {
                    gcs_clone.set_actor_state(id, ActorState::Dead);
                    return;
                }
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
                        let _ = call.reply.send(bincode::serialize(&msg).unwrap());
                        if restarts < max_restarts {
                            // Reset to initial state and keep going.
                            restarts += 1;
                            state = Box::new(initial_state.clone());
                            gcs_clone.set_actor_state(id, ActorState::Running);
                            continue 'restart;
                        } else {
                            gcs_clone.set_actor_state(id, ActorState::Dead);
                            return;
                        }
                    }
                };
                let _ = call.reply.send(result);
            }
            // Mailbox closed — actor is done.
            gcs_clone.set_actor_state(id, ActorState::Dead);
            return;
        }
    });

    ActorHandle {
        inner: Arc::new(ActorHandleInner {
            id,
            tx,
            store,
            gcs,
            shutdown,
        }),
        _state: std::marker::PhantomData,
    }
}
