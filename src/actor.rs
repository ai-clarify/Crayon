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
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

use crate::common::{ActorID, ActorState, CrayonError, ObjectID};
use crate::gcs::Gcs;
use crate::node::NodeID;
use crate::object_store::ObjectStore;

/// A type-erased actor method: takes the actor's state (as `&mut dyn Any`),
/// returns the serialized result.
pub(crate) type ActorMethod = Box<dyn FnOnce(&mut dyn Any) -> Vec<u8> + Send>;

/// A registered named method: takes state + serialized args, returns serialized
/// result. Used for cross-node actor calls (closures can't be sent over the
/// network, so remote callers address methods by name).
type MethodFn = Arc<dyn Fn(&mut dyn Any, Vec<u8>) -> Vec<u8> + Send + Sync>;

pub(crate) struct Call {
    func: ActorMethod,
    reply: oneshot::Sender<Result<Vec<u8>, CrayonError>>,
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
    /// Registered named methods for cross-node calls.
    methods: Arc<Mutex<HashMap<String, MethodFn>>>,
    /// The node that owns this actor. `None` means local (same process).
    pub owner_node: Option<NodeID>,
    /// The local node, used to route calls to remote actors.
    pub node: Option<Arc<crate::node::Node>>,
}

impl ActorHandleInner {
    /// Build a proxy handle for an actor that lives on a remote node. The
    /// mailbox is never used — `call_named` routes through `node` instead.
    pub(crate) fn remote_proxy(
        id: ActorID,
        owner_node: NodeID,
        node: Arc<crate::node::Node>,
        store: ObjectStore,
        gcs: Gcs,
    ) -> Arc<Self> {
        let (tx, _rx) = mpsc::channel::<Call>(1);
        Arc::new(ActorHandleInner {
            id,
            tx,
            store,
            gcs,
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            methods: Arc::new(Mutex::new(HashMap::new())),
            owner_node: Some(owner_node),
            node: Some(node),
        })
    }

    pub(crate) async fn send_call(&self, call: Call) -> Result<(), CrayonError> {
        if self.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(CrayonError::ActorDead(self.id));
        }
        self.tx
            .send(call)
            .await
            .map_err(|_| CrayonError::ActorDead(self.id))
    }

    /// Signal the actor to shut down. New calls will fail with ActorDead.
    pub(crate) fn shutdown(&self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn store(&self) -> &ObjectStore {
        &self.store
    }

    pub(crate) fn gcs(&self) -> &Gcs {
        &self.gcs
    }

    /// Register a named method so it can be called from remote nodes.
    pub(crate) fn register_method(&self, name: &str, func: MethodFn) {
        self.methods.lock().insert(name.to_string(), func);
    }

    /// Invoke a registered method by name (called by the node when it receives
    /// a remote `ActorCall` message).
    pub(crate) async fn call_method(
        &self,
        name: &str,
        args: Vec<u8>,
    ) -> Result<Vec<u8>, CrayonError> {
        if self.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(CrayonError::ActorDead(self.id));
        }
        let func = self
            .methods
            .lock()
            .get(name)
            .cloned()
            .ok_or_else(|| CrayonError::ActorMethodNotFound(self.id, name.to_string()))?;
        let (reply, rx) = oneshot::channel();
        let call = Call {
            func: Box::new(move |state| func(state, args)),
            reply,
        };
        self.send_call(call).await?;
        rx.await.map_err(|_| CrayonError::ActorDead(self.id))?
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

        let erased: ActorMethod = Box::new(move |state: &mut dyn Any| {
            let s = state
                .downcast_mut::<S>()
                .expect("actor state type mismatch");
            bincode::serialize(&func(s)).expect("serialize actor result")
        });

        let (reply, rx) = oneshot::channel();
        self.inner
            .send_call(Call {
                func: erased,
                reply,
            })
            .await?;

        let store = self.inner.store().clone();
        let gcs = self.inner.gcs().clone();
        let id = self.inner.id;
        tokio::spawn(async move {
            match rx.await {
                Ok(Ok(bytes)) => {
                    store.put_bytes(output_id, bytes, None);
                    gcs.record_actor_task(id, true);
                }
                Ok(Err(e)) => {
                    store.put_error(output_id, e);
                    gcs.record_actor_task(id, true);
                }
                Err(_) => {
                    // Actor died (mailbox closed). Surface the error to the caller
                    // instead of leaving them hanging on `get()`.
                    store.put_error(output_id, CrayonError::ActorDead(id));
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
        self.inner
            .gcs()
            .set_actor_state(self.inner.id, ActorState::Dead);
    }

    /// Register a named method so it can be invoked from remote nodes via
    /// [`ActorHandle::call_remote`]. The closure receives serialized args and
    /// returns serialized bytes — this is the cross-network calling convention
    /// (closures can't be sent over the wire).
    pub fn register_method<F>(&self, name: &str, func: F)
    where
        F: Fn(&mut S, Vec<u8>) -> Vec<u8> + Send + Sync + 'static,
    {
        let erased: MethodFn = Arc::new(move |state: &mut dyn Any, args| {
            let s = state
                .downcast_mut::<S>()
                .expect("actor state type mismatch");
            func(s, args)
        });
        self.inner.register_method(name, erased);
    }

    /// The node that owns this actor, if it lives on a remote node.
    pub fn owner_node(&self) -> Option<NodeID> {
        self.inner.owner_node
    }

    /// Invoke a registered named method on the actor. Automatically routes to
    /// the actor's owner node if it lives on a remote node (Ray's cross-node
    /// actor call pattern). Returns the serialized result bytes.
    ///
    /// Use [`ActorHandle::register_method`] on the owner node to expose methods,
    /// and [`ActorHandle::call_named`] on any node to invoke them.
    pub async fn call_named(&self, method: &str, args: Vec<u8>) -> Result<Vec<u8>, CrayonError> {
        match (self.inner.owner_node, self.inner.node.clone()) {
            // Remote actor: forward via the local node to the owner node.
            (Some(owner), Some(node)) => node
                .remote_actor_call(self.inner.id, owner, method, args)
                .await
                .map_err(|e| CrayonError::TaskFailed(e.to_string())),
            // Local actor (or no node attached): call directly through the mailbox.
            _ => self.inner.call_method(method, args).await,
        }
    }
}

/// Spawn a new actor with the given initial state. If `max_restarts > 0`,
/// the actor will be restarted from its initial state on panic (Ray's
/// `max_restarts`). Each restart resets the state to the original value.
///
/// `owner_node` identifies the node hosting this actor (for cross-node
/// discovery). Pass `None` for local-only actors.
pub fn spawn_actor<S: Send + Clone + 'static>(
    name: &str,
    state: S,
    max_restarts: u32,
    gcs: Gcs,
    store: ObjectStore,
    owner_node: Option<NodeID>,
    node: Option<Arc<crate::node::Node>>,
) -> ActorHandle<S> {
    let id = ActorID::new();
    gcs.add_actor(crate::gcs::ActorMeta {
        id,
        name: name.to_string(),
        state: ActorState::Running,
        owner_node,
        created_at: Instant::now(),
        pending_tasks: 0,
        completed_tasks: 0,
    });

    let (tx, mut rx) = mpsc::channel::<Call>(ACTOR_MAILBOX_SIZE);
    let gcs_clone = gcs.clone();
    let initial_state = state.clone();
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    let methods = Arc::new(Mutex::new(HashMap::new()));

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
                    Ok(v) => Ok(v),
                    Err(p) => {
                        let _ = call.reply.send(Err(crate::common::panic_to_error(
                            p,
                            "actor method panicked",
                        )));
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
            methods,
            owner_node,
            node,
        }),
        _state: std::marker::PhantomData,
    }
}
