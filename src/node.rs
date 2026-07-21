//! Multi-node networking — Crayon's analog of Ray's raylet + GCS client.
//!
//! Each [`Node`] runs a TCP server. A head node tracks peers; worker nodes
//! register and receive the peer list. Objects are fetched from remote peers
//! via a pooled connection (see [`ConnectionPool`]).
//!
//! Wire format: 4-byte BE length + bincode [`Message`].

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::common::ObjectID;
use crate::object_store::ObjectStore;

/// Maximum allowed message size (256 MiB).
const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

/// Unique identifier for a node.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodeID(pub [u8; 16]);

impl NodeID {
    pub fn new() -> Self {
        NodeID(*uuid::Uuid::new_v4().as_bytes())
    }
}

impl Default for NodeID {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NodeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

/// Messages exchanged between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    RegisterNode { addr: String },
    NodeList(Vec<(NodeID, String)>),
    Ping { addr: String },
    Pong,
    GetObject(ObjectID),
    ObjectData(ObjectID, Vec<u8>),
    ObjectNotFound(ObjectID),
    /// Periodic GCS sync: object locations, task states, and actor metadata
    /// from the sender. Only carries serializable fields (no Instant timestamps).
    GcsSync {
        objects: Vec<(ObjectID, usize)>, // (id, size_bytes)
        tasks: Vec<(crate::common::TaskID, crate::common::TaskState)>,
        /// (id, name, state, owner_node) — lets peers discover named actors
        /// across nodes and route calls to the right node.
        actors: Vec<(
            crate::common::ActorID,
            String,
            crate::common::ActorState,
            Option<NodeID>,
        )>,
    },
    /// Invoke a registered method on a remote actor. The pooled connection
    /// guarantees one request/response per checkout, so no correlation ID
    /// is needed — the reply is always the next message.
    ActorCall {
        actor_id: crate::common::ActorID,
        method: String,
        args: Vec<u8>,
    },
    /// Reply to an [`Message::ActorCall`]. On success, carries the
    /// [`ObjectID`] of the method's result, stored in the owner node's object
    /// store — the caller fetches it via the existing remote-fetch path.
    ActorCallReply {
        result: Result<ObjectID, String>,
    },
}

type NodeError = Box<dyn std::error::Error + Send + Sync>;

/// A pooled TCP connection that can send/receive [`Message`]s.
struct Connection {
    stream: TcpStream,
}

impl Connection {
    async fn connect(addr: &str) -> Result<Self, NodeError> {
        Ok(Connection {
            stream: TcpStream::connect(addr).await?,
        })
    }

    async fn send(&mut self, msg: &Message) -> Result<(), NodeError> {
        let bytes = bincode::serialize(msg)?;
        let len = (bytes.len() as u32).to_be_bytes();
        self.stream.write_all(&len).await?;
        self.stream.write_all(&bytes).await?;
        Ok(())
    }

    async fn recv(&mut self) -> Result<Option<Message>, NodeError> {
        let mut len_buf = [0u8; 4];
        if self.stream.read_exact(&mut len_buf).await.is_err() {
            return Ok(None);
        }
        let len = u32::from_be_bytes(len_buf) as usize;
        if len > MAX_MESSAGE_SIZE {
            return Err(format!("message too large: {len} bytes (max {MAX_MESSAGE_SIZE})").into());
        }
        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf).await?;
        Ok(Some(bincode::deserialize(&buf)?))
    }
}

/// A simple LIFO connection pool keyed by address. Reusing connections avoids
/// the TCP handshake cost on every remote object fetch — critical for RL
/// workloads that pull many small samples.
///
/// Bounded per address to prevent file-descriptor leaks under high concurrency:
/// excess connections are dropped (closed) on return instead of cached forever.
#[derive(Default)]
struct ConnectionPool {
    inner: Mutex<HashMap<String, Vec<Connection>>>,
}

/// Max idle connections cached per peer address. Beyond this, returned
/// connections are closed to bound resource usage.
const MAX_CONNS_PER_ADDR: usize = 16;

impl ConnectionPool {
    fn new() -> Arc<Self> {
        Arc::new(ConnectionPool::default())
    }

    /// Get a connection from the pool, or open a new one.
    async fn get(self: &Arc<Self>, addr: &str) -> Result<Connection, NodeError> {
        if let Some(c) = self.inner.lock().get_mut(addr).and_then(|v| v.pop()) {
            return Ok(c);
        }
        Connection::connect(addr).await
    }

    /// Return a connection to the pool for reuse. If the pool for this address
    /// is full, the connection is dropped (closed) instead of cached.
    fn put(&self, addr: &str, conn: Connection) {
        let mut inner = self.inner.lock();
        let v = inner.entry(addr.to_string()).or_default();
        if v.len() < MAX_CONNS_PER_ADDR {
            v.push(conn);
        }
        // else: conn drops here, closing the socket
    }
}

/// A remote node's address + last-seen timestamp (for heartbeat eviction).
struct RemoteNode {
    addr: String,
    last_seen: std::time::Instant,
}

/// The local node's networking state.
#[derive(Clone)]
pub struct Node {
    pub id: NodeID,
    pub addr: String,
    store: ObjectStore,
    peers: Arc<Mutex<HashMap<NodeID, RemoteNode>>>,
    pool: Arc<ConnectionPool>,
    /// Optional GCS for metadata sync. When attached, the node periodically
    /// broadcasts object/task metadata to peers and merges incoming metadata.
    gcs: Arc<Mutex<Option<crate::gcs::Gcs>>>,
    /// Locally-hosted actors, keyed by ID. Used to route incoming
    /// [`Message::ActorCall`] messages to the right actor task.
    local_actors: Arc<Mutex<HashMap<crate::common::ActorID, Arc<crate::actor::ActorHandleInner>>>>,
}

impl Node {
    /// Start a node. If `head_addr` is `None`, this is the head node.
    pub async fn start(
        addr: &str,
        head_addr: Option<&str>,
        store: ObjectStore,
    ) -> Result<Arc<Self>, NodeError> {
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?.to_string();

        let node = Arc::new(Node {
            id: NodeID::new(),
            addr: local_addr.clone(),
            store,
            peers: Arc::new(Mutex::new(HashMap::new())),
            pool: ConnectionPool::new(),
            gcs: Arc::new(Mutex::new(None)),
            local_actors: Arc::new(Mutex::new(HashMap::new())),
        });

        // Accept loop
        let node_clone = node.clone();
        tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let n = node_clone.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, n).await {
                            tracing::warn!("connection error: {e}");
                        }
                    });
                }
            }
        });

        // Worker: register with head + send heartbeats
        if let Some(head) = head_addr {
            let mut conn = Connection::connect(head).await?;
            conn.send(&Message::RegisterNode {
                addr: local_addr.clone(),
            })
            .await?;
            if let Some(Message::NodeList(peers)) = conn.recv().await? {
                let mut p = node.peers.lock();
                for (pid, paddr) in peers {
                    // Filter out ourselves by address (head assigns us a new
                    // NodeID we don't know about, so we can't filter by ID).
                    if paddr != node.addr {
                        p.insert(
                            pid,
                            RemoteNode {
                                addr: paddr,
                                last_seen: std::time::Instant::now(),
                            },
                        );
                    }
                }
            }
            // Don't pool the registration connection — the head's
            // handle_connection owns it and will close it on disconnect.
            drop(conn);

            let head = head.to_string();
            let my_addr = local_addr;
            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(std::time::Duration::from_secs(5));
                loop {
                    ticker.tick().await;
                    // Use a fresh connection for heartbeats — don't pollute the
                    // fetch pool with Ping/Pong traffic.
                    if let Ok(mut conn) = Connection::connect(&head).await {
                        let _ = conn
                            .send(&Message::Ping {
                                addr: my_addr.clone(),
                            })
                            .await;
                    }
                }
            });
        }

        // Evict dead peers
        let node_clone = node.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(10));
            loop {
                ticker.tick().await;
                let deadline = std::time::Instant::now() - std::time::Duration::from_secs(15);
                node_clone
                    .peers
                    .lock()
                    .retain(|_, p| p.last_seen > deadline);
            }
        });

        Ok(node)
    }

    /// Attach a GCS for metadata sync. The node will periodically broadcast
    /// object/task metadata to peers and merge incoming metadata.
    pub fn with_gcs(self: &Arc<Self>, gcs: crate::gcs::Gcs) -> Arc<Self> {
        *self.gcs.lock() = Some(gcs.clone());

        // Periodically broadcast GCS state to all peers.
        let node = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(2));
            loop {
                ticker.tick().await;
                node.broadcast_gcs().await;
            }
        });

        self.clone()
    }

    /// Register a locally-hosted actor so incoming [`Message::ActorCall`]
    /// messages can be routed to it. Triggers an immediate GCS broadcast so
    /// peers discover the new actor without waiting for the next periodic sync.
    pub fn register_actor(&self, inner: Arc<crate::actor::ActorHandleInner>) {
        self.local_actors.lock().insert(inner.id, inner);
        // Kick off a broadcast right away so peers learn about this actor
        // without waiting up to 2s for the next periodic tick.
        let node = self.clone();
        tokio::spawn(async move {
            node.broadcast_gcs().await;
        });
    }

    /// Invoke a registered method on a remote actor. Sends an
    /// [`Message::ActorCall`] to the actor's owner node and waits for the reply,
    /// which carries the [`ObjectID`] of the result stored on the owner node.
    pub async fn remote_actor_call(
        &self,
        actor_id: crate::common::ActorID,
        owner: NodeID,
        method: &str,
        args: Vec<u8>,
    ) -> Result<ObjectID, NodeError> {
        let addr = self
            .peers
            .lock()
            .get(&owner)
            .map(|p| p.addr.clone())
            .ok_or_else(|| format!("no peer with node id {owner}"))?;

        let mut conn = self.pool.get(&addr).await?;
        conn.send(&Message::ActorCall {
            actor_id,
            method: method.to_string(),
            args,
        })
        .await?;
        // Pool guarantees one request/response per checkout — the next message
        // is always our reply.
        let reply = conn.recv().await?;
        self.pool.put(&addr, conn);
        match reply {
            Some(Message::ActorCallReply { result }) => result.map_err(|e| e.into()),
            _ => Err("unexpected reply to actor call".into()),
        }
    }
    async fn broadcast_gcs(&self) {
        let gcs = match self.gcs.lock().clone() {
            Some(g) => g,
            None => return,
        };
        let objects: Vec<_> = gcs
            .objects()
            .into_iter()
            .map(|o| (o.id, o.size_bytes))
            .collect();
        let tasks: Vec<_> = gcs
            .tasks()
            .into_iter()
            .map(|t| (t.id, t.state))
            .collect();
        let actors: Vec<_> = gcs
            .actors()
            .into_iter()
            .map(|a| (a.id, a.name, a.state, a.owner_node))
            .collect();
        let msg = Message::GcsSync {
            objects,
            tasks,
            actors,
        };
        let peers: Vec<_> = self.peers.lock().values().map(|p| p.addr.clone()).collect();
        for addr in peers {
            let pool = self.pool.clone();
            let msg = msg.clone();
            tokio::spawn(async move {
                if let Ok(mut conn) = pool.get(&addr).await {
                    let _ = conn.send(&msg).await;
                    pool.put(&addr, conn);
                }
            });
        }
    }

    /// Fetch an object from a remote node by ID. Queries all peers concurrently
    /// and returns the first successful result. This avoids the latency of
    /// trying peers one-by-one when some are slow or unreachable.
    pub async fn fetch_remote_object(&self, id: ObjectID) -> Result<Vec<u8>, NodeError> {
        let peers: Vec<_> = self.peers.lock().values().map(|p| p.addr.clone()).collect();
        let pool = self.pool.clone();

        let futs: Vec<_> = peers
            .into_iter()
            .map(|addr| {
                let pool = pool.clone();
                async move {
                    let mut conn = pool.get(&addr).await?;
                    conn.send(&Message::GetObject(id)).await?;
                    match conn.recv().await? {
                        Some(Message::ObjectData(_, bytes)) => {
                            pool.put(&addr, conn);
                            Ok::<_, NodeError>(bytes)
                        }
                        _ => Err("object not found on peer".into()),
                    }
                }
            })
            .collect();

        // Run all fetches concurrently; return the first success.
        let results = futures::future::join_all(futs).await;
        results
            .into_iter()
            .find_map(|r| r.ok())
            .ok_or_else(|| "object not found on any peer".into())
    }

    pub fn peers(&self) -> Vec<(NodeID, String)> {
        self.peers
            .lock()
            .iter()
            .map(|(id, p)| (*id, p.addr.clone()))
            .collect()
    }
}

impl crate::object_store::RemoteFetcher for Node {
    fn fetch_remote(
        &self,
        id: ObjectID,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>,
                > + Send,
        >,
    > {
        let this = self.clone();
        Box::pin(async move { this.fetch_remote_object(id).await })
    }
}

async fn handle_connection(stream: TcpStream, node: Arc<Node>) -> Result<(), NodeError> {
    let mut conn = Connection { stream };
    while let Some(msg) = conn.recv().await? {
        match msg {
            Message::RegisterNode { addr } => {
                node.peers.lock().insert(
                    NodeID::new(),
                    RemoteNode {
                        addr: addr.clone(),
                        last_seen: std::time::Instant::now(),
                    },
                );
                let mut peers: Vec<_> = node
                    .peers
                    .lock()
                    .iter()
                    .map(|(id, p)| (*id, p.addr.clone()))
                    .collect();
                peers.push((node.id, node.addr.clone()));
                conn.send(&Message::NodeList(peers)).await?;
            }
            Message::Ping { addr } => {
                let now = std::time::Instant::now();
                for p in node.peers.lock().values_mut() {
                    if p.addr == addr {
                        p.last_seen = now;
                    }
                }
                conn.send(&Message::Pong).await?;
            }
            Message::GetObject(id) => {
                if node.store.contains(id) {
                    let bytes = node.store.get_bytes(id).await?;
                    conn.send(&Message::ObjectData(id, bytes.to_vec())).await?;
                } else {
                    conn.send(&Message::ObjectNotFound(id)).await?;
                }
            }
            Message::GcsSync {
                objects,
                tasks,
                actors,
            } => {
                // Merge peer's metadata into our GCS so we know object
                // locations, task states, and actor locations across the cluster.
                if let Some(gcs) = node.gcs.lock().clone() {
                    for (id, size_bytes) in objects {
                        gcs.add_object(crate::gcs::ObjectMeta {
                            id,
                            size_bytes,
                            created_at: std::time::Instant::now(),
                            owner: None,
                        });
                    }
                    for (id, state) in tasks {
                        gcs.add_task(crate::gcs::TaskMeta {
                            id,
                            state,
                            created_at: std::time::Instant::now(),
                            finished_at: None,
                            output: None,
                        });
                    }
                    for (id, name, state, owner_node) in actors {
                        // Upsert: insert if new (with owner_node for routing),
                        // otherwise just update state so we don't clobber local
                        // task counters.
                        if gcs.get_actor(id).is_some() {
                            gcs.set_actor_state(id, state);
                        } else {
                            gcs.add_actor(crate::gcs::ActorMeta {
                                id,
                                name,
                                state,
                                owner_node,
                                created_at: std::time::Instant::now(),
                                pending_tasks: 0,
                                completed_tasks: 0,
                            });
                        }
                    }
                }
            }
            Message::ActorCall {
                actor_id,
                method,
                args,
            } => {
                // Route the incoming call to the local actor task. Store the
                // result in the object store so the caller can fetch it via the
                // existing remote-fetch path (same as task outputs).
                let actor = node.local_actors.lock().get(&actor_id).cloned();
                let reply = match actor {
                    Some(actor) => match actor.call_method(&method, args).await {
                        Ok(bytes) => {
                            let id = crate::common::ObjectID::new();
                            node.store.put_bytes(id, bytes, None);
                            Ok(id)
                        }
                        Err(e) => Err(e.to_string()),
                    },
                    None => Err(format!("actor {actor_id} not found on this node")),
                };
                conn.send(&Message::ActorCallReply { result: reply })
                    .await?;
            }
            _ => {}
        }
    }
    Ok(())
}
