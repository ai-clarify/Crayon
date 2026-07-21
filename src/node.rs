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

use crate::common::{ActorID, ObjectID};
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
#[derive(Debug, Serialize, Deserialize)]
pub enum Message {
    RegisterNode { addr: String },
    NodeList(Vec<(NodeID, String)>),
    Ping { addr: String },
    Pong,
    GetObject(ObjectID),
    ObjectData(ObjectID, Vec<u8>),
    ObjectNotFound(ObjectID),
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
#[derive(Default)]
struct ConnectionPool {
    inner: Mutex<HashMap<String, Vec<Connection>>>,
}

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

    /// Return a connection to the pool for reuse.
    fn put(&self, addr: &str, conn: Connection) {
        self.inner
            .lock()
            .entry(addr.to_string())
            .or_default()
            .push(conn);
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

async fn handle_connection(mut stream: TcpStream, node: Arc<Node>) -> Result<(), NodeError> {
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
            _ => {}
        }
    }
    Ok(())
}
