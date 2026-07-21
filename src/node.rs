//! Multi-node networking — Crayon's analog of Ray's raylet + GCS client.
//!
//! Each [`Node`] runs a TCP server. A head node hosts the GCS; worker nodes
//! connect to the head and register themselves. Objects can be fetched from
//! remote nodes; tasks can be submitted to remote workers.
//!
//! Wire format: 4-byte big-endian length prefix + bincode-serialized
//! [`Message`]. Simple and robust.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

use crate::common::{ActorID, ObjectID};
use crate::object_store::ObjectStore;

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
    // ---- node lifecycle ----
    RegisterNode { addr: String },
    NodeList(Vec<(NodeID, String)>),
    Ping { addr: String },
    Pong,
    // ---- object store ----
    GetObject(ObjectID),
    ObjectData(ObjectID, Vec<u8>),
    ObjectNotFound(ObjectID),
    // ---- task submission ----
    SubmitTask {
        id: [u8; 16],
        output_id: ObjectID,
        resources: (f64, f64), // (cpu, gpu)
        func_bytes: Vec<u8>,
    },
    TaskAck,
    // ---- actor ----
    CreateActor {
        name: String,
        state_bytes: Vec<u8>,
    },
    ActorCall {
        actor_id: ActorID,
        method_bytes: Vec<u8>,
    },
}

/// A remote node connection.
struct RemoteNode {
    addr: String,
    last_seen: std::time::Instant,
}

/// The local node's networking state.
pub struct Node {
    pub id: NodeID,
    pub addr: String,
    store: ObjectStore,
    peers: Arc<Mutex<HashMap<NodeID, RemoteNode>>>,
    head_addr: Option<String>,
    pending: Arc<Mutex<HashMap<ObjectID, oneshot::Sender<Vec<u8>>>>>,
}

impl Node {
    /// Start a node. If `head_addr` is `None`, this is the head node (hosts
    /// the GCS). Otherwise, connect to the head and register.
    pub async fn start(
        addr: &str,
        head_addr: Option<&str>,
        store: ObjectStore,
    ) -> Result<Arc<Self>, NodeError> {
        // Start TCP server first to get the actual bound address
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?.to_string();

        let id = NodeID::new();
        let node = Arc::new(Node {
            id,
            addr: local_addr.clone(),
            store,
            peers: Arc::new(Mutex::new(HashMap::new())),
            head_addr: head_addr.map(|s| s.to_string()),
            pending: Arc::new(Mutex::new(HashMap::new())),
        });

        let node_clone = node.clone();
        tokio::spawn(async move {
            loop {
                if let Ok((stream, _)) = listener.accept().await {
                    let n = node_clone.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, n).await {
                            eprintln!("connection error: {e}");
                        }
                    });
                }
            }
        });

        // If worker, register with head
        let my_addr = local_addr.clone();
        if let Some(head) = head_addr {
            let mut stream = TcpStream::connect(head).await?;
            send_message(
                &mut stream,
                &Message::RegisterNode {
                    addr: local_addr,
                },
            )
            .await?;
            // Receive peer list (includes head's address)
            if let Some(Message::NodeList(peers)) = recv_message(&mut stream).await? {
                let mut p = node.peers.lock();
                for (pid, paddr) in peers {
                    if pid != node.id {
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

            // Worker: send periodic heartbeats to head
            let head = head.to_string();
            let node_clone = node.clone();
            let my_addr_clone = my_addr.clone();
            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(std::time::Duration::from_secs(5));
                loop {
                    ticker.tick().await;
                    if let Ok(mut stream) = TcpStream::connect(&head).await {
                        let _ = send_message(
                            &mut stream,
                            &Message::Ping {
                                addr: my_addr_clone.clone(),
                            },
                        )
                        .await;
                    }
                    // If head is unreachable, peer will be cleaned up by head's check
                    let _ = node_clone; // keep node alive
                }
            });
        }

        // Head (and everyone): periodically evict dead peers
        let node_clone = node.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(10));
            loop {
                ticker.tick().await;
                let deadline = std::time::Instant::now() - std::time::Duration::from_secs(15);
                node_clone.peers.lock().retain(|_, p| p.last_seen > deadline);
            }
        });

        Ok(node)
    }

    /// Fetch an object from a remote node by ID. Tries all peers.
    pub async fn fetch_remote_object(
        &self,
        id: ObjectID,
    ) -> Result<Vec<u8>, NodeError> {
        let peers: Vec<_> = self.peers.lock().values().map(|p| p.addr.clone()).collect();
        for addr in peers {
            if let Ok(mut stream) = TcpStream::connect(&addr).await {
                send_message(&mut stream, &Message::GetObject(id)).await?;
                match recv_message(&mut stream).await? {
                    Some(Message::ObjectData(_, bytes)) => return Ok(bytes),
                    _ => continue,
                }
            }
        }
        Err("object not found on any peer".into())
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
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>> + Send>>
    {
        let peers: Vec<_> = self.peers.lock().values().map(|p| p.addr.clone()).collect();
        Box::pin(async move {
            for addr in peers {
                if let Ok(mut stream) = TcpStream::connect(&addr).await {
                    let _ = send_message(&mut stream, &Message::GetObject(id)).await;
                    if let Ok(Some(Message::ObjectData(_, bytes))) = recv_message(&mut stream).await
                    {
                        return Ok(bytes);
                    }
                }
            }
            Err("object not found on any peer".into())
        })
    }
}

type NodeError = Box<dyn std::error::Error + Send + Sync>;

async fn handle_connection(
    mut stream: TcpStream,
    node: Arc<Node>,
) -> Result<(), NodeError> {
    while let Some(msg) = recv_message(&mut stream).await? {
        match msg {
            Message::RegisterNode { addr } => {
                let peer_id = NodeID::new();
                node.peers.lock().insert(
                    peer_id,
                    RemoteNode {
                        addr: addr.clone(),
                        last_seen: std::time::Instant::now(),
                    },
                );
                // Send back peer list, including this node (the head) so the
                // worker knows where to fetch objects from.
                let mut peers: Vec<_> = node
                    .peers
                    .lock()
                    .iter()
                    .map(|(id, p)| (*id, p.addr.clone()))
                    .collect();
                peers.push((node.id, node.addr.clone()));
                send_message(&mut stream, &Message::NodeList(peers)).await?;
            }
            Message::Ping { addr } => {
                // Update last_seen for the peer with this addr
                let now = std::time::Instant::now();
                {
                    let mut peers = node.peers.lock();
                    for p in peers.values_mut() {
                        if p.addr == addr {
                            p.last_seen = now;
                        }
                    }
                }
                send_message(&mut stream, &Message::Pong).await?;
            }
            Message::GetObject(id) => {
                if node.store.contains(id) {
                    let bytes = node.store.get_bytes(id).await?;
                    send_message(&mut stream, &Message::ObjectData(id, bytes)).await?;
                } else {
                    send_message(&mut stream, &Message::ObjectNotFound(id)).await?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

async fn send_message(
    stream: &mut TcpStream,
    msg: &Message,
) -> Result<(), NodeError> {
    let bytes = bincode::serialize(msg)?;
    let len = (bytes.len() as u32).to_be_bytes();
    stream.write_all(&len).await?;
    stream.write_all(&bytes).await?;
    Ok(())
}

/// Maximum allowed message size (256 MiB). Protects against corrupted length
/// prefixes that would otherwise cause a huge allocation (OOM).
const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

async fn recv_message(
    stream: &mut TcpStream,
) -> Result<Option<Message>, NodeError> {
    let mut len_buf = [0u8; 4];
    if stream.read_exact(&mut len_buf).await.is_err() {
        return Ok(None);
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_MESSAGE_SIZE {
        return Err(format!("message too large: {len} bytes (max {MAX_MESSAGE_SIZE})").into());
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(Some(bincode::deserialize(&buf)?))
}
