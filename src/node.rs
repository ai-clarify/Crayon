//! Multi-node networking for object transfer, membership, and actor calls.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::stream::{FuturesUnordered, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::common::ObjectID;
use crate::object_store::ObjectStore;

const MAX_MESSAGE_SIZE: usize = 8 * 1024 * 1024 * 1024;
const CHUNK_SIZE: usize = 1024 * 1024;
const RPC_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_CONNS_PER_ADDR: usize = 16;

type NodeError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct NodeID(pub [u8; 16]);

impl NodeID {
    pub fn new() -> Self {
        Self(*uuid::Uuid::new_v4().as_bytes())
    }
}
impl Default for NodeID {
    fn default() -> Self {
        Self::new()
    }
}
impl std::fmt::Display for NodeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Message {
    RegisterNode {
        id: NodeID,
        addr: String,
    },
    NodeList {
        epoch: u64,
        nodes: Vec<(NodeID, String)>,
    },
    Ping {
        id: NodeID,
        addr: String,
    },
    GetObject(ObjectID),
    ObjectData(ObjectID, Vec<u8>),
    ObjectNotFound(ObjectID),
    GcsSync {
        source: NodeID,
        objects: Vec<(ObjectID, usize)>,
        tasks: Vec<(crate::common::TaskID, crate::common::TaskState)>,
        actors: Vec<(
            crate::common::ActorID,
            String,
            crate::common::ActorState,
            Option<NodeID>,
        )>,
    },
    ActorCall {
        actor_id: crate::common::ActorID,
        method: String,
        args: Vec<u8>,
    },
    ActorCallReply(Result<Vec<u8>, String>),
    Ack,
}

struct Connection {
    stream: TcpStream,
}

impl Connection {
    async fn connect(addr: &str, deadline: tokio::time::Instant) -> Result<Self, NodeError> {
        let stream = tokio::time::timeout_at(deadline, TcpStream::connect(addr))
            .await
            .map_err(|_| "connect timeout")??;
        Ok(Self { stream })
    }

    async fn send(
        &mut self,
        message: &Message,
        deadline: tokio::time::Instant,
    ) -> Result<(), NodeError> {
        let bytes = bincode::serialize(message)?;
        if bytes.len() > MAX_MESSAGE_SIZE {
            return Err(format!("message too large: {} bytes", bytes.len()).into());
        }
        let write = async {
            self.stream
                .write_all(&(bytes.len() as u64).to_be_bytes())
                .await?;
            for chunk in bytes.chunks(CHUNK_SIZE) {
                self.stream.write_all(chunk).await?;
            }
            Ok::<(), std::io::Error>(())
        };
        tokio::time::timeout_at(deadline, write)
            .await
            .map_err(|_| "write timeout")??;
        Ok(())
    }

    async fn recv(&mut self, deadline: tokio::time::Instant) -> Result<Option<Message>, NodeError> {
        let read = async {
            let mut len = [0u8; 8];
            match self.stream.read_exact(&mut len).await {
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(error) => return Err(error),
            }
            let len = usize::try_from(u64::from_be_bytes(len)).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "length overflow")
            })?;
            if len > MAX_MESSAGE_SIZE {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "message too large",
                ));
            }
            let mut bytes = vec![0; len];
            for chunk in bytes.chunks_mut(CHUNK_SIZE) {
                self.stream.read_exact(chunk).await?;
            }
            Ok(Some(bytes))
        };
        let Some(bytes) = tokio::time::timeout_at(deadline, read)
            .await
            .map_err(|_| "read timeout")??
        else {
            return Ok(None);
        };
        Ok(Some(bincode::deserialize(&bytes)?))
    }
}

#[derive(Default)]
struct ConnectionPool {
    inner: Mutex<HashMap<String, Vec<Connection>>>,
}

impl ConnectionPool {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    async fn get(
        &self,
        addr: &str,
        deadline: tokio::time::Instant,
    ) -> Result<Connection, NodeError> {
        if let Some(connection) = self.inner.lock().get_mut(addr).and_then(Vec::pop) {
            return Ok(connection);
        }
        Connection::connect(addr, deadline).await
    }

    fn put(&self, addr: &str, connection: Connection) {
        let mut inner = self.inner.lock();
        let connections = inner.entry(addr.to_string()).or_default();
        if connections.len() < MAX_CONNS_PER_ADDR {
            connections.push(connection);
        }
    }

    fn invalidate(&self, addr: &str) {
        self.inner.lock().remove(addr);
    }
}

struct RemoteNode {
    addr: String,
    last_seen: std::time::Instant,
}

#[derive(Clone)]
pub struct Node {
    pub id: NodeID,
    pub addr: String,
    store: ObjectStore,
    peers: Arc<Mutex<HashMap<NodeID, RemoteNode>>>,
    membership_epoch: Arc<AtomicU64>,
    is_head: bool,
    pool: Arc<ConnectionPool>,
    gcs: Arc<Mutex<Option<crate::gcs::Gcs>>>,
    local_actors: Arc<Mutex<HashMap<crate::common::ActorID, Arc<crate::actor::ActorHandleInner>>>>,
}

impl Node {
    pub async fn start(
        addr: &str,
        head_addr: Option<&str>,
        store: ObjectStore,
    ) -> Result<Arc<Self>, NodeError> {
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?.to_string();
        let node = Arc::new(Self {
            id: NodeID::new(),
            addr: local_addr.clone(),
            store,
            peers: Arc::new(Mutex::new(HashMap::new())),
            membership_epoch: Arc::new(AtomicU64::new(1)),
            is_head: head_addr.is_none(),
            pool: ConnectionPool::new(),
            gcs: Arc::new(Mutex::new(None)),
            local_actors: Arc::new(Mutex::new(HashMap::new())),
        });

        let server = node.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let server = server.clone();
                tokio::spawn(async move {
                    if let Err(error) = handle_connection(stream, server).await {
                        tracing::warn!("connection error: {error}");
                    }
                });
            }
        });

        if let Some(head) = head_addr {
            node.refresh_membership(head, true).await?;
            let node_for_heartbeat = node.clone();
            let head = head.to_string();
            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(Duration::from_secs(5));
                loop {
                    ticker.tick().await;
                    if let Err(error) = node_for_heartbeat.refresh_membership(&head, false).await {
                        tracing::warn!("heartbeat failed: {error}");
                    }
                }
            });
        } else {
            let head = node.clone();
            tokio::spawn(async move {
                let mut ticker = tokio::time::interval(Duration::from_secs(10));
                loop {
                    ticker.tick().await;
                    let cutoff = std::time::Instant::now() - Duration::from_secs(15);
                    let removed: Vec<_> = {
                        let mut peers = head.peers.lock();
                        let removed = peers
                            .iter()
                            .filter(|(_, peer)| peer.last_seen <= cutoff)
                            .map(|(id, peer)| (*id, peer.addr.clone()))
                            .collect::<Vec<_>>();
                        peers.retain(|_, peer| peer.last_seen > cutoff);
                        removed
                    };
                    if !removed.is_empty() {
                        head.membership_epoch.fetch_add(1, Ordering::AcqRel);
                        for (_, addr) in removed {
                            head.pool.invalidate(&addr);
                        }
                    }
                }
            });
        }
        Ok(node)
    }

    async fn refresh_membership(&self, head: &str, register: bool) -> Result<(), NodeError> {
        let deadline = tokio::time::Instant::now() + RPC_TIMEOUT;
        let mut connection = Connection::connect(head, deadline).await?;
        let message = if register {
            Message::RegisterNode {
                id: self.id,
                addr: self.addr.clone(),
            }
        } else {
            Message::Ping {
                id: self.id,
                addr: self.addr.clone(),
            }
        };
        connection.send(&message, deadline).await?;
        let Some(Message::NodeList { epoch, nodes }) = connection.recv(deadline).await? else {
            return Err("invalid membership reply".into());
        };
        if epoch < self.membership_epoch.load(Ordering::Acquire) {
            return Ok(());
        }
        self.membership_epoch.store(epoch, Ordering::Release);
        let now = std::time::Instant::now();
        let next: HashMap<_, _> = nodes
            .into_iter()
            .filter(|(id, _)| *id != self.id)
            .map(|(id, addr)| {
                (
                    id,
                    RemoteNode {
                        addr,
                        last_seen: now,
                    },
                )
            })
            .collect();
        let removed: Vec<_> = {
            let peers = self.peers.lock();
            peers
                .iter()
                .filter(|(id, _)| !next.contains_key(id))
                .map(|(_, peer)| peer.addr.clone())
                .collect()
        };
        *self.peers.lock() = next;
        for addr in removed {
            self.pool.invalidate(&addr);
        }
        Ok(())
    }

    pub fn with_gcs(self: &Arc<Self>, gcs: crate::gcs::Gcs) -> Arc<Self> {
        *self.gcs.lock() = Some(gcs);
        let node = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(2));
            loop {
                ticker.tick().await;
                node.broadcast_gcs().await;
            }
        });
        self.clone()
    }

    pub fn register_actor(&self, inner: Arc<crate::actor::ActorHandleInner>) {
        self.local_actors.lock().insert(inner.id, inner);
        let node = self.clone();
        tokio::spawn(async move { node.broadcast_gcs().await });
    }

    pub async fn remote_actor_call(
        &self,
        actor_id: crate::common::ActorID,
        owner: NodeID,
        method: &str,
        args: Vec<u8>,
    ) -> Result<Vec<u8>, NodeError> {
        let addr = self.peer_addr(owner)?;
        let message = Message::ActorCall {
            actor_id,
            method: method.to_string(),
            args,
        };
        match self.request(&addr, message).await? {
            Message::ActorCallReply(result) => result.map_err(Into::into),
            _ => Err("unexpected actor reply".into()),
        }
    }

    async fn request(&self, addr: &str, message: Message) -> Result<Message, NodeError> {
        let deadline = tokio::time::Instant::now() + RPC_TIMEOUT;
        for attempt in 0..2 {
            let mut connection = self.pool.get(addr, deadline).await?;
            let result = async {
                connection.send(&message, deadline).await?;
                connection
                    .recv(deadline)
                    .await?
                    .ok_or_else(|| "peer closed connection".into())
            }
            .await;
            match result {
                Ok(reply) => {
                    self.pool.put(addr, connection);
                    return Ok(reply);
                }
                Err(error) if attempt == 0 && tokio::time::Instant::now() < deadline => {
                    self.pool.invalidate(addr);
                    tracing::debug!("retrying RPC after pooled connection failure: {error}");
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!()
    }

    fn peer_addr(&self, id: NodeID) -> Result<String, NodeError> {
        self.peers
            .lock()
            .get(&id)
            .map(|peer| peer.addr.clone())
            .ok_or_else(|| format!("no peer with node id {id}").into())
    }

    async fn broadcast_gcs(&self) {
        let Some(gcs) = self.gcs.lock().clone() else {
            return;
        };
        let message = Message::GcsSync {
            source: self.id,
            objects: gcs
                .objects()
                .into_iter()
                .filter(|o| o.owner_node.is_none() || o.owner_node == Some(self.id))
                .map(|o| (o.id, o.size_bytes))
                .collect(),
            tasks: gcs
                .tasks()
                .into_iter()
                .map(|task| (task.id, task.state))
                .collect(),
            actors: gcs
                .actors()
                .into_iter()
                .filter(|actor| actor.owner_node.is_none() || actor.owner_node == Some(self.id))
                .map(|actor| (actor.id, actor.name, actor.state, actor.owner_node))
                .collect(),
        };
        let peers: Vec<_> = self
            .peers
            .lock()
            .values()
            .map(|peer| peer.addr.clone())
            .collect();
        for addr in peers {
            let node = self.clone();
            let message = message.clone();
            tokio::spawn(async move {
                let _ = node.request(&addr, message).await;
            });
        }
    }

    pub async fn fetch_remote_object(&self, id: ObjectID) -> Result<Vec<u8>, NodeError> {
        let peers: Vec<_> = self
            .peers
            .lock()
            .values()
            .map(|peer| peer.addr.clone())
            .collect();
        let mut requests = FuturesUnordered::new();
        for addr in peers {
            let node = self.clone();
            requests.push(async move {
                match node.request(&addr, Message::GetObject(id)).await? {
                    Message::ObjectData(returned, bytes) if returned == id => Ok(bytes),
                    _ => Err::<Vec<u8>, NodeError>("object not found on peer".into()),
                }
            });
        }
        while let Some(result) = requests.next().await {
            if let Ok(bytes) = result {
                return Ok(bytes);
            }
        }
        Err("object not found on any peer".into())
    }

    pub fn peers(&self) -> Vec<(NodeID, String)> {
        self.peers
            .lock()
            .iter()
            .map(|(id, peer)| (*id, peer.addr.clone()))
            .collect()
    }
}

impl crate::object_store::RemoteFetcher for Node {
    fn fetch_remote(&self, id: ObjectID) -> crate::object_store::RemoteFetchFuture {
        let node = self.clone();
        Box::pin(async move { node.fetch_remote_object(id).await })
    }
}

async fn handle_connection(stream: TcpStream, node: Arc<Node>) -> Result<(), NodeError> {
    let mut connection = Connection { stream };
    loop {
        let deadline = tokio::time::Instant::now() + RPC_TIMEOUT;
        let Some(message) = connection.recv(deadline).await? else {
            return Ok(());
        };
        match message {
            Message::RegisterNode { id, addr } | Message::Ping { id, addr } => {
                if !node.is_head {
                    return Err("membership request sent to worker".into());
                }
                let changed = node
                    .peers
                    .lock()
                    .get(&id)
                    .is_none_or(|peer| peer.addr != addr);
                node.peers.lock().insert(
                    id,
                    RemoteNode {
                        addr,
                        last_seen: std::time::Instant::now(),
                    },
                );
                if changed {
                    node.membership_epoch.fetch_add(1, Ordering::AcqRel);
                }
                let mut nodes: Vec<_> = node
                    .peers
                    .lock()
                    .iter()
                    .map(|(id, peer)| (*id, peer.addr.clone()))
                    .collect();
                nodes.push((node.id, node.addr.clone()));
                connection
                    .send(
                        &Message::NodeList {
                            epoch: node.membership_epoch.load(Ordering::Acquire),
                            nodes,
                        },
                        deadline,
                    )
                    .await?;
            }
            Message::GetObject(id) => {
                let reply = if node.store.contains(id) {
                    Message::ObjectData(id, node.store.get_bytes(id).await?.to_vec())
                } else {
                    Message::ObjectNotFound(id)
                };
                connection.send(&reply, deadline).await?;
            }
            Message::GcsSync {
                source,
                objects,
                tasks,
                actors,
            } => {
                if let Some(gcs) = node.gcs.lock().clone() {
                    for (id, size_bytes) in objects {
                        gcs.add_object(crate::gcs::ObjectMeta {
                            id,
                            size_bytes,
                            created_at: std::time::Instant::now(),
                            owner: None,
                            owner_node: Some(source),
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
                        if gcs.get_actor(id).is_some() {
                            gcs.set_actor_state(id, state);
                        } else {
                            gcs.add_actor(crate::gcs::ActorMeta {
                                id,
                                name,
                                state,
                                owner_node: owner_node.or(Some(source)),
                                created_at: std::time::Instant::now(),
                                pending_tasks: 0,
                                completed_tasks: 0,
                            });
                        }
                    }
                }
                connection.send(&Message::Ack, deadline).await?;
            }
            Message::ActorCall {
                actor_id,
                method,
                args,
            } => {
                let actor = node.local_actors.lock().get(&actor_id).cloned();
                let result = match actor {
                    Some(actor) => actor
                        .call_method(&method, args)
                        .await
                        .map_err(|error| error.to_string()),
                    None => Err(format!("actor {actor_id} not found on this node")),
                };
                connection
                    .send(&Message::ActorCallReply(result), deadline)
                    .await?;
            }
            _ => return Err("unexpected request message".into()),
        }
    }
}
