//! One-call local cluster: spawn a coordinator + N workers as child processes on
//! this host, for `ray.init()`-style "just run it" ergonomics.
//!
//! # Contract
//!
//! This is a convenience wrapper, not new runtime capability — it shells out to
//! the same `crayon-cluster` binary a user would launch by hand, and holds the
//! child handles so they are killed on drop. Deliberately minimal: no service
//! discovery, no autoscaling, no resource probing. The target is same-host RL
//! rollout development; a real multi-host deployment launches the processes
//! itself (systemd/k8s), not through this.
//!
//! The `crayon-cluster` binary is resolved from `$CRAYON_CLUSTER_BIN`, else a
//! `crayon-cluster` next to the current executable, else `crayon-cluster` on
//! `PATH`. Python callers (whose wheel bundles only the extension module) set
//! the env var or rely on PATH.

use std::{
    net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use crate::error::Error;

/// A coordinator + workers spawned on this host. Dropping it kills every child.
pub struct LocalCluster {
    /// `host:port` the coordinator is listening on; hand this to a client.
    pub coordinator_addr: String,
    children: Vec<Child>,
}

impl LocalCluster {
    /// Spawns a coordinator and `workers` worker processes, each serving `ops`
    /// (the operation set string the worker registers, e.g. `"all"` or `"rl"`),
    /// and blocks until the coordinator is accepting and all workers registered.
    pub fn start(workers: usize, ops: &str) -> Result<Self, Error> {
        let binary = resolve_binary()?;
        let coordinator_addr = free_addr()?;

        let mut children = Vec::with_capacity(workers + 1);
        children.push(spawn(&binary, &["coordinator", &coordinator_addr, "5000"])?);
        wait_listening(&coordinator_addr, Duration::from_secs(10))?;

        for _ in 0..workers {
            let advertise = free_addr()?;
            let node_id = crate::ids::NodeId::new().to_string();
            children.push(spawn(
                &binary,
                &[
                    "worker",
                    &coordinator_addr,
                    &advertise,
                    &node_id,
                    "1.0",
                    ops,
                ],
            )?);
        }

        let cluster = Self {
            coordinator_addr,
            children,
        };
        cluster.wait_workers(workers, Duration::from_secs(15))?;
        Ok(cluster)
    }

    /// Blocks until at least `n` workers have registered, by asking the
    /// coordinator over its own protocol. Runs on a dedicated OS thread with its
    /// own runtime, so it is safe to call whether or not the caller already has
    /// an ambient tokio runtime (the CLI's `#[tokio::main]`, Python's runtime).
    fn wait_workers(&self, n: usize, timeout: Duration) -> Result<(), Error> {
        if n == 0 {
            return Ok(());
        }
        let addr = self.coordinator_addr.clone();
        thread::scope(|scope| {
            scope
                .spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| Error::Io(e.to_string()))?;
                    runtime.block_on(async move {
                        let mut client = crate::client::ClusterClient::connect(&addr);
                        client.connect_epoch().await?;
                        let deadline = Instant::now() + timeout;
                        loop {
                            match client.workers().await {
                                Ok(list) if list.len() >= n => return Ok(()),
                                _ if Instant::now() < deadline => {
                                    tokio::time::sleep(Duration::from_millis(100)).await
                                }
                                _ => {
                                    return Err(Error::Protocol(format!(
                                        "expected {n} workers but not all registered in time"
                                    )))
                                }
                            }
                        }
                    })
                })
                .join()
                .map_err(|_| Error::Io("readiness thread panicked".into()))?
        })
    }
}

impl Drop for LocalCluster {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// A free `127.0.0.1:port`, chosen by binding port 0 and releasing it. Racy in
/// principle (another process could grab it before spawn), but fine for local
/// dev where nothing else is fighting for ports.
fn free_addr() -> Result<String, Error> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| Error::Io(e.to_string()))?;
    Ok(listener
        .local_addr()
        .map_err(|e| Error::Io(e.to_string()))?
        .to_string())
}

fn spawn(binary: &PathBuf, args: &[&str]) -> Result<Child, Error> {
    Command::new(binary)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::Io(format!("spawn {}: {e}", binary.display())))
}

/// Polls a TCP connect until the coordinator accepts or the deadline passes.
fn wait_listening(addr: &str, timeout: Duration) -> Result<(), Error> {
    let socket: SocketAddr = addr
        .parse()
        .map_err(|_| Error::Protocol(format!("bad coordinator addr {addr}")))?;
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&socket, Duration::from_millis(200)).is_ok() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err(Error::Protocol("coordinator did not start in time".into()))
}

/// `$CRAYON_CLUSTER_BIN`, else a sibling of the current exe, else bare
/// `crayon-cluster` (found via `PATH` by `Command`).
fn resolve_binary() -> Result<PathBuf, Error> {
    if let Some(path) = std::env::var_os("CRAYON_CLUSTER_BIN") {
        return Ok(PathBuf::from(path));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join("crayon-cluster");
            if sibling.exists() {
                return Ok(sibling);
            }
        }
    }
    Ok(PathBuf::from("crayon-cluster"))
}
