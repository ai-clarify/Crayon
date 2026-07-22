use std::{
    net::{TcpListener, TcpStream},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

struct WorkerProcess {
    node_id: String,
    child: Child,
}

struct Cluster {
    binary: &'static str,
    coordinator: String,
    coordinator_child: Child,
    workers: Vec<WorkerProcess>,
}

impl Cluster {
    fn start(lease_ms: u64) -> Self {
        let binary = env!("CARGO_BIN_EXE_crayon-cluster");
        let coordinator = format!("127.0.0.1:{}", free_port());
        let child = Command::new(binary)
            .args(["coordinator", &coordinator, &lease_ms.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let cluster = Self {
            binary,
            coordinator,
            coordinator_child: child,
            workers: Vec::new(),
        };
        cluster.eventually(Duration::from_secs(3), || {
            cluster
                .run(["workers", &cluster.coordinator])
                .status
                .success()
        });
        cluster
    }

    fn worker(&mut self, operation: &str, cpu: f64) -> String {
        let address = format!("127.0.0.1:{}", free_port());
        let node_id = uuid::Uuid::new_v4().simple().to_string();
        let child = Command::new(self.binary)
            .args([
                "worker",
                &self.coordinator,
                &address,
                &node_id,
                &cpu.to_string(),
                operation,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        self.workers.push(WorkerProcess { node_id, child });
        self.eventually(Duration::from_secs(3), || {
            stdout(self.run(["workers", &self.coordinator])).contains(&address)
        });
        address
    }

    fn run<const N: usize>(&self, args: [&str; N]) -> Output {
        Command::new(self.binary).args(args).output().unwrap()
    }

    fn submit(
        &self,
        operation: &str,
        value: u64,
        object: Option<&str>,
        cpu: f64,
        attempts: u32,
    ) -> (String, String) {
        let output = self.run([
            "submit-detach",
            &self.coordinator,
            operation,
            &value.to_string(),
            object.unwrap_or("-"),
            &cpu.to_string(),
            &attempts.to_string(),
        ]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = stdout(output);
        let mut values = text.split_whitespace();
        let task = values.next().unwrap().to_owned();
        let output = values.next().unwrap().to_owned();
        assert!(values.next().is_none());
        (task, output)
    }

    fn status(&self, task: &str) -> String {
        stdout(self.run(["status", &self.coordinator, task]))
    }

    fn kill_worker(&mut self, node_id: &str) {
        let worker = self
            .workers
            .iter_mut()
            .find(|worker| worker.node_id == node_id)
            .unwrap();
        worker.child.kill().unwrap();
        worker.child.wait().unwrap();
    }

    fn eventually(&self, timeout: Duration, predicate: impl FnMut() -> bool) {
        self.eventually_every(timeout, Duration::from_millis(25), predicate)
    }

    fn eventually_every(
        &self,
        timeout: Duration,
        interval: Duration,
        mut predicate: impl FnMut() -> bool,
    ) {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if predicate() {
                return;
            }
            thread::sleep(interval);
        }
        panic!("condition not met within {timeout:?}");
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        let _ = self.coordinator_child.kill();
        let _ = self.coordinator_child.wait();
        for worker in &mut self.workers {
            let _ = worker.child.kill();
            let _ = worker.child.wait();
        }
    }
}

fn stdout(output: Output) -> String {
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

#[test]
fn remote_execution_and_typed_readiness() {
    let mut cluster = Cluster::start(5_000);
    cluster.worker("add", 1.0);
    let output = cluster.run(["submit", &cluster.coordinator, "20", "22"]);
    assert!(output.status.success());
    assert_eq!(stdout(output), "42");
}

#[test]
fn multi_stage_dag_fetches_worker_local_output() {
    let mut cluster = Cluster::start(5_000);
    cluster.worker("copy", 1.0);
    let (first, first_output) = cluster.submit("copy", 41, None, 1.0, 1);
    cluster.eventually(Duration::from_secs(3), || {
        cluster.status(&first).contains("Succeeded")
    });
    let (second, second_output) = cluster.submit("copy", 0, Some(&first_output), 1.0, 1);
    cluster.eventually(Duration::from_secs(3), || {
        cluster.status(&second).contains("Succeeded")
    });
    assert_eq!(
        stdout(cluster.run(["get", &cluster.coordinator, &second_output])),
        "41"
    );
}

#[test]
fn capability_and_resource_routing_are_enforced() {
    let mut cluster = Cluster::start(5_000);
    let copy_worker = cluster.worker("copy", 0.5);
    let add_worker = cluster.worker("add", 1.0);
    let workers = stdout(cluster.run(["workers", &cluster.coordinator]));
    assert!(workers
        .lines()
        .any(|line| line.contains(&copy_worker) && line.contains("builtin:copy@1")));
    assert!(workers
        .lines()
        .any(|line| line.contains(&add_worker) && line.contains("builtin:add@1")));
    let failure = cluster.run([
        "submit-detach",
        &cluster.coordinator,
        "copy",
        "1",
        "-",
        "1",
        "1",
    ]);
    assert!(!failure.status.success());
    assert!(String::from_utf8_lossy(&failure.stderr).contains("InvalidResource"));
}

#[test]
fn running_cancellation_releases_capacity_after_ack() {
    let mut cluster = Cluster::start(5_000);
    let worker = cluster.worker("sleep", 1.0);
    let (task, output) = cluster.submit("sleep", 5_000, None, 1.0, 1);
    cluster.eventually(Duration::from_secs(3), || {
        cluster.status(&task).contains("Running")
    });
    let cancel = cluster.run(["cancel", &cluster.coordinator, &task]);
    assert!(cancel.status.success());
    cluster.eventually(Duration::from_secs(3), || {
        cluster.status(&task).contains("Cancelled")
    });
    let workers = stdout(cluster.run(["workers", &cluster.coordinator]));
    assert!(workers
        .lines()
        .any(|line| line.contains(&worker) && line.split_whitespace().nth(3) == Some("1")));
    assert!(!cluster
        .run(["get", &cluster.coordinator, &output])
        .status
        .success());
}

#[test]
fn worker_loss_retries_and_loses_owned_objects() {
    let mut cluster = Cluster::start(300);
    cluster.worker("sleep", 1.0);
    cluster.worker("sleep", 1.0);
    let (task, output) = cluster.submit("sleep", 800, None, 1.0, 2);
    let mut owner = None;
    cluster.eventually(Duration::from_secs(3), || {
        let status = cluster.status(&task);
        let fields: Vec<_> = status.split_whitespace().collect();
        if fields.get(2) == Some(&"Running") {
            owner = fields.get(4).map(|value| (*value).to_owned());
            true
        } else {
            false
        }
    });
    let first_owner = owner.unwrap();
    cluster.kill_worker(&first_owner);
    let mut second_owner = None;
    cluster.eventually(Duration::from_secs(5), || {
        let status = cluster.status(&task);
        let fields: Vec<_> = status.split_whitespace().collect();
        if fields.get(2) == Some(&"Running") && fields.get(3) == Some(&"2") {
            second_owner = fields.get(4).map(|value| (*value).to_owned());
            true
        } else {
            false
        }
    });
    cluster.eventually(Duration::from_secs(3), || {
        cluster.status(&task).contains("Succeeded")
    });
    assert_eq!(
        stdout(cluster.run(["get", &cluster.coordinator, &output])),
        "800"
    );
    cluster.kill_worker(&second_owner.unwrap());
    cluster.eventually_every(Duration::from_secs(3), Duration::from_millis(150), || {
        let lost = cluster.run(["get", &cluster.coordinator, &output]);
        !lost.status.success() && String::from_utf8_lossy(&lost.stderr).contains("ObjectLost")
    });
}

#[test]
fn protocol_deadline_and_frame_limits_are_bounded() {
    let cluster = Cluster::start(5_000);
    let mut stream = TcpStream::connect(&cluster.coordinator).unwrap();
    use std::io::Write;
    stream
        .write_all(&((8 * 1024 * 1024 + 1) as u32).to_be_bytes())
        .unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(1)))
        .unwrap();
    let mut byte = [0];
    assert!(std::io::Read::read(&mut stream, &mut byte).is_err() || byte == [0]);
    assert!(cluster
        .run(["workers", &cluster.coordinator])
        .status
        .success());
}
