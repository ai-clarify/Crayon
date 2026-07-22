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

struct Cluster {
    binary: &'static str,
    coordinator: String,
    children: Vec<Child>,
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
            children: vec![child],
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
        let child = Command::new(self.binary)
            .args([
                "worker",
                &self.coordinator,
                &address,
                &uuid::Uuid::new_v4().simple().to_string(),
                &cpu.to_string(),
                operation,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        self.children.push(child);
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
        let values: Vec<_> = stdout(output)
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        (values[0].clone(), values[1].clone())
    }

    fn status(&self, task: &str) -> String {
        stdout(self.run(["status", &self.coordinator, task]))
    }

    fn eventually(&self, timeout: Duration, mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if predicate() {
                return;
            }
            thread::sleep(Duration::from_millis(25));
        }
        panic!("condition not met within {timeout:?}");
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
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
    cluster.eventually(Duration::from_secs(3), || {
        cluster.status(&task).contains("Running")
    });
    cluster.children[1].kill().unwrap();
    cluster.children[1].wait().unwrap();
    cluster.eventually(Duration::from_secs(5), || {
        cluster.status(&task).contains("Succeeded") && cluster.status(&task).ends_with("2")
    });
    assert_eq!(
        stdout(cluster.run(["get", &cluster.coordinator, &output])),
        "800"
    );
    cluster.children[2].kill().unwrap();
    cluster.children[2].wait().unwrap();
    thread::sleep(Duration::from_millis(500));
    let lost = cluster.run(["get", &cluster.coordinator, &output]);
    assert!(!lost.status.success());
    assert!(String::from_utf8_lossy(&lost.stderr).contains("ObjectLost"));
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
