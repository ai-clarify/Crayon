use std::{
    net::TcpListener,
    process::{Child, Command, Stdio},
    thread,
    time::Duration,
};

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

struct Children(Vec<Child>);
impl Drop for Children {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[test]
fn task_executes_in_a_separate_worker_process() {
    let binary = env!("CARGO_BIN_EXE_crayon-cluster");
    let coordinator_port = free_port();
    let worker_port = free_port();
    let coordinator = format!("127.0.0.1:{coordinator_port}");
    let worker = format!("127.0.0.1:{worker_port}");
    let mut children = Children(Vec::new());
    children.0.push(
        Command::new(binary)
            .args(["coordinator", &coordinator])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    thread::sleep(Duration::from_millis(100));
    children.0.push(
        Command::new(binary)
            .args(["worker", &coordinator, &worker])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    thread::sleep(Duration::from_millis(300));
    let mut last = None;
    for _ in 0..50 {
        let output = Command::new(binary)
            .args(["submit", &coordinator, "20", "22"])
            .output();
        match output {
            Ok(output)
                if output.status.success()
                    && String::from_utf8_lossy(&output.stdout).trim() == "42" =>
            {
                return
            }
            value => last = Some(format!("{value:?}")),
        }
        thread::sleep(Duration::from_millis(100));
    }
    panic!("cluster did not produce 42: {last:?}");
}
