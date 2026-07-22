use crayon::common::TaskState;
use crayon::Ray;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_running_probe() {
    let ray = Ray::init(1);
    let result = ray.spawn((), |()| {
        std::thread::sleep(Duration::from_millis(300));
        42u64
    });

    let task_id = loop {
        if let Some(task) = ray
            .gcs()
            .tasks()
            .into_iter()
            .find(|task| task.state == TaskState::Running)
        {
            break task.id;
        }
        tokio::task::yield_now().await;
    };

    assert!(ray.cancel(task_id));
    assert_eq!(ray.get(&result).await.unwrap(), 42);
    let task = ray.gcs().tasks().into_iter().find(|task| task.id == task_id).unwrap();
    assert_eq!(task.state, TaskState::Finished);
}
