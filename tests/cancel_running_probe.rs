use crayon::common::{CrayonError, TaskState};
use crayon::Ray;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_running_is_immediate_and_suppresses_late_result() {
    let ray = Ray::init(1);
    let result = ray.spawn((), |()| {
        std::thread::sleep(Duration::from_millis(300));
        42u64
    });
    let task_id = result.task_id().unwrap();

    loop {
        if ray
            .gcs()
            .tasks()
            .into_iter()
            .any(|task| task.id == task_id && task.state == TaskState::Running)
        {
            break;
        }
        tokio::task::yield_now().await;
    }

    assert!(ray.cancel(task_id));
    assert!(matches!(ray.get(&result).await, Err(CrayonError::TaskCancelled(id)) if id == task_id));
    tokio::time::sleep(Duration::from_millis(350)).await;
    let task = ray
        .gcs()
        .tasks()
        .into_iter()
        .find(|task| task.id == task_id)
        .unwrap();
    assert_eq!(task.state, TaskState::Cancelled);
    assert!(!ray.cancel(task_id));
}
