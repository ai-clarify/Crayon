use futures::FutureExt;

#[tokio::test]
async fn test_catch_unwind_works() {
    let fut = std::panic::AssertUnwindSafe(async {
        panic!("test panic");
    });
    let result = fut.catch_unwind().await;
    eprintln!("result is_err: {}", result.is_err());
    assert!(result.is_err());
}
