//! Demo: exercise the Crayon runtime end-to-end.
//!
//! Run with: `cargo run --bin crayon-demo`

use crayon::Ray;

#[derive(Default)]
struct Counter {
    n: i64,
}

impl Counter {
    fn increment(&mut self) -> i64 {
        self.n += 1;
        self.n
    }

    fn add(&mut self, x: i64) -> i64 {
        self.n += x;
        self.n
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let ray = Ray::init(4);

    // 1. put / get
    let r = ray.put(42);
    let v: i32 = ray.get(&r).await?;
    println!("put/get: {v}");
    assert_eq!(v, 42);

    // 2. remote task
    let task_ref = ray.spawn((), |()| {
        let mut sum = 0i64;
        for i in 0..1000 {
            sum += i;
        }
        sum
    });
    let sum: i64 = ray.get(&task_ref).await?;
    println!("task sum 0..1000 = {sum}");
    assert_eq!(sum, 499500);

    // 3. many concurrent tasks (fan-out)
    let refs: Vec<_> = (0..8).map(|i| ray.spawn((), move |()| i * i)).collect();
    let mut total = 0i64;
    for r in &refs {
        total += ray.get::<i64>(r).await?;
    }
    println!("fan-out sum of squares 0..8 = {total}");
    assert_eq!(total, (0..8).map(|i| i * i).sum());

    // 3b. task dependency resolution (ObjectRef args auto-fetched)
    let a = ray.put(10);
    let b = ray.put(20);
    let dep_ref = ray.spawn((a, b), |(a, b): (i32, i32)| a + b);
    let dep_sum: i32 = ray.get(&dep_ref).await?;
    println!("task dependency 10 + 20 = {dep_sum}");
    assert_eq!(dep_sum, 30);

    // 4. actor
    let counter = ray.create_actor("counter", Counter::default());
    let r1 = counter.call(|c| c.increment()).await?;
    let r2 = counter.call(|c| c.add(10)).await?;
    let v1: i64 = ray.get(&r1).await?;
    let v2: i64 = ray.get(&r2).await?;
    println!("actor increment -> {v1}, add(10) -> {v2}");
    assert_eq!(v1, 1);
    assert_eq!(v2, 11);

    // 4b. named actor lookup
    let counter2: crayon::actor::ActorHandle<Counter> = ray
        .get_actor("counter")
        .expect("named actor should exist");
    let r3 = counter2.call(|c| c.increment()).await?;
    let v3: i64 = ray.get(&r3).await?;
    println!("named actor increment -> {v3}");
    assert_eq!(v3, 12);

    // 5. status
    let status = ray.status();
    println!("\n{}", status.pretty());

    println!("\nAll demos passed ✔");
    Ok(())
}
