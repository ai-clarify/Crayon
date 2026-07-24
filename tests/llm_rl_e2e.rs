//! Real-model end-to-end RL through the full stack: coordinator, llm-actor
//! workers with resident Python sidecars, llm-judge, REINFORCE learner, and
//! arena weight broadcast — driven by `benchmarks/crayon_llm_rl.py`.
//!
//! Needs a GPU host with torch/transformers and a local HF model, so it only
//! runs when `CRAYON_LLM_MODEL` points at a model directory (e.g. on the v100
//! benchmark host); everywhere else it's a no-op and `cargo test` stays green.
use std::process::Command;

#[test]
fn llm_rl_end_to_end() {
    let Ok(model) = std::env::var("CRAYON_LLM_MODEL") else {
        eprintln!("skipping llm_rl_end_to_end: set CRAYON_LLM_MODEL to run");
        return;
    };
    let repo = env!("CARGO_MANIFEST_DIR");
    let out = format!("{}/crayon_llm_rl_test.json", std::env::temp_dir().display());
    let status = Command::new("python3")
        .current_dir(repo)
        .args([
            "benchmarks/crayon_llm_rl.py",
            "--model",
            &model,
            "--iterations",
            "2",
            "--binary",
            env!("CARGO_BIN_EXE_crayon-cluster"),
            "--out",
            &out,
        ])
        .status()
        .expect("python3 not runnable");
    assert!(status.success(), "llm rl driver failed: {status:?}");
    let summary = std::fs::read_to_string(&out).expect("driver wrote no summary");
    let summary: serde_json::Value = serde_json::from_str(&summary).unwrap();
    assert_eq!(summary["iterations"].as_u64(), Some(2));
    // Two iterations prove the loop, not learning; accuracy just has to exist
    // and be a sane probability.
    let accuracy = summary["rows"][1]["accuracy"].as_f64().expect("accuracy");
    assert!((0.0..=1.0).contains(&accuracy), "accuracy {accuracy}");
}
