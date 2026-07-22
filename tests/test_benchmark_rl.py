import benchmark_rl as benchmark


def test_seed_derivation_is_stable_and_distinct():
    assert benchmark.derive_seed(42, 1, 2, 3) == benchmark.derive_seed(42, 1, 2, 3)
    assert benchmark.derive_seed(42, 1, 2, 3) != benchmark.derive_seed(42, 1, 2, 4)


def test_summary_reports_distribution():
    result = benchmark.summarize([1.0, 2.0, 3.0, 4.0])
    assert result["median"] == 2.5
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["count"] == 4


def test_nvidia_smi_parser_filters_process_tree():
    result = benchmark.parse_nvidia_smi("10, GPU-a, 100\n20, GPU-b, 50\n", {10})
    assert result["total_mib"] == 100
    assert result["rows"] == [{"pid": 10, "gpu": "GPU-a", "memory_mib": 100.0}]
