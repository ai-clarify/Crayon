#!/usr/bin/env python3
"""Ray object-store baseline: put(bytes)+get(ref) per-op latency and throughput,
matching src/bin/storage_benchmark.rs so the two are directly comparable.

Serial loop (concurrency 1) mirrors the Rust e2e path: put a fresh payload, get
it back, drop the ref. Reports p50/p99 microseconds and MB/s per payload size.
"""
import argparse, gc, json, statistics, time

import ray


def pct(xs, p):
    xs = sorted(xs)
    k = round(p / 100.0 * (len(xs) - 1))
    return xs[k]


def mb_s(size, us):
    return (size / 1_048_576.0) / (us / 1e6) if us > 0 else 0.0


def run(size, samples, warmups):
    payload = bytearray(b"\xA5" * size)
    put_us, get_us = [], []
    for i in range(warmups + samples):
        payload[:8] = (i & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
        b = bytes(payload)

        t = time.perf_counter()
        ref = ray.put(b)
        put_e = (time.perf_counter() - t) * 1e6

        t = time.perf_counter()
        got = ray.get(ref)
        get_e = (time.perf_counter() - t) * 1e6
        assert len(got) == size

        del ref, got  # drop the ref so plasma reclaims, like Rust release()
        if i >= warmups:
            put_us.append(put_e)
            get_us.append(get_e)
    return {
        "size_bytes": size,
        "put_p50_us": round(pct(put_us, 50), 2),
        "put_p99_us": round(pct(put_us, 99), 2),
        "put_mb_s": round(mb_s(size, statistics.mean(put_us)), 1),
        "get_p50_us": round(pct(get_us, 50), 2),
        "get_p99_us": round(pct(get_us, 99), 2),
        "get_mb_s": round(mb_s(size, statistics.mean(get_us)), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--warmups", type=int, default=50)
    ap.add_argument("--sizes", default="1024,65536,1048576")
    ap.add_argument("--out", default="/tmp/ray_storage_summary.json")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]

    ray.init(configure_logging=False, log_to_driver=False)
    gc.disable()
    rows = [run(s, a.samples, a.warmups) for s in sizes]
    gc.enable()
    ray.shutdown()

    report = {"samples": a.samples, "warmups": a.warmups, "e2e": rows}
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
