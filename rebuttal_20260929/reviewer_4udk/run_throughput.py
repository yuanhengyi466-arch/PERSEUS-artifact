"""Actual serial/concurrent replay of the frozen native all-fallback policy."""
from __future__ import annotations

import csv
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import psutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NATIVE = ROOT / "rebuttal_20260928/protocol_attribution"
sys.path.insert(0, str(NATIVE))
import run_protocol_attribution as native


def run_block(cases, gates, selections, policy, concurrency, repetition):
    jobs = Queue()
    for position, case in enumerate(cases):
        jobs.put((position, case))
    barrier = threading.Barrier(concurrency + 1)
    started = {}
    counter, lock = [0], threading.Lock()
    connections = [native.connect() for _ in range(concurrency)]

    def worker(worker_id):
        results = []
        with connections[worker_id].cursor() as cur:
            barrier.wait(timeout=30)
            while True:
                try:
                    position, case = jobs.get_nowait()
                except Empty:
                    break
                offset = time.perf_counter() - started["time"]
                observation = native.request(cur, case, policy, {}, gates, selections)
                results.append({"concurrency": concurrency, "repetition": repetition, "policy": policy,
                    "worker": worker_id, "position": position, "qid": case["qid"], "sql_id": case["sql_id"],
                    "offset_seconds": offset, **observation})
                with lock:
                    counter[0] += 1
                    if counter[0] % 100 == 0:
                        native.emit("throughput_progress", concurrency=concurrency, repetition=repetition,
                                    policy=policy, completed=counter[0], total=len(cases))
        return results

    cpu_before = psutil.cpu_percent()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(concurrency)]
        started["time"] = time.perf_counter()
        barrier.wait(timeout=30)
        rows = [row for f in futures for row in f.result()]
        elapsed = time.perf_counter() - started["time"]
    for conn in connections:
        conn.close()
    assert len(rows) == len(cases)
    assert all(not r["admitted"] and r["active"] == "pg" for r in rows)
    latency = np.array([r["wall_ms"] for r in rows])
    good = sum(r["status"] == "ok" for r in rows)
    summary = {"concurrency": concurrency, "repetition": repetition, "policy": policy,
               "attempted": len(rows), "completed_correctly": good, "timeouts": sum(r["status"] == "timeout" for r in rows),
               "other_failures": sum(r["status"] not in ("ok", "timeout") for r in rows),
               "elapsed_seconds": elapsed, "goodput_qps": good / elapsed, "attempted_qps": len(rows) / elapsed,
               "request_p50_ms": float(np.percentile(latency, 50)), "request_p95_ms": float(np.percentile(latency, 95)),
               "request_p99_ms": float(np.percentile(latency, 99)), "max_request_ms": float(latency.max()),
               "host_cpu_before_percent": cpu_before, "host_cpu_during_percent": psutil.cpu_percent(),
               "available_memory": psutil.virtual_memory().available}
    native.emit("throughput_block_complete", **summary)
    return rows, summary


def main():
    native.check_prepared()
    directory = OUT / "throughput"
    if directory.exists():
        raise RuntimeError("Refusing to overwrite a throughput replay")
    directory.mkdir()
    cases = [c for c in native.read_json(NATIVE / "cases.json") if c["role"] == "test"]
    gates = native.read_json(NATIVE / "estimator_gates.json")
    selections = native.read_json(NATIVE / "selected_rules.json")["policies"]
    assert len(cases) == 540
    assert all(selections[f"{c['outer']}/hgb/full"]["rule"] is None for c in cases)
    configuration = {"concurrencies": [1, 4], "repetitions": [0, 1, 2], "policies": ["pg", "full:hgb"],
                     "cases": len(cases), "distinct_sql": len({c["sql_id"] for c in cases}),
                     "protocol_sha256": native.hash_file(OUT / "THROUGHPUT_PROTOCOL.md"),
                     "runner_sha256": native.hash_file(__file__), "native_runner_sha256": native.hash_file(NATIVE / "run_protocol_attribution.py"),
                     "native_selection_sha256": native.hash_file(NATIVE / "selected_rules.json"),
                     "cases_sha256": native.hash_file(NATIVE / "cases.json"), "all_fallback": True}
    native.write_json(directory / "configuration.json", configuration)
    warmup = []
    conn = native.connect()
    with conn.cursor() as cur:
        for i, case in enumerate(cases):
            obs = native.request(cur, case, "pg", {}, gates)
            warmup.append({"qid": case["qid"], **obs})
            if (i + 1) % 100 == 0:
                native.emit("throughput_warmup", completed=i + 1, total=len(cases))
    conn.close()
    native.write_csv(directory / "warmup.csv", warmup)
    summaries, rows = [], []
    for repetition in range(3):
        concurrency_order = (1, 4) if repetition % 2 == 0 else (4, 1)
        for concurrency in concurrency_order:
            ordered = list(cases)
            random.Random(f"4udk-throughput|{repetition}|{concurrency}").shuffle(ordered)
            policies = ("pg", "full:hgb") if (repetition + (concurrency == 4)) % 2 == 0 else ("full:hgb", "pg")
            for policy in policies:
                block_rows, summary = run_block(ordered, gates, selections, policy, concurrency, repetition)
                rows.extend(block_rows)
                summaries.append(summary)
                native.write_csv(directory / "requests.csv", rows)
                native.write_csv(directory / "blocks.csv", summaries)
    assert len(rows) == 6480
    lookup = {(r["concurrency"], r["repetition"], r["policy"]): r for r in summaries}
    comparisons = []
    for concurrency in (1, 4):
        for repetition in range(3):
            pg = lookup[(concurrency, repetition, "pg")]
            full = lookup[(concurrency, repetition, "full:hgb")]
            comparisons.append({"concurrency": concurrency, "repetition": repetition,
                "default_goodput_qps": pg["goodput_qps"], "full_goodput_qps": full["goodput_qps"],
                "goodput_ratio_full_over_default": full["goodput_qps"] / pg["goodput_qps"],
                "p50_ratio": full["request_p50_ms"] / pg["request_p50_ms"],
                "p95_ratio": full["request_p95_ms"] / pg["request_p95_ms"],
                "p99_ratio": full["request_p99_ms"] / pg["request_p99_ms"]})
    native.write_csv(directory / "paired_blocks.csv", comparisons)
    native.write_json(directory / "completed.json", {**configuration, "requests": len(rows),
        "correct": sum(r["status"] == "ok" for r in rows), "timeouts": sum(r["status"] == "timeout" for r in rows),
        "other_failures": sum(r["status"] not in ("ok", "timeout") for r in rows),
        "file_hashes": {p.name: native.hash_file(p) for p in directory.glob("*.csv")}})
    lines = ["# Native All-Fallback Throughput Replay", "",
             "Actual closed-loop request replay; the frozen policy accepts zero repairs. These results measure the current fallback path, not nonzero-admission acceleration. All request statuses are retained.", "",
             "| Clients | Repeat | Default goodput (q/s) | Full-HGB goodput (q/s) | Goodput ratio | P95 latency ratio | P99 latency ratio |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['concurrency']} | {r['repetition']} | {r['default_goodput_qps']:.3f} | {r['full_goodput_qps']:.3f} | {r['goodput_ratio_full_over_default']:.3f} | {r['p95_ratio']:.3f} | {r['p99_ratio']:.3f} |")
    lines += ["", "Each row compares separate replay blocks with matched SQL order. Three repetitions are reported individually; request-level bootstrap is not used as throughput uncertainty. Source predictions/maps are cached, and the all-fallback rule skips repair inference. Host activity and complete raw latencies appear in blocks.csv and requests.csv."]
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    native.emit("throughput_complete", requests=len(rows), correct=sum(r["status"] == "ok" for r in rows))


if __name__ == "__main__":
    main()
