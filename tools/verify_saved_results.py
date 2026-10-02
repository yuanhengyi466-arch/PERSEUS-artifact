"""Recompute released statistical results without model fitting or DB access."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rebuttal_20260929/reviewer_4udk"
FAMILIES = ("residual_hgb", "direct_hgb", "residual_rf", "direct_rf", "mscn_finetune", "mscn_retrain")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qerror(pred, truth):
    p, t = np.maximum(1., pred), np.maximum(1., truth)
    return np.maximum(p / t, t / p)


def metrics(pred, truth):
    q = qerror(pred, truth)
    return dict(zip(("median", "p90", "p95", "p99"), np.percentile(q, (50, 90, 95, 99))),
                n=len(q), max=q.max(), mean=q.mean())


def check_row(computed, recorded):
    for name, actual in computed.items():
        if not math.isclose(float(actual), float(recorded[name]), rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError((name, actual, recorded[name]))


def bootstrap_intervals(prediction, raw, truth, ids, seed):
    unique, inverse = np.unique(ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=2000)
    weights = counts[:, inverse].astype(np.int32)

    def quantiles(values):
        order = np.argsort(values)
        ordered = values[order]
        cumulative = np.cumsum(weights[:, order], axis=1)
        total = cumulative[:, -1]
        result = []
        for probability in (.95, .99):
            target = (total - 1) * probability
            lower, upper = np.floor(target), np.ceil(target)
            lo = np.minimum((cumulative <= lower[:, None]).sum(axis=1), len(ordered) - 1)
            hi = np.minimum((cumulative <= upper[:, None]).sum(axis=1), len(ordered) - 1)
            result.append(ordered[lo] + (target - lower) * (ordered[hi] - ordered[lo]))
        return np.stack(result, axis=1)

    ratios = quantiles(qerror(prediction, truth)) / quantiles(qerror(raw, truth))
    return np.percentile(ratios, (2.5, 97.5), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "PERSEUS-verification-report.json")
    args = parser.parse_args()
    manifest = read_json(ROOT / "provenance/export_manifest.json")
    checksum_path = ROOT / "provenance/release_checksums.json"
    release_hashes = 0
    if checksum_path.exists():
        for name, expected in read_json(checksum_path).items():
            assert sha(ROOT / name) == expected, name
            release_hashes += 1
    for record in manifest["files"]:
        assert sha(ROOT / record["path"]) == record["release_sha256"], record["path"]
    split_reference = {(r["drift"], int(r["seed"]), r["method"], r["mode"]): r for r in rows(OUT / "test_per_split.csv")}
    pooled_reference = {(r["drift"], r["method"]): r for r in rows(OUT / "test_pooled_intervals.csv")}
    per_split_checks, pooled_checks, ci_checks = 0, 0, 0
    for drift_index, drift in enumerate(("column", "table", "value")):
        combined = defaultdict(list)
        for seed in (501, 502, 503):
            base = OUT / drift / "controls" / str(seed)
            split = read_json(base / "splits.json")
            report = read_json(base / "results.json")
            with np.load(base / "predictions.npz", allow_pickle=False) as stored:
                data = {key: stored[key] for key in stored.files}
            index = np.array(split["test"])
            groups = {role: set(data["sql_id"][indices]) for role, indices in split.items()}
            for left in groups:
                for right in groups:
                    if left != right:
                        assert not groups[left] & groups[right], (drift, seed, left, right)
            combined["truth"].append(data["truth"][index])
            combined["sql_id"].append(data["sql_id"][index])
            for family in ("raw", "scalar", *FAMILIES):
                prediction = data[family][index]
                check_row(metrics(prediction, data["truth"][index]), split_reference[drift, seed, family, "candidate"])
                per_split_checks += 1
                combined[family].append(prediction)
                if family in FAMILIES:
                    admitted = report["families"][family]["gates"]["default"]
                    routed = prediction if admitted else data["raw"][index]
                    check_row(metrics(routed, data["truth"][index]), split_reference[drift, seed, family, "gated"])
                    per_split_checks += 1
                    combined[family + "_gated"].append(routed)
        combined = {key: np.concatenate(value) for key, value in combined.items()}
        for name, prediction in combined.items():
            if name in {"truth", "sql_id"}:
                continue
            check_row(metrics(prediction, combined["truth"]), pooled_reference[drift, name])
            pooled_checks += 1
        if args.bootstrap:
            ci = bootstrap_intervals(combined["residual_hgb"], combined["raw"], combined["truth"],
                                     combined["sql_id"], 2026092990 + drift_index)
            reference = pooled_reference[drift, "residual_hgb"]
            for j, pct in enumerate((95, 99)):
                check_row({f"p{pct}_ratio_ci_low": ci[0, j], f"p{pct}_ratio_ci_high": ci[1, j]}, reference)
                ci_checks += 1
        print("Verified", drift, "predictions and SQL-grouped roles", flush=True)

    cases = {c["qid"]: c for c in read_json(ROOT / "rebuttal_20260928/protocol_attribution/cases.json") if c["role"] == "test"}
    blocks = defaultdict(list)
    requests = rows(OUT / "throughput/requests.csv")
    for request in requests:
        assert request["status"] == "ok"
        assert request["active"] == "pg" and request["admitted"] == "False"
        assert request["sql_id"] == cases[request["qid"]]["sql_id"]
        assert float(request["result_count"]) == float(cases[request["qid"]]["truth"])
        blocks[int(request["concurrency"]), int(request["repetition"]), request["policy"]].append(request)
    assert len(requests) == 6480 and len(blocks) == 12
    for block in blocks.values():
        assert len(block) == len(cases) == 540
        assert {r["qid"] for r in block} == set(cases)
    for block in rows(OUT / "throughput/blocks.csv"):
        key = int(block["concurrency"]), int(block["repetition"]), block["policy"]
        assert len(blocks[key]) == int(block["completed_correctly"])
        assert math.isclose(float(block["goodput_qps"]), len(blocks[key]) / float(block["elapsed_seconds"]), rel_tol=1e-12)
    result = {"release_hashes_verified": release_hashes,
              "exported_hashes_verified": len(manifest["files"]), "per_split_metric_rows_recomputed": per_split_checks,
              "pooled_metric_rows_recomputed": pooled_checks, "bootstrap_percentile_pairs_recomputed": ci_checks,
              "correct_native_requests": len(requests), "throughput_blocks_verified": len(blocks),
              "new_model_training": False, "new_database_execution": False}
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
