"""Fixed-budget estimator controls and matched legacy-population re-evaluation."""
from __future__ import annotations

import os

for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[env_name] = "2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import psutil
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import run_workload_drift_mscn_windows as mscn
import run_workload_drift_tail_residual_transport as repair
import run_workload_drift_optimizer_risk_proxy as legacy

FOLDERS = {"column": "workload_drift_mscn_windows_bitmap", "table": "workload_drift_mscn_windows_table_bitmap", "value": "workload_drift_mscn_windows_value_bitmap"}
SEEDS = (501, 502, 503)
FAMILIES = ("residual_hgb", "direct_hgb", "residual_rf", "direct_rf", "mscn_finetune", "mscn_retrain")
PROFILES = {
    "lenient": ((.20, .30, .50), (.15, .25), 1.25),
    "default": ((.30, .40, .60), (.25, .40), 1.20),
    "strict": ((.50, .70, .90), (.50, .70), 1.10),
}
LEGACY_DETAIL = ROOT / "paper/workload_drift_residual_field/results_duckdb_k1450/duckdb_execution_probe.csv"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relocated(value):
    value = str(value).replace("\\", "/")
    old = ""
    return ROOT / value[len(old):] if value.startswith(old) else ROOT / value


def emit(stage, **kwargs):
    print(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "stage": stage, **kwargs}), flush=True)


def identity(row):
    parts = row[2].split(",") if row[2] else []
    cond = [x.strip().lower() for x in row[1].split(",") if x.strip()]
    cond += [" ".join(x.strip().lower() for x in parts[i:i + 3]) for i in range(0, len(parts), 3)]
    value = [sorted(" ".join(x.lower().split()) for x in row[0].split(",")), sorted(cond)]
    return digest(json.dumps(value).encode())


def role_for(sid, seed):
    value = int(digest(f"4udk-v1|{seed}|{sid}".encode())[:16], 16) / 2**64
    return "fit" if value < .50 else "tune" if value < .65 else "gate" if value < .80 else "test"


def qerror(prediction, truth):
    p, t = np.maximum(1., prediction), np.maximum(1., truth)
    return np.maximum(p / t, t / p)


def metrics(prediction, truth):
    q = qerror(prediction, truth)
    return {"n": len(q), "median": float(np.median(q)), "p90": float(np.percentile(q, 90)),
            "p95": float(np.percentile(q, 95)), "p99": float(np.percentile(q, 99)),
            "max": float(q.max()), "mean": float(q.mean())}


def tuning_key(pred, truth, config_id):
    m = metrics(pred, truth)
    return (.5 * (math.log(m["p95"]) + math.log(m["p99"])), m["median"], config_id)


def gate(raw, scalar, pred, truth, profile):
    r, s, p = (metrics(x, truth) for x in (raw, scalar, pred))
    tails, globals_, med = PROFILES[profile]
    return (all(1 - p[k] / r[k] >= v for k, v in zip(("p90", "p95", "p99"), tails))
            and all(1 - p[k] / s[k] >= v for k, v in zip(("p95", "p99"), globals_))
            and p["median"] <= med * r["median"])


def load_data(drift):
    base = ROOT / "research_plan" / FOLDERS[drift] / "seed_1"
    met = read_json(base / "metrics.json")
    source_path = base / "data" / met["provenance"]["train_rel"]
    target_path = base / "data" / met["provenance"]["predict_rel"]
    rows, labels = repair.load_target_rows(target_path)
    source_rows, _ = repair.load_target_rows(source_path)
    raw = np.loadtxt(relocated(met["artifacts"]["predictions"]), dtype=np.float64)
    assert len(raw) == len(labels)
    sids = np.array([identity(row) for row in rows])
    first = {}
    for i, sid in enumerate(sids):
        if sid in first:
            assert labels[first[sid]] == labels[i]
        else:
            first[sid] = i
    source_ids = {identity(row) for row in source_rows}
    assert not set(sids) & source_ids, "Source checkpoint may have seen target SQL"
    met["artifacts"] = {k: str(relocated(v)) for k, v in met["artifacts"].items()}
    bitmap = met["provenance"]["bitmap_files"]
    enc_args = SimpleNamespace(num_materialized_samples=met["num_materialized_samples"],
        train_bitmap_file=str(relocated(bitmap["train_bitmap_file"])),
        predict_bitmap_file=str(relocated(bitmap["predict_bitmap_file"])))
    encoded = mscn.encode_dataset(source_path, target_path, base / "data/baselines/MSCN-modified/workloads/stats", met["seed"], enc_args)
    inputs = [source_path, target_path, Path(met["artifacts"]["model"]), Path(met["artifacts"]["predictions"]),
              Path(enc_args.train_bitmap_file), Path(enc_args.predict_bitmap_file), base / "metrics.json"]
    return {"rows": rows, "labels": labels, "raw": raw, "sids": sids, "unique": np.array(list(first.values())),
            "metrics": met, "encoded": encoded, "inputs": inputs, "source_unique_sql": len(source_ids)}


def make_splits(data, drift, seed):
    sizes = {"fit": 500, "tune": 250, "gate": 250} if drift == "value" else {"fit": 2000, "tune": 500, "gate": 500}
    roles = {role: [] for role in ("fit", "tune", "gate", "test")}
    for i in data["unique"]:
        roles[role_for(data["sids"][i], seed)].append(int(i))
    for role, n in sizes.items():
        roles[role].sort(key=lambda i: digest(f"rank|{seed}|{data['sids'][i]}".encode()))
        assert len(roles[role]) >= n, (drift, seed, role, len(roles[role]))
        roles[role] = roles[role][:n]
    assert len(set(sum(roles.values(), []))) == sum(map(len, roles.values()))
    return {k: np.array(v, dtype=np.int64) for k, v in roles.items()}


def train_trees(data, splits, family, seed, directory):
    fit, tune = splits["fit"], splits["tune"]
    vocab = repair.signature_vocab([data["rows"][i] for i in fit])
    features = repair.row_features(data["rows"], data["raw"], vocab)
    residual = family.startswith("residual")
    target = np.log(data["labels"])
    if residual:
        target -= np.log(np.maximum(1., data["raw"]))
    clip = np.percentile(target[fit], [1, 99])
    if family.endswith("hgb"):
        configs = [{"max_iter": n, "learning_rate": lr, "max_leaf_nodes": 31, "l2_regularization": .01} for lr in (.05, .10) for n in (150, 300)]
        make = HistGradientBoostingRegressor
    else:
        configs = [{"n_estimators": 250, "max_depth": depth, "min_samples_leaf": leaf, "n_jobs": 2} for depth in (10, 14) for leaf in (3, 5)]
        make = RandomForestRegressor
    best_key, best_model, best_config = None, None, None
    records = []
    for cid, cfg in enumerate(configs):
        started = time.perf_counter()
        model = make(**cfg, random_state=seed)
        model.fit(features[fit], np.clip(target[fit], *clip))
        tpred = np.exp(np.clip(model.predict(features[tune]), *clip))
        if residual:
            tpred *= np.maximum(1., data["raw"][tune])
        key = tuning_key(tpred, data["labels"][tune], cid)
        records.append({"family": family, "config_id": cid, "config": cfg, "tuning": metrics(tpred, data["labels"][tune]),
                        "fit_and_tune_seconds": time.perf_counter() - started})
        if best_key is None or key < best_key:
            best_key, best_model, best_config = key, model, cid
    record = {"family": family, "selected_config_id": best_config, "candidates": records,
              "clip": clip.tolist(), "selected_before_gate_and_test": True}
    write_json(directory / f"{family}_selection.json", record)
    joblib.dump({"model": best_model, "vocab": vocab, "clip": clip}, directory / f"{family}.joblib")
    pred = np.exp(np.clip(best_model.predict(features), *clip))
    if residual:
        pred *= np.maximum(1., data["raw"])
    return np.maximum(1., pred), record


def pad_parts(torch, tensors, maxima):
    out = []
    for i, part in enumerate(tensors):
        if i == 3:
            out.append(part)
            continue
        target = maxima[i if i < 3 else i - 4]
        out.append(torch.nn.functional.pad(part, (0, 0, 0, target - part.shape[1])))
    return out


def model_forward(model, tensors, idx):
    samples, predicates, joins, _targets, sm, pm, jm = [t[idx] for t in tensors]
    return model(samples, predicates, joins, sm, pm, jm).view(-1)


def neural_predict(torch, model, tensors, indices, lo, hi):
    model.eval()
    results = []
    with torch.no_grad():
        for offset in range(0, len(indices), 512):
            out = model_forward(model, tensors, indices[offset:offset + 512])
            results.append(torch.exp(out * (hi - lo) + lo).cpu().numpy())
    return np.maximum(1., np.round(np.concatenate(results))).astype(np.float64)


def train_neural(torch, data, splits, family, seed, directory, target_gpu, source_gpu, device):
    from mscn.model import SetConv
    dims, met, enc = data["encoded"]["feature_dims"], data["metrics"], data["encoded"]
    fit, tune = splits["fit"], splits["tune"]
    fit_tensors = [t[torch.as_tensor(fit, device=device)] for t in target_gpu]
    train_log = np.log(data["labels"][fit])
    if family == "mscn_retrain":
        maxima = [max(source_gpu[i].shape[1], fit_tensors[i].shape[1]) for i in range(3)]
        src, tgt = pad_parts(torch, source_gpu, maxima), pad_parts(torch, fit_tensors, maxima)
        train_tensors = [torch.cat((s, t), dim=0) for s, t in zip(src, tgt)]
        train_log = np.r_[np.log(np.asarray(enc["labels"]["train"], dtype=float)), train_log]
        lo, hi = float(train_log.min()), float(train_log.max())
    else:
        train_tensors = fit_tensors
        lo, hi = float(enc["min_val"]), float(enc["max_val"])
    train_truth = torch.tensor(train_log, dtype=torch.float32, device=device)
    tune_idx = torch.as_tensor(tune, device=device)
    records, histories = [], []
    best_key, best_state, best_id = None, None, None
    for rate_index, lr in enumerate((1e-4, 1e-3)):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SetConv(dims["sample_feats"], dims["predicate_feats"], dims["join_feats"], met["hidden_units"]).to(device)
        if family == "mscn_finetune":
            model.load_state_dict(torch.load(met["artifacts"]["model"], map_location=device, weights_only=True))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        generator = torch.Generator(device=device).manual_seed(seed)
        started = time.perf_counter()
        for epoch in range(1, 61):
            model.train()
            order = torch.randperm(len(train_log), device=device, generator=generator)
            losses = []
            for offset in range(0, len(order), 256):
                idx = order[offset:offset + 256]
                optimizer.zero_grad(set_to_none=True)
                logpred = model_forward(model, train_tensors, idx) * (hi - lo) + lo
                loss = torch.abs(logpred - train_truth[idx]).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss: {family}, {seed}, {lr}, {epoch}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            histories.append({"lr": lr, "epoch": epoch, "loss": float(np.mean(losses))})
            if epoch in (20, 60):
                cid = rate_index * 2 + (epoch == 60)
                pred = neural_predict(torch, model, target_gpu, tune_idx, lo, hi)
                key = tuning_key(pred, data["labels"][tune], int(cid))
                record = {"family": family, "config_id": int(cid), "config": {"lr": lr, "epochs": epoch},
                          "tuning": metrics(pred, data["labels"][tune]), "cumulative_fit_and_tune_seconds": time.perf_counter() - started}
                records.append(record)
                if best_key is None or key < best_key:
                    best_key, best_id = key, int(cid)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                emit("neural_checkpoint", drift=directory.parent.name, seed=seed, family=family, lr=lr, epoch=epoch,
                     tuning_p95=record["tuning"]["p95"], tuning_p99=record["tuning"]["p99"])
        del optimizer, model
    record = {"family": family, "selected_config_id": best_id, "candidates": records, "output_log_min": lo,
              "output_log_max": hi, "source_training_rows": len(enc["labels"]["train"]) if family == "mscn_retrain" else 0,
              "target_training_rows": len(fit), "selected_before_gate_and_test": True,
              "original_output_range_exceeded_by_fit_labels": int(np.count_nonzero((np.log(data["labels"][fit]) < enc["min_val"]) | (np.log(data["labels"][fit]) > enc["max_val"])))}
    write_json(directory / f"{family}_selection.json", record)
    write_csv(directory / f"{family}_losses.csv", histories)
    torch.save({"state_dict": best_state, "lo": lo, "hi": hi}, directory / f"{family}.pt")
    model = SetConv(dims["sample_feats"], dims["predicate_feats"], dims["join_feats"], met["hidden_units"]).to(device)
    model.load_state_dict(best_state)
    predictions = neural_predict(torch, model, target_gpu, torch.arange(len(data["labels"]), device=device), lo, hi)
    del model, train_tensors, fit_tensors, train_truth
    return predictions, record


def run_main(torch, data, drift, directory, device):
    directory.mkdir(exist_ok=True)
    target_gpu = [t.to(device) for t in data["encoded"]["predict_dataset"].tensors]
    source_gpu = [t.to(device) for t in data["encoded"]["train_dataset"].tensors]
    dims, enc = data["encoded"]["feature_dims"], data["encoded"]
    from mscn.model import SetConv
    probe = SetConv(dims["sample_feats"], dims["predicate_feats"], dims["join_feats"], data["metrics"]["hidden_units"]).to(device)
    probe.load_state_dict(torch.load(data["metrics"]["artifacts"]["model"], map_location=device, weights_only=True))
    reproduced = neural_predict(torch, probe, target_gpu, torch.arange(len(data["labels"]), device=device), float(enc["min_val"]), float(enc["max_val"]))
    assert np.allclose(reproduced, data["raw"], rtol=1e-4, atol=2), "Checkpoint input reconstruction does not match frozen baseline"
    write_json(directory / "checkpoint_check.json", {"exact_match_fraction": float(np.mean(reproduced == data["raw"])),
                "max_relative_difference": float(np.max(np.abs(reproduced - data["raw"]) / np.maximum(1., data["raw"]))),
                "within_tolerance": True})
    del probe
    membership = []
    for seed in SEEDS:
        subdir = directory / str(seed)
        subdir.mkdir()
        splits = make_splits(data, drift, seed)
        write_json(subdir / "splits.json", {k: v.tolist() for k, v in splits.items()})
        for role, ids in splits.items():
            membership.extend({"outer": seed, "drift": drift, "role": role, "row_index": int(i), "sql_id": data["sids"][i]} for i in ids)
        pred = {"raw": data["raw"]}
        fit = splits["fit"]
        shift = repair.tail_risk_shift(np.log(data["labels"][fit]) - np.log(np.maximum(1., data["raw"][fit])), 1.5, .005)
        pred["scalar"] = np.maximum(1., data["raw"] * math.exp(shift))
        reports = {}
        for family in FAMILIES:
            started = time.perf_counter()
            if family.startswith("mscn"):
                values, selection = train_neural(torch, data, splits, family, seed, subdir, target_gpu, source_gpu, device)
            else:
                values, selection = train_trees(data, splits, family, seed, subdir)
            assert np.isfinite(values).all() and np.min(values) >= 1
            pred[family] = values
            gate_ids, test_ids = splits["gate"], splits["test"]
            reports[family] = {"seconds": time.perf_counter() - started, "selected_config_id": selection["selected_config_id"],
                "gates": {p: gate(pred["raw"][gate_ids], pred["scalar"][gate_ids], values[gate_ids], data["labels"][gate_ids], p) for p in PROFILES},
                "gate_metrics": metrics(values[gate_ids], data["labels"][gate_ids]), "final_test_metrics": metrics(values[test_ids], data["labels"][test_ids])}
            emit("family_complete", drift=drift, seed=seed, family=family, seconds=reports[family]["seconds"],
                 gates=reports[family]["gates"], test_p95=reports[family]["final_test_metrics"]["p95"], available_memory=psutil.virtual_memory().available)
            write_json(subdir / "progress.json", reports)
        np.savez_compressed(subdir / "predictions.npz", truth=data["labels"], sql_id=data["sids"], **pred)
        write_json(subdir / "results.json", {"outer": seed, "drift": drift, "counts": {k: len(v) for k, v in splits.items()},
                   "scalar_shift": shift, "families": reports, "raw_final": metrics(pred["raw"][splits["test"]], data["labels"][splits["test"]])})
    write_csv(directory / "membership.csv", membership)
    del target_gpu, source_gpu
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def run_matched(torch, data, drift, directory):
    directory.mkdir(exist_ok=True)
    original = [r for r in read_csv(LEGACY_DETAIL) if r["drift"] == drift]
    summaries, matches, membership = [], [], []
    predictions = {}
    for seed in (1, 2, 3):
        order = np.random.default_rng(seed).permutation(len(data["labels"]))
        nfit = 500 if drift == "value" else 2000
        fit, test = order[:nfit], order[nfit:]
        retained_rows = [r for r in original if int(r["seed"]) == seed and r["method"] == "raw-mscn"]
        retained = {int(r["query_index"]) for r in retained_rows}
        assert len(retained) == 1450 and retained <= set(test)
        for row in retained_rows:
            i = int(row["query_index"])
            assert int(row["actual_count"]) == int(row["label"]) == int(data["labels"][i])
        started = time.perf_counter()
        residual = legacy.compute_residual_predictions(data["rows"], data["labels"], data["raw"], fit, test, seed)
        args = SimpleNamespace(finetune_batch_size=256, finetune_lr=.001, finetune_epochs=20, threads=2)
        tuned, seconds, losses = legacy.compute_finetune_predictions(data["metrics"], data["encoded"], fit, test, seed, args)
        pred = {"raw-mscn": data["raw"][test], "residual-field": residual, "mscn-finetune-best": tuned}
        kept = np.isin(test, sorted(retained))
        in_range = (data["labels"][test] >= 10) & (data["labels"][test] <= 1000000)
        masks = {"full": np.ones(len(test), dtype=bool), "in_range": in_range, "retained": kept,
                 "in_range_not_retained": in_range & ~kept, "out_of_range": ~in_range}
        for i, original_i in enumerate(test):
            membership.append({"drift": drift, "seed": seed, "row_index": int(original_i), "sql_id": data["sids"][original_i],
                               "in_range": bool(in_range[i]), "retained": bool(kept[i])})
        for method, vals in pred.items():
            lookup = {int(i): float(v) for i, v in zip(test, vals)}
            saved = [r for r in original if int(r["seed"]) == seed and r["method"] == method]
            saved_values = np.array([float(r["prediction"]) for r in saved])
            new_values = np.array([lookup[int(r["query_index"])] for r in saved])
            matches.append({"drift": drift, "seed": seed, "method": method, "retained_cases": len(saved),
                            "exact_prediction_fraction": float(np.mean(new_values == saved_values)),
                            "rtol_1e_6_atol_1_match_fraction": float(np.mean(np.isclose(new_values, saved_values, rtol=1e-6, atol=1))),
                            "max_relative_prediction_difference": float(np.max(np.abs(new_values - saved_values) / np.maximum(1., saved_values)))})
            predictions[f"s{seed}_{method}"] = vals
            for name, mask in masks.items():
                if not mask.any():
                    continue
                summaries.append({"drift": drift, "seed": seed, "population": name, "method": method,
                                  **metrics(vals[mask], data["labels"][test][mask])})
        predictions[f"s{seed}_indices"] = test
        predictions[f"s{seed}_retained"] = kept
        predictions[f"s{seed}_in_range"] = in_range
        write_json(directory / f"seed_{seed}_fit.json", {"finetune_seconds": seconds, "losses": losses,
                       "fit_indices": fit.tolist(), "eval_indices": test.tolist(), "total_seconds": time.perf_counter() - started})
        emit("matched_population", drift=drift, seed=seed, seconds=time.perf_counter() - started)
    np.savez_compressed(directory / "predictions.npz", truth=data["labels"], sql_id=data["sids"], **predictions)
    write_csv(directory / "per_seed.csv", summaries)
    write_csv(directory / "saved_prediction_agreement.csv", matches)
    write_csv(directory / "population_membership.csv", membership)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", choices=list(FOLDERS), required=True)
    args = parser.parse_args()
    directory = OUT / args.drift
    if directory.exists():
        raise RuntimeError(f"Refusing to overwrite {directory}; inspect any partial run before resuming")
    directory.mkdir()
    started = time.perf_counter()
    torch, _ = mscn.require_torch()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "drift": args.drift,
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "device": device,
                "command": sys.argv, "protocol_hash": file_hash(OUT / "PROTOCOL.md"), "runner_hash": file_hash(__file__),
                "dependencies": {p.name: file_hash(p) for p in (Path(mscn.__file__), Path(repair.__file__), Path(legacy.__file__))}}
    write_json(directory / "started.json", manifest)
    emit("start", drift=args.drift, device=device)
    with threadpool_limits(limits=2):
        data = load_data(args.drift)
        manifest["input_hashes"] = {str(p): file_hash(p) for p in data["inputs"]}
        manifest["source_unique_sql"] = data["source_unique_sql"]
        manifest["within_drift_source_target_sql_overlap"] = 0
        write_json(directory / "started.json", manifest)
        run_main(torch, data, args.drift, directory / "controls", device)
        run_matched(torch, data, args.drift, directory / "matched_population")
    manifest.update({"completed_at": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started,
                     "output_hashes": {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*") if p.is_file() and p.name != "completed.json"}})
    write_json(directory / "completed.json", manifest)
    emit("complete", drift=args.drift, seconds=manifest["seconds"])


if __name__ == "__main__":
    main()
