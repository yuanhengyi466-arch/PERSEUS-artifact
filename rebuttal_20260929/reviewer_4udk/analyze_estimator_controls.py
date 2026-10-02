"""Conditional SQL-cluster intervals and paired population summaries."""
from __future__ import annotations

import argparse
import gc
from collections import defaultdict

import numpy as np

from run_estimator_controls import (
    OUT, FOLDERS, SEEDS, FAMILIES, PROFILES, LEGACY_DETAIL, file_hash, read_json, read_csv,
    write_json, write_csv, metrics, qerror, emit,
)


def weighted_quantiles(values, weights, probabilities):
    order = np.argsort(values)
    vals = values[order]
    cumulative = np.cumsum(weights[:, order], axis=1)
    totals = cumulative[:, -1]
    out = []
    for prob in probabilities:
        target = (totals - 1) * prob
        lower, upper = np.floor(target), np.ceil(target)
        lo = np.minimum((cumulative <= lower[:, None]).sum(axis=1), len(vals) - 1)
        hi = np.minimum((cumulative <= upper[:, None]).sum(axis=1), len(vals) - 1)
        result = vals[lo] + (target - lower) * (vals[hi] - vals[lo])
        result[totals == 0] = np.nan
        out.append(result)
    return np.stack(out, axis=1)


def weights_for(ids, seed, repeats=2000):
    unique, inverse = np.unique(ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=repeats)
    return counts[:, inverse].astype(np.int32)


def interval(array):
    return np.percentile(array, [2.5, 97.5]).tolist()


def analyze_controls():
    per_split, pooled, paired, sensitivity, resources, membership = [], [], [], [], [], []
    for drift_index, drift in enumerate(FOLDERS):
        directory = OUT / drift
        done = read_json(directory / "completed.json")
        assert done["protocol_hash"] == file_hash(OUT / "PROTOCOL.md")
        for path, expected in done["input_hashes"].items():
            assert file_hash(path) == expected, path
        for rel, expected in done["output_hashes"].items():
            assert file_hash(directory / rel) == expected, rel
        drift_membership = read_csv(directory / "controls/membership.csv")
        membership.extend(drift_membership)
        combined = defaultdict(list)
        for seed in SEEDS:
            base = directory / "controls" / str(seed)
            results = read_json(base / "results.json")
            splits = {k: np.asarray(v) for k, v in read_json(base / "splits.json").items()}
            with np.load(base / "predictions.npz") as stored:
                data = {k: stored[k] for k in stored.files}
            test, gate = splits["test"], splits["gate"]
            combined["sql_id"].append(data["sql_id"][test])
            combined["truth"].append(data["truth"][test])
            gate_weights = weights_for(data["sql_id"][gate], 2026092900 + seed + drift_index * 100)
            raw_boot = weighted_quantiles(qerror(data["raw"][gate], data["truth"][gate]), gate_weights, (.5, .9, .95, .99))
            scalar_boot = weighted_quantiles(qerror(data["scalar"][gate], data["truth"][gate]), gate_weights, (.95, .99))
            raw_metrics = metrics(data["raw"][test], data["truth"][test])
            for family in ("raw", "scalar", *FAMILIES):
                values = data[family]
                combined[family].append(values[test])
                stat = metrics(values[test], data["truth"][test])
                per_split.append({"drift": drift, "seed": seed, "method": family, "mode": "candidate", **stat,
                                  "p95_ratio_vs_raw": stat["p95"] / raw_metrics["p95"], "p99_ratio_vs_raw": stat["p99"] / raw_metrics["p99"]})
                if family not in FAMILIES:
                    continue
                passed = results["families"][family]["gates"]["default"]
                routed = values[test] if passed else data["raw"][test]
                combined[family + "_gated"].append(routed)
                routed_stat = metrics(routed, data["truth"][test])
                per_split.append({"drift": drift, "seed": seed, "method": family, "mode": "gated", **routed_stat,
                                  "p95_ratio_vs_raw": routed_stat["p95"] / raw_metrics["p95"], "p99_ratio_vs_raw": routed_stat["p99"] / raw_metrics["p99"]})
                boot = weighted_quantiles(qerror(values[gate], data["truth"][gate]), gate_weights, (.5, .9, .95, .99))
                for profile, (tail, glob, med) in PROFILES.items():
                    tail_red = 1 - boot[:, 1:] / raw_boot[:, 1:]
                    global_red = 1 - boot[:, 2:] / scalar_boot
                    passed_boot = ((tail_red >= np.array(tail)).all(axis=1)
                                   & (global_red >= np.array(glob)).all(axis=1)
                                   & (boot[:, 0] <= med * raw_boot[:, 0]))
                    low95, high95 = interval(tail_red[:, 1])
                    low99, high99 = interval(tail_red[:, 2])
                    sensitivity.append({"drift": drift, "seed": seed, "method": family, "profile": profile,
                        "gate_sql": len(gate), "observed_pass": results["families"][family]["gates"][profile],
                        "bootstrap_pass_fraction": float(passed_boot.mean()), "bootstrap_resamples": len(passed_boot),
                        "p95_reduction_ci_low": low95, "p95_reduction_ci_high": high95,
                        "p99_reduction_ci_low": low99, "p99_reduction_ci_high": high99})
                selection = read_json(base / f"{family}_selection.json")
                resources.append({"drift": drift, "seed": seed, "method": family,
                    "fit_labels": len(splits["fit"]), "tune_labels": len(splits["tune"]), "gate_labels": len(gate),
                    "total_target_labels": sum(len(splits[r]) for r in ("fit", "tune", "gate")),
                    "test_sql": len(test), "configurations_compared": len(selection["candidates"]),
                    "selected_configuration": selection["selected_config_id"],
                    "fitting_selection_and_prediction_seconds": results["families"][family]["seconds"],
                    "source_training_rows_replayed": selection.get("source_training_rows", 0),
                    "frozen_admissible": not family.startswith("mscn_"),
                    "label_acquisition_time": "not_measured"})
            del gate_weights, data, boot, raw_boot, scalar_boot
            emit("gate_analysis", drift=drift, seed=seed)
        combined = {k: np.concatenate(v) for k, v in combined.items()}
        weights = weights_for(combined["sql_id"], 2026092990 + drift_index)
        qraw = qerror(combined["raw"], combined["truth"])
        raw_ci_quantiles = weighted_quantiles(qraw, weights, (.95, .99))
        raw_stat = metrics(combined["raw"], combined["truth"])
        all_boot = {}
        for name in combined:
            if name in ("sql_id", "truth"):
                continue
            values = combined[name]
            boot = weighted_quantiles(qerror(values, combined["truth"]), weights, (.95, .99))
            stat = metrics(values, combined["truth"])
            record = {"drift": drift, "method": name, "test_cases": len(values),
                      "unique_sql": len(np.unique(combined["sql_id"])), **stat}
            for j, pct in enumerate((95, 99)):
                ratio = stat[f"p{pct}"] / raw_stat[f"p{pct}"]
                ci = interval(boot[:, j] / raw_ci_quantiles[:, j])
                record.update({f"p{pct}_ratio_vs_raw": ratio, f"p{pct}_reduction": 1 - ratio,
                               f"p{pct}_ratio_ci_low": ci[0], f"p{pct}_ratio_ci_high": ci[1],
                               f"p{pct}_reduction_ci_low": 1 - ci[1], f"p{pct}_reduction_ci_high": 1 - ci[0]})
            pooled.append(record)
            all_boot[name] = (boot, stat)
        comparisons = [("residual_hgb", "direct_hgb"), ("residual_rf", "direct_rf"),
                       ("residual_hgb", "mscn_finetune"), ("residual_hgb", "mscn_retrain"),
                       ("residual_hgb", "residual_rf")]
        for left, right in comparisons:
            for mode in ("candidate", "gated"):
                a, b = (x + ("_gated" if mode == "gated" else "") for x in (left, right))
                record = {"drift": drift, "mode": mode, "numerator": left, "denominator": right}
                for j, pct in enumerate((95, 99)):
                    ci = interval(all_boot[a][0][:, j] / all_boot[b][0][:, j])
                    record.update({f"p{pct}_ratio": all_boot[a][1][f"p{pct}"] / all_boot[b][1][f"p{pct}"],
                                   f"p{pct}_ci_low": ci[0], f"p{pct}_ci_high": ci[1]})
                paired.append(record)
        del combined, weights, all_boot
        gc.collect()
        emit("pooled_analysis", drift=drift)

    per_outer = defaultdict(lambda: defaultdict(set))
    for row in membership:
        per_outer[row["outer"]][row["role"]].add(row["sql_id"])
    budgets = []
    for outer, roles in per_outer.items():
        keys = sorted(roles)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                assert not roles[a] & roles[b], (outer, a, b)
        budgets.append({"outer": outer, **{r + "_unique_sql": len(s) for r, s in roles.items()},
                        "total_supervision_unique_sql": len(roles["fit"] | roles["tune"] | roles["gate"])})
    write_csv(OUT / "test_per_split.csv", per_split)
    write_csv(OUT / "test_pooled_intervals.csv", pooled)
    write_csv(OUT / "paired_method_comparisons.csv", paired)
    write_csv(OUT / "gate_bootstrap_sensitivity.csv", sensitivity)
    write_csv(OUT / "resource_accounting.csv", resources)
    write_csv(OUT / "cross_drift_unique_label_budget.csv", budgets)
    write_csv(OUT / "sql_membership.csv", membership)
    return per_split, pooled, paired, sensitivity


def analyze_population():
    records, agreement = [], []
    original = read_csv(LEGACY_DETAIL)
    for drift in FOLDERS:
        base = OUT / drift / "matched_population"
        per_seed = read_csv(base / "per_seed.csv")
        with np.load(base / "predictions.npz") as stored:
            data = {k: stored[k] for k in stored.files}
        groups = defaultdict(list)
        for seed in (1, 2, 3):
            ids = data[f"s{seed}_indices"]
            rng, kept = data[f"s{seed}_in_range"], data[f"s{seed}_retained"]
            masks = {"full": np.ones(len(ids), dtype=bool), "in_range": rng, "retained": kept,
                     "in_range_not_retained": rng & ~kept, "out_of_range": ~rng}
            for method in ("raw-mscn", "residual-field", "mscn-finetune-best"):
                # The archived CSV stores predictions after the q-error floor at 1.
                saved = [r for r in original if r["drift"] == drift
                         and int(r["seed"]) == seed and r["method"] == method]
                lookup = dict(zip(ids.tolist(), data[f"s{seed}_{method}"].tolist()))
                old = np.maximum(1., np.array([float(r["prediction"]) for r in saved]))
                new = np.maximum(1., np.array([lookup[int(r["query_index"])] for r in saved]))
                truth = np.array([float(r["label"]) for r in saved])
                assert len(saved) == int(kept.sum()) == 1450
                relative = np.abs(new - old) / old
                old_stat, new_stat = metrics(old, truth), metrics(new, truth)
                diagnostic = {"drift": drift, "seed": seed, "method": method,
                    "retained_cases": len(saved), "normalization": "max(1,prediction)",
                    "exact_prediction_fraction": float(np.mean(new == old)),
                    "rtol_1e_6_atol_1_match_fraction": float(np.mean(np.isclose(new, old, rtol=1e-6, atol=1))),
                    "max_relative_prediction_difference": float(relative.max()),
                    "median_relative_prediction_difference": float(np.median(relative)),
                    "p95_relative_prediction_difference": float(np.percentile(relative, 95))}
                for name in ("median", "p95", "p99"):
                    diagnostic.update({f"saved_{name}": old_stat[name], f"rerun_{name}": new_stat[name],
                                       f"{name}_rerun_over_saved": new_stat[name] / old_stat[name]})
                agreement.append(diagnostic)
                for pop, mask in masks.items():
                    if mask.any():
                        groups[(pop, method)].append((data[f"s{seed}_{method}"][mask], data["truth"][ids[mask]], data["sql_id"][ids[mask]]))
        for (pop, method), parts in groups.items():
            pred, truth, sids = (np.concatenate([p[i] for p in parts]) for i in range(3))
            stat = metrics(pred, truth)
            records.append({"drift": drift, "population": pop, "method": method, "aggregation": "pooled_seed_query_records",
                            "unique_sql": len(np.unique(sids)), **stat})
            matching = [r for r in per_seed if r["population"] == pop and r["method"] == method]
            aggregate = {k: float(np.median([float(r[k]) for r in matching])) for k in stat}
            records.append({"drift": drift, "population": pop, "method": method, "aggregation": "median_of_per_seed_statistics",
                            "unique_sql": len(np.unique(sids)), **aggregate})
    write_csv(OUT / "matched_population_summary.csv", records)
    write_csv(OUT / "legacy_prediction_reproduction.csv", agreement)
    return records, agreement


def report(per_split, pooled, paired, sensitivity, population, agreement):
    gate_counts = defaultdict(int)
    gate_frequencies = defaultdict(list)
    for row in sensitivity:
        if row["profile"] == "default":
            key = (row["drift"], row["method"])
            gate_counts[key] += row["observed_pass"]
            gate_frequencies[key].append(row["bootstrap_pass_fraction"])
    lines = ["# Matched-Supervision and Population Audit Results", "", "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run + validate",
             "- Origin Date: 2026-09-29", "- Verification Status: ANALYZED; training completed and saved outputs/hash/split checks passed",
             "- Version Label: reviewer_4udk_v1", "", "## Matched Target Supervision", "",
             "Three outer splits, six model families and four fixed tuning configurations per family. Structural budgets are 2,000 fit + 500 tune + 500 gate labels; value uses 500 + 250 + 250. Test SQL is separate. Full retraining starts from random initialization and replays original source-training rows plus target fitting SQL. Fine-tuning starts from the source checkpoint. These are weight-updating comparators.",
             "", "Pooled percentiles aggregate test cases across trials; repeated canonical SQL are clustered for 2,000 bootstrap resamples. Per-trial results are saved separately. Ratios below 1 favor the candidate over raw MSCN.", "",
             "| Drift | Method | P95 ratio [95% CI] | P99 ratio [95% CI] | Default gate passes |",
             "|---|---|---:|---:|---:|"]
    for row in pooled:
        if row["method"] not in FAMILIES:
            continue
        lines.append(f"| {row['drift']} | {row['method']} | {row['p95_ratio_vs_raw']:.4f} [{row['p95_ratio_ci_low']:.4f}, {row['p95_ratio_ci_high']:.4f}] | {row['p99_ratio_vs_raw']:.4f} [{row['p99_ratio_ci_low']:.4f}, {row['p99_ratio_ci_high']:.4f}] | {gate_counts[row['drift'], row['method']]}/3 |")
    lines += ["", "## Direct Attribution Comparisons", "",
              "All comparisons use the same test SQL. Ratios below 1 favor the numerator. Candidate results below are accompanied by gated/raw-fallback results in paired_method_comparisons.csv.", "",
              "| Drift | Numerator / denominator | P95 ratio [95% CI] | P99 ratio [95% CI] |", "|---|---|---:|---:|"]
    for row in paired:
        if row["mode"] == "candidate":
            lines.append(f"| {row['drift']} | {row['numerator']} / {row['denominator']} | {row['p95_ratio']:.3f} [{row['p95_ci_low']:.3f}, {row['p95_ci_high']:.3f}] | {row['p99_ratio']:.3f} [{row['p99_ci_low']:.3f}, {row['p99_ci_high']:.3f}] |")
    lines += ["", "## Gate Resampling", "", "Pass fractions below describe sensitivity to the independent gate sample, conditional on the selected model. They are not probabilities of safety.", "",
              "| Drift | Method | Observed passes | Bootstrap pass fractions by outer split |", "|---|---|---:|---|"]
    for (drift, method), fractions in gate_frequencies.items():
        lines.append(f"| {drift} | {method} | {gate_counts[drift, method]}/3 | " + ", ".join(f"{x:.3f}" for x in fractions) + " |")
    lines += ["", "## Matched Population Re-evaluation", "",
              "Fixed legacy predictions for each adaptation seed are evaluated on full and retained populations with identical labels/configurations. This comparison isolates population membership within the rerun. Medians of per-seed quantiles and pooled quantiles are both retained in matched_population_summary.csv.", "",
              "| Drift | Population | Method | P95 | P99 | Seed-query cases |", "|---|---|---|---:|---:|---:|"]
    for row in population:
        if row["aggregation"] == "pooled_seed_query_records" and row["population"] in ("full", "retained"):
            lines.append(f"| {row['drift']} | {row['population']} | {row['method']} | {row['p95']:.2f} | {row['p99']:.2f} | {row['n']} |")
    weakest = min(float(r["rtol_1e_6_atol_1_match_fraction"]) for r in agreement)
    lines += ["", f"Minimum retained-prediction agreement across method/drift/seed records at rtol=1e-6, atol=1: {weakest:.6f}. Both predictions are floored at 1, matching the archived q-error computation. Effective prediction agreement and old/rerun retained quantiles are in legacy_prediction_reproduction.csv; original unfloored diagnostics remain in each drift directory. Any imperfect match must be reported as a rerun, not byte-identical recovery.",
              "", "## Statistical Scope", "",
              "1. Aggregation: pooled and per-split summaries are separate; population audit additionally retains both aggregation conventions.",
              "2. Unit of inference: canonical SQL clusters, not seed-query rows treated as independent.",
              "3. Selection: all new target SQL assigned to the test role are kept; legacy retained SQL remains a selected executable population.",
              "4. Conditioning: legacy count matching and cardinality caps affect the evaluated tails; no causal diagnosis of each exclusion is inferred.",
              "5. Base rates: all model families, gate failures and raw fallback outcomes are retained.",
              "6. Regression to the mean: tuning, independent gate, and final test are separated; the benchmark itself was already studied.",
              "7. Survivorship: experiment failures must remain visible in run logs; no unfavorable completed family is removed.",
              "8. Multiple comparisons: intervals are descriptive and unadjusted; no omnibus superiority claim follows from one favorable cell.",
              "9. Researcher choices: fixed local protocol/configurations preceded new fitting; no public preregistration claim.",
              "10. Causality: paired comparisons control labels and split membership; equal labels/configuration counts do not mean equal model capacity or training compute.",
              "11. Temporal order: only tuning chooses configurations; gate/test metrics do not retune. Bootstrap conditions on fitted/selected models and is not full selection-process inference.",
              "", "## Remaining Scope", "",
              "No concurrent DBMS throughput, cross-dataset native transfer, or Warper/ALECE/Robust-MSCN implementation is supplied by this estimator experiment. The MSCN full-retraining alternative is evaluated explicitly. Label acquisition time is unavailable for pre-existing labels. The matched legacy rerun retains the old normalization semantics; the new controls train on true log labels."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "analysis.json", {"outer_seeds": list(SEEDS), "families": list(FAMILIES), "model_selections": 54,
        "configurations_scored": 216, "bootstrap_resamples": 2000, "canonical_role_overlap": 0,
        "pooled_intervals": pooled, "paired_comparisons": paired,
        "minimum_legacy_prediction_agreement": weakest,
        "hashes": {p.name: file_hash(p) for p in OUT.glob("*.csv")}})


def main():
    argparse.ArgumentParser().parse_args()
    per_split, pooled, paired, sensitivity = analyze_controls()
    population, agreement = analyze_population()
    report(per_split, pooled, paired, sensitivity, population, agreement)
    emit("analysis_complete", selections=54, configurations=216)


if __name__ == "__main__":
    main()
