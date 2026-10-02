# Reviewer 4UdK: Matched Supervision and Population Audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-29
- Verification Status: prospective specification; see generated results for completion
- Version Label: reviewer_4udk_v1

## Questions

1. Does residual repair retain an advantage against validation-tuned MSCN fine-tuning and full MSCN retraining with the same target supervision?
2. How do the full estimator and executable-SQL populations differ when evaluated with identical predictions, labels, configurations and aggregation?
3. How uncertain are final q-error improvements and the independently measured gate decisions?

Existing MSCN/STATS results motivated the questions. This local plan precedes the new training; it is not a public preregistration or an unseen external benchmark. No positive outcome is required.

## A. New Matched-Supervision Controls

- Outer seeds 501/502/503; all canonical copies of a target SQL, including across drift files, share a role within an outer split. Hash proportions are .50 fitting / .15 tuning / .15 gating / .20 final test. Distinct outer trials may reuse SQL; inference clusters that reuse.
- Per structural drift, take 2,000/500/500 SQL from the first three roles by a fixed hash order. Value drift uses 500/250/250. Final test uses all distinct SQL assigned to the test role. Record and check source-training overlap within each estimator's drift.
- Total target supervision is 3,000 structural / 1,000 value labels per drift and trial. Tuning and gate labels are distinct. No gate or final-test label participates in fitting or hyperparameter selection. Report cross-drift and cross-trial reuse separately.
- Each model family has four fixed configurations. Select the lowest geometric mean of tuning P95 and P99 q-error; break ties by tuning median then configuration ID. Apply the selected model once to the separate gate and final test.
- Residual/direct HGB: learning rate .05/.10 crossed with 150/300 iterations, 31 leaves, L2 .01. Residual/direct RF: depth 10/14 crossed with minimum leaf 3/5, 250 trees. Direct and residual versions share query-signature and cached prediction-scale features, fitted on the fitting partition. Clip the fitted target and predicted output at fitting-target percentiles 1/99; the target is log cardinality for direct learners and log residual for residual learners.
- MSCN fine-tuning: source checkpoint, learning rate 1e-4/1e-3, evaluated at 20/60 epochs. Full retraining: fresh SetConv with the same width, trained over the original source training partition plus target fitting SQL, same learning-rate/epoch grid. Retraining normalizes its output on the source-plus-fitting-label range; fine-tuning retains the checkpoint output scale. Both train against true log labels using the existing MSCN log-q-error objective. Target labels are not silently clipped for these new controls.
- Training uses Windows-native PyTorch CUDA if available, one GPU training job at a time, deterministic seeds/settings. Capture actual fitting/tuning time; matched configuration counts do not imply equal compute. The source dataset, sample bitmaps, and checkpoint are existing resources, not newly acquired target labels. Original frozen source MSCN was trained for five epochs.
- Raw MSCN and fitting-only scalar calibration are references. Apply the same estimator gate to each selected candidate. Weight-updating models are comparators; their gate scores do not turn them into admissible actions under a frozen-model constraint.
- Gate profiles: raw P90/P95/P99 improvement 20/30/50%, 30/40/60%, 50/70/90%; corresponding scalar P95/P99 requirements 15/25%, 25/40%, 50/70%; median caps 1.25/1.20/1.10. Each outer trial has one independent gate; report pass counts across trials as repeatability, not a cross-trial policy or a recreation of the original inner 3-of-3 rule.
- Save all candidate tuning scores and the selected configuration before evaluating gate/test metrics. Save per-query predictions and model states. Report every family, including failures and fallback.

## B. Matched Table 1 / Table 8 Re-evaluation

- Recover original adaptation seeds 1/2/3, budgets 2,000 structural / 500 value, and the exact retained query indices from the saved k1450 DuckDB audit.
- Recompute the full residual predictions with the original helper. Re-run the original 20-epoch, 1e-3 CPU fine-tuning helper, including its original label-normalization semantics. These legacy reruns are separate from the new controls in A.
- Verify saved retained raw labels/counts, SQL row membership, and prediction agreement with the saved audit. Report any disagreement; do not substitute results while calling them exact reproduction.
- Use the same full-evaluation prediction arrays to summarize full held-out, cardinality-range-eligible, retained, range-eligible-but-not-retained, and out-of-range rows. Report per-seed quantiles, medians of per-seed summaries, and pooled quantiles separately.
- Within-range exclusions combine count mismatches, execution errors and the selection cap where original logs do not retain per-query exclusion reasons. Do not invent a per-query causal diagnosis.
- This is a paired population/configuration re-evaluation on existing benchmark labels, not a fresh COUNT audit or a new external dataset.

## C. Uncertainty and Gate Stability

- Use 2,000 paired bootstrap resamples over canonical SQL clusters to summarize pooled final-test P95/P99 ratios and reductions for the newly selected models. Keep occurrences across outer trials together. Also report each trial separately.
- Compare residual/direct pairs and residual/weight-updating controls on identical test SQL. These are descriptive conditional intervals, not multiplicity-adjusted superiority tests or full resampling of the model-selection process.
- On each independent gate partition, use 2,000 paired SQL resamples to report pass frequency for all three fixed profiles. These frequencies measure sensitivity to gate composition, not posterior safety probabilities.
- Three complete training/selection replications show observed between-split variation; they do not establish broad transfer or production safety.

## Outputs and Boundaries

All new files remain here. Preserve original manuscripts and results. Record file hashes, exact commands, dependency versions, seeds, split membership, selected configurations, raw predictions and observed costs. Label acquisition time remains unavailable for these pre-existing labels. The experiment does not implement Warper/ALECE/Robust-MSCN, measure concurrent DBMS throughput, or validate native policy transfer across datasets; those are separate experiments. The retraining control directly addresses the reviewer's MSCN alternative.
