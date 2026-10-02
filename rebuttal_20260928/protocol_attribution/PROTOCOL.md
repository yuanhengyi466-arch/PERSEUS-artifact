# Protocol Attribution Experiment: ovz3

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-28
- Verification Status: UNVERIFIED until execution and analysis complete
- Version Label: protocol_attribution_v1

## Question and Scope

Does estimator-side admission add measurable benefit or reduce required runtime validation work compared with ordinary residual repair and runtime-only admission, holding the candidate repair and SQL population fixed?

This is a new experiment on the existing MSCN/STATS benchmark and Windows PG13.1 callback. It is not an untouched external benchmark or a new estimator family. Earlier traces informed the motivating question. Freeze this document and the machine-readable configuration before observing the new timing outcomes. This is a local prospective analysis plan, not a public preregistration.

## Fixed Design

- Source MSCN predictions/checkpoint remain frozen. All predictors use the same cached source predictions.
- Two residual regressors: existing HGB settings (300 iterations, learning rate .05, 31 leaves, L2 .01) and RF (250 trees, depth 14, leaf minimum 3). No tuning on new validation/test timings.
- Outer split seeds: 401, 402, 403. A canonical SQL identity hashes to calibration/runtime-validation/test with proportions .60/.20/.20. Identity ignores input table/condition ordering. All copies, including across drift files, receive the same role for each outer seed.
- Calibration pools: 3,000 unique SQL per structural drift and 1,000 for value drift. Within the pool, three inner partitions (seeds 1,2,3) use fit/gate sizes 2,000/1,000 and 500/500. Require all three gate evaluations to satisfy Eq. (5). Deploy the first replica fixed in advance; never select the replica on validation performance.
- All labels in a calibration pool are charged once as unique calibration supervision, with fitting/gating uses recorded separately. Different outer runs are resampling replications, not independent external datasets.
- Native eligible population: unique SQL identities already represented in the original native callback input. This is an explicitly selected existing workload subset. No selection by new model gains, latency, or plan-change outcome.
- At most 60 SQL per drift per outer runtime-validation role, and 60 per test role, chosen by a fixed hash. Keep all available if fewer. All included cases have at least three tables, following the original native subset.
- Use the original connected-subplan construction, held fixed across ablations: subplans use native PG rows times the predicted full-query residual ratio; the root uses the repaired MSCN estimate. Include raw-MSCN root injection as a reference. This experiment does not validate the transfer assumption itself.
- Callback maps are prepared offline. Warm-cache timings measure incremental online residual feature extraction/inference, connection-session configuration, candidate planning when required, admission checks, SQL planning/execution, and result receipt. Cached source-MSCN inference and map construction/refresh are excluded and reported explicitly. No claim of a complete production service.
- Main policies: PG native fallback/reference; repair-all; estimator-gate-only; runtime-gate-only; full two-stage PERSEUS. Run both repair models. Raw-MSCN injection is an extra reference, distinct from PG native fallback.
- Runtime rules: original drift subsets and cardinality/structural caps, restricting the injection method to connected-subplan repair so that 2,940 rules are compared per model. Candidate-plan maximum rows is measured using an actual EXPLAIN, whose time is charged. Rule class fixed to minimum 20 validation SQL, accepted median ratio <= .98, accepted P95 <= 1.10, fraction with ratio > 1.01 <= .25, and whole-policy P95 <= 1.01. Maximize validation accepted count, tie-break by lower median/P95. No eligible rule means global fallback.
- Runtime-only searches the full rule family. Full PERSEUS restricts admission to drifts whose estimator gate passes. Both use the same validation observations and the same rule constraints. Cost accounting separately identifies the validation executions the full protocol could omit; shared experimental measurements are not claimed as actual collection savings.
- Outside a rule's drift set, skip inference/planning and use fallback. Inside its drift set, charge inference/candidate planning even when the rule rejects the case based on candidate features.
- Validation and test stages run separately. Save selected rules and hashes before starting test execution. Test measurements must not change the rule.
- Per-query warmup, three measured repetitions, deterministic randomized query and method order, one SQL execution at a time. Other host activity is recorded; this is not a dedicated idle machine. PostgreSQL session parallelism and JIT are disabled, as in the source experiment.
- Statement timeout: 5 seconds. Keep errors/timeouts in the result and report them; timeout wall times are censored observations and must not be called completed-query latency. No successful-case-only policy reporting.

## Outcomes

Primary: sum of measured client request times over all test SQL, relative to PG native, including fallback and online admission costs. Secondary: coverage, unique SQL, ratio median/P95, latency P50/P95/P99, maximum slowdown with absolute times, correct-result counts, candidate plan changes, and offline labeling/validation work. Native total latency does not include cached MSCN generation, fitting, or offline map preparation; list these separately.

Use per-SQL medians across the three timing repeats. Report each outer split. For pooled intervals, bootstrap SQL identities (all appearances across splits together) with 2,000 resamples. These are conditional uncertainty intervals for the frozen selected policies; between-split variation is separately shown, not claimed to be included in a population-level selection guarantee. With zero accepted SQL, accepted-subset metrics are undefined, not perfect scores.

Report every ablation, including no admission or regressions. Identical selected policies do not demonstrate an incremental runtime benefit for the extra estimator gate. Relative benefits must be accompanied by absolute milliseconds and excluded costs.

## Execution

From `.`, run `python rebuttal_20260928/protocol_attribution/run_protocol_attribution.py prepare`, then `validate`, `select`, `test`, and `analyze`. All outputs remain in this isolated directory. The submitted manuscript and original experimental result files remain unchanged.
