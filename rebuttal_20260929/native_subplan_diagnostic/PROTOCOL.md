# Native Connected-Join Cardinality Diagnostic

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Date: 2026-09-29
- Status: prospective local diagnostic specification; see completion records
- Version: native_subplan_diagnostic_v1

## Question and Population

Does correcting connected join cardinalities improve plans and measured requests on the development population, and how much does the existing candidate-planning path cost?

Use all 180 cases from outer 401's runtime-validation role in the completed protocol-attribution experiment: 60 each of column/table/value drift, without outcome filtering. These queries and earlier validation timings have already been studied. This is a development diagnostic, not a fresh held-out deployment result. Cases have 3--6 tables. Preserve all query identities and record any label discrepancy or failed acquisition.

## Four Fixed Conditions

1. `pg`: native PostgreSQL, callback disabled.
2. `pg_map`: freshly obtained standalone PG estimates for every connected intermediate join, plus the native full-query root estimate, injected through the callback. This controls for using standalone estimates in the parent query; it is not assumed to preserve the original plan.
3. `hgb_ratio`: the frozen HGB estimate map from the prior experiment, including its root estimate and root-ratio scaling of connected intermediate joins. Retain value drift as an unconditional diagnostic, without implying estimator-gate admission.
4. `oracle_join`: freshly executed COUNT labels for every connected intermediate join and the root, injected with the callback's floor at 1. Base-relation estimates remain native; this is a connected-join oracle, not an oracle for every planner quantity. Cardinality accuracy does not guarantee optimal plans. This condition is diagnostic and uses labels unavailable to a deployable method.

Collect EXPLAIN estimates and COUNT labels with the callback disabled. Deduplicate by canonical subquery identity within this acquisition. Record every actual execution and its elapsed time; use a 5-second timeout, without outcome-driven retries. Preserve zero counts before the callback floor. If any required label is unavailable, `oracle_join` falls back to native PG for the entire parent query and records incomplete coverage. Complete-only analyses are secondary and explicitly selected. Root counts are checked against the archived query truth.

## Measurement

- Existing Windows PostgreSQL 13.1, same schema/session configuration as prior native experiment; parallel query and JIT disabled. No server/data/statistics changes. Other user workloads are not stopped.
- Freeze the population/protocol before COUNT acquisition. Record input/code/server hashes and versions. Do not change the original submission or completed result files.
- Confirm injected roots using untimed EXPLAIN and inspect chosen-join rows separately. Parameterized path rows are not assumed identical to joinrel cardinalities.
- Two timing paths, each with three independently shuffled repetitions over all 180 cases and four conditions: 4,320 timed SELECT requests in total. Randomize case and condition order deterministically within each repetition; alternate path order. Execute serially to avoid self-contention. No SQL PREPARE or automatic prepared statements.
- `mapped`: timer includes map/session activation and SELECT completion; predictions/maps are cached. This isolates the effect of estimates under the existing callback path, excluding label acquisition and inference.
- `candidate_path`: PG uses activation plus SELECT. For other conditions include activation, candidate EXPLAIN, and SELECT; HGB additionally recomputes and verifies the frozen residual prediction. Report inference, activation, candidate-planning, SELECT round trip, and total request time separately. This is unconditional candidate evaluation, not a learned admission policy.
- Run one untimed mapped warmup for every case/condition; preserve its outcomes. Record every timed status and returned count. Connection establishment, map construction, source-MSCN inference, and label acquisition are outside request time and reported separately. Incomplete oracle cases use the PG path explicitly.
- After timed requests, run one EXPLAIN ANALYZE per case/condition to separate server planning/execution and inspect physical plans. Its overhead and results are separate from direct SELECT request latency. Keep profiler failures; never fill missing server time with zero.

## Analysis and Decision

Report all-query totals and latency quantiles, medians of per-query repeated times, per-query ratio quantiles, plan changes, failures, oracle coverage, and results by drift and join size. Distinguish the percentile of paired ratios from the ratio of latency percentiles. Use 2,000 paired query bootstrap resamples for conditional total-request-time ratios; these are development-set diagnostics, not selection guarantees.

Compare actual subplan cardinality to native and old HGB-ratio estimates; count cases where a full-query correction has the opposite direction to the needed intermediate correction. Use both PG-root-relative and MSCN-root-relative directions explicitly. A mismatch identifies transfer disagreement, not by itself the causal source of latency.

No deployable success or minimum coverage is required. If complete oracle cases show no convincing net request benefit, first inspect cost-model/path behavior and overhead rather than expand residual training. If they show useful gains, a subsequent subplan-specific learning experiment needs its own supervision budget, frozen protocol, and genuinely separate evaluation. Do not tune or declare a deployment policy from this oracle diagnostic.
