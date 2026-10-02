# UC4d matched-cost and native attribution supplement

Fixed before the new fits and timing on 2026-09-29. The unchanged submission
and previous measurements remain inputs. This is a focused response to UC.2
and UC.5, not a new broad benchmark or a search for a favorable test subset.

## A. Total target-supervision budget

Frozen seed-1 MSCN predictions and the three official STATS target files.
Use equality-normalized SQL identities, deduplicate within each target, and
check source/target overlap. Identity hashes assign fit/tune/gate/test with
probabilities .50/.15/.15/.20 for seeds 711, 712, 713. The hash ignores drift,
so copies across drifts have the same role within each trial. Rank within
each role by a separate fixed identity hash. Larger budgets extend prefixes;
test membership is fixed across budgets and all compared methods.

Total budgets: 500/1000/2000/3000/4000 for column and table; 500/1000 for
value. Allocate .60/.20/.20 to fit/tune/gate. Count unique SQL labels, including
tuning and admission labels; do not multiply the budget by model families.
Report reuse across trials and drifts. Final-test labels are evaluation only.
Historical benchmark acquisition time is unknown. Native acquisition and
validation costs below are separate measurements, not estimates of benchmark
acquisition. This experiment evaluates the estimator admission stage; its
x-axis is not the complete cost of deploying an optimizer policy.

Compare raw MSCN, fit-only scalar correction, residual/direct HGB, and
residual/direct RF. Reuse reviewer_4udk's four configurations per family,
fit-only vocabulary, 1/99 fitted-target clipping, and tuning objective:
geometric mean of P95/P99 q-error, then median and configuration ID. Preserve
the source prediction-scale feature for direct and residual models. No refit
on tuning/gate. HGB 150/300 iterations x .05/.10 learning rate, 31 leaves,
L2=.01; RF 250 trees, depth 10/14 x minimum leaf 3/5. At most two CPU threads.

Fixed gate: raw P90/P95/P99 improvements >=30/40/60%, scalar P95/P99
improvements >=25/40%, median <=1.20 raw. One independent gate per outer
trial; a rejected learned candidate falls back to raw MSCN. Scalar is a
reference, not subject to an impossible improvement-over-itself condition.
Save candidate and routed errors, pass counts, selected configurations,
fitting/selection times, and every split. Plot total-label P95/P99 curves.
Use 2,000 paired SQL-cluster resamples with all repeated test appearances
together for conditional intervals and direct/residual comparisons. These
are not selection-adjusted or multiple-comparison-adjusted guarantees.

## B. Direct versus residual complete native pipeline

Use the PG-relative connected-subplan adapter, explicitly distinct from the
original MSCN-root-ratio construction. Deduplicate equality-normalized
successful development labels from native_subplan_learning, preserving its
fit/tune/gate roles. All four model families share these exact labels and
features. Four configurations per family: HGB leaves15/31 x minleaf5/20,
300 iterations, lr.05, L2=.01, early_stopping=False; RF depth14/None x
minleaf2/5, 250 trees. Seed=811, one CPU thread. Residual predicts log(actual/PG),
direct predicts log(actual), both fit-clipped at percentiles1/99. Select on
unique tuning labels only, using mean log P95/P99 and fixed config tie order.
The same numerical estimator gate as A uses PG and fit-only scalar-PG,
with >=30 gate subqueries per drift. No test-based tuning or threshold change.

From the previous structurally audited validation/test cohorts select the
first 30 parents per drift per role by seed811 hash of SQL identity. This
gives 90 validation and 90 test SQL. Preserve the old roles and all
cross-role connected-subquery exclusions. This is a matched attribution
replay on previously evaluated held-out SQL, not a fresh external test or
independent confirmation of a newly discovered deployment benefit. Selection
uses identity only, without latency, error, label-match or old gate outcomes.

Measure PG, PG-map replay, and all four unconditional models on validation.
For each model select runtime-only and full policies with the same 336-rule
grid: nonempty drift subsets; min tables2/3/4/5; min PG root rows
0/100/1000/10000; root correction direction any/down/up. Full requires
estimator-gate passage. Minimum20 parents; all accepted baseline/candidate
repetitions correct; ratio median<=.98, ratio P95<=1.10, fraction ratio>1.01
<=.25, whole-population Q95(policy time)/Q95(PG time)<=1.01, positive saved
total time. Maximize saved validation time, then support, lower subset P95,
and rule order. Otherwise PG fallback. Freeze all eight policies before
test. Test executes six controls plus eight actual policies, not simulated
policy times. Retain both runtime-only and full decisions for attribution.

Before each role's paired measurement, acquire one PG COUNT per SQL with the
same 5s timeout. Use successful counts as current database correctness truth;
store archived-label discrepancies separately. Failed reference counts stay
in the population and prohibit their use as validated successful admissions.
Root labels never fit the models. This corrects the earlier conflation of
archived benchmark discrepancy with query-result corruption. Do not discard
queries or infer truth from a preferred model. Returned cross-method count
conflicts invalidate correctness evidence and remain explicit.

One database connection and timing worker, existing Windows patched PG13.1
on port5434, same stats_inj schema, JIT/parallelism off, no prepared statements,
5s timeout. One warmup per SQL/method and three seeded randomized paired
repetitions. Include policy lookup, callback activation, planning and COUNT
execution in request wall time. Fresh standalone EXPLAIN and four-model map
inference are separately charged offline. Cached-map mode is not online
model inference, concurrent throughput or production map refresh.

Audit actual root/join rows before timing. Preserve all request outcomes,
timeout-censored wall times, method order, settings and host-process snapshots.
No concurrent training from this task during DB timing. Other user processes
are recorded without terminating them. SQL-parent paired bootstrap (2,000)
reports total-time ratio, accepted coverage and subset median/P95, worst
slowdowns, percentile-of-ratios and ratio-of-percentiles separately. Timeout
observations are not completed-query speedups. Compare residual/direct pairs
under both unconditional and full/runtime-only admission. A zero accepted
subset has undefined subset metrics, not perfect safety or acceleration.

## Completion

Store immutable input hashes, scripts, protocols, fit/gate/policy manifests,
per-query results, complete cost ledgers and analysis commands. Do not overwrite
partial or complete observations. Record deviations before dependent stages.
Scientific targets remain matched-cost advantage and useful native admission;
unfavorable or inconclusive outcomes do not satisfy those targets. No public
artifact release, submitted-manuscript edit or new generalization claim follows
automatically from these local experiments.
