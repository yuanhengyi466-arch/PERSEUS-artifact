# Frozen-rule transfer under workload-mix and planner-cost changes

Fixed before new maps, COUNTs or timings on2026-09-29. The user requested
rrdL's cross-workload and second-configuration experiments. This is an
existing-STATS, Windows-native PG13.1 configuration-transfer experiment;
it does not constitute a second dataset, DBMS, or learned-estimator family.
The original strict deployment objective and all earlier outcomes remain.

## Hypotheses and source commitments

Primary question: can a source-selected policy retain useful, tail-controlled
admission on novel SQL with a changed workload mix and cost setting? Keep
the source1.10 accepted-ratio-P95 cap as primary; separately evaluate the
previously explored1.25 risk-tolerance arm. A success at1.25 cannot satisfy
the1.10 objective. Report null selection and adverse transfer as outcomes.

Freeze the residual HGB from uc4d_followup/native: seed811,1340 unique
development labels (866fit/237tune/237gate), its selected parameters, vocabulary,
clipping and source estimator gates. Source validation has30 parents per
column/table/value stratum at random_page_cost4. Reconstruct both source
rules exclusively from its saved validation records with the identical
336-rule grid and non-P95 limits. Check against the previous rrdL sensitivity
selection. Source1.10 is fallback; source1.25 is the table/value region with
positive root correction. These source results motivated the current design;
the new target cohort has not provided any timing or acquisition evidence.

## Target population

prepare_population.py and PREFLIGHT.md froze structural selection with seed912.
New target validation has10/20/30 column/table/value parents (60 total); target
test has20/40/60 (120 total). Role assignment and selection used only SQL
identity/structure. Exclude all parents in enumerated previous native traces,
all their represented connected subqueries, and successful/failed development
subqueries. New parents are mutually disjoint on equality-normalized connected
subqueries. Preserve each role across configurations. Existing benchmark
root labels are archived for discrepancy reports, not selection.

Primary target mixture is1:2:3, versus the source1:1:1. For a source-like
descriptive view, use the first20 test parents per drift in the already frozen
order (60 parents), under both configurations. This nested subset is not an
independent replication or a causal isolation of mixture from query identity.
Report strata individually and the full120-query mixture as primary.

## Database and maps

Configurations: pg4 with random_page_cost4; pg11 with random_page_cost1.1.
All other controllable settings stay fixed: same data/catalog, work_mem4MB,
JIT off, parallel gather0, force_custom_plan, no prepared statements,
join/from collapse limits8, statement timeout5s. SET is session-local; no
global setting, service, schema or data update. One connection and timing
worker. Record actual settings, table catalog estimates and relevant process
entrypoints; do not terminate other user processes.

Create native standalone connected-subquery row estimates separately under
each configuration and apply the same source model. Cache maps by config,
query_id and alias_set. Model weights and selection rules remain frozen;
configuration-specific PG input features/maps are regenerated without labels.
Compare native PG, native PG-map replay, and unconditional subplan repair.
Audit root/join row visibility and plan fingerprints before timed execution.

## Source-frozen and target-revalidated policies

At each configuration, target validation measures the three controls. Select
target_runtime110 and target_runtime125 using these new validation traces,
while retaining the source estimator gate/model. This comparator is runtime
revalidation, not target model retraining or a repeated estimator-label gate.

Both cap profiles hold all other limits fixed: at least20 validation parents;
all selected PG/candidate repetitions correct; median ratio<=.98; loss fraction
with ratio>1.01<=.25; Q95(T_policy)/Q95(T_PG)<=1.01; positive total-time saving.
Grid: nonempty drift subsets x min tables{2,3,4,5} x min PG rows
{0,100,1000,10000} x root correction direction{any,down,up}. Apply the fixed
source estimator gate to each region. Maximize saved validation request time,
then support, lower subset P95 and enumeration order. Empty selection uses PG.

Freeze source110/source125 and each configuration's target110/target125 before
any new target-test COUNT or timing. Test actually executes all four policies
plus PG, PG-map and repair. Do not replace source rules with target choices or
pick a preferred cap/configuration after seeing test results.

## Acquisition and timing

Before each role's timing, execute one PG COUNT per SQL/configuration with5s
timeout. Successful counts provide database truth; retain failures and old-label
discrepancies. Missing references are not filled from candidate outputs and
do not remove SQL. Compare successful references across configurations.

For each role warm up each SQL/config/method once, then execute three repeats.
Seeded order shuffles SQL, configuration order per SQL, and method order per
configuration. Session configuration precedes the timed method block; its cost
is separately recorded. Request time includes policy decision, map/GUC activation,
planning, execution and COUNT return. This represents an already configured
session with cached maps, excluding offline source inference and map preparation.
Time configuration switches separately; do not add them to only one policy.

Validation:60*2*3*3=1080 timed requests,360 warmups. Test:120*2*7*3=5040 timed
requests,1680 warmups. PG reference acquisition adds360 attempts across roles
and configurations. Worst-case timeouts remain explicit; timings of canceled
requests are censored, not completed-query latency. No parallel training during
DB timing. Save progress and observations without overwriting previous phases.

## Analysis and interpretation

Per-query median over the three repeats is the statistical unit. Report query
coverage, fraction of baseline time covered, accepted median/P95/P99/worst and
count failures, total request time, correct-pair sensitivity, plan changes,
Qp(T_policy/T_PG) and Qp(T_policy)/Qp(T_PG), and all stratum/config results.
Use2000 paired SQL-parent bootstrap draws conditional on source model and
selected rules. Paired configuration contrasts keep a SQL together. Do not
treat method repetitions or the nested60/120 views as independent samples.

Source-frozen transfer and target runtime revalidation are separate conditions.
Quantify configuration-specific PG-feature/map changes; feature calculability
does not prove runtime-risk invariance. Existing source labels/fitting/validation,
new target revalidation, map regeneration, and final evaluation are costed
separately. Source-only deployment does not consume target runtime-validation
labels/timings; the comparator does. Avoid a numeric amortization claim when
savings are uncertain or driven by censored executions.

This study can answer within-benchmark workload-mix and planner-cost transfer.
It cannot establish universal structural/value separation, a new estimator
family, map refresh/catalog invalidation, online model inference or concurrent
planner compilation. No public release or submitted-manuscript edit is implied.
