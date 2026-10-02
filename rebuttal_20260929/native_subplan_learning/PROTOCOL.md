# Native PG subplan learning, prospective version 1

Fixed on 2026-09-29 before new model fits and native timing. This is a new
PG-anchored adapter experiment, not a rerun of the submitted MSCN adapter.
The original submission and prior diagnostic remain immutable.

## Population and supervision

Use the 180 development parents and 1,354 unique COUNT attempts acquired in
`native_subplan_diagnostic`. Failed counts cannot supervise models. Zero counts
are retained and floored at one only for log targets/q-error/injection.
Form connected components of parents sharing any canonical connected subquery
(including the root). Sort components by decreasing number of
unique subqueries, with seed-601 hashed identity as tie-break. Allocate each
component to the least-filled role relative to pooled target fractions
fit/tune/gate = 0.65/0.175/0.175. Components can cross drift categories and
must remain atomic. A shared subquery may appear in two drift summaries but
is supervised and budgeted once. No refitting on tune or gate labels.
Shared labels are charged once. These are already studied development data;
the gate is internally held out for this new fit, not new external evidence.

From the three official target files, exclude every parent represented in
available `paper/**/native_callback_detail.csv` files or prior attribution
cases, and every parent sharing a connected subquery with development.
Sort remaining canonical parents by seed-601 hash. Greedily select 120 per
drift with no shared connected subquery across selected parents or drifts.
Assign alternating parents to runtime validation and final test (60 each
per drift; 180 per role). Selection uses SQL structure only. This is new
native testing relative to the enumerated manifests on the existing STATS
benchmark, not a new dataset or a claim that every SQL was never used in
any past estimator experiment. Save exclusions, manifests and input hashes.

## Learning and controls

One pooled model per family, using native PG standalone connected-subquery
row estimates as the baseline. Fit `log(max(actual,1)/max(pg_rows,1))`.
Features reuse the existing full signature implementation; vocabulary,
residual clipping (fit percentiles 1/99) and scalar offset (median fit
residual) use fit labels only. HGB: 300 iterations, learning rate .05,
L2=.01, early_stopping=False, leaves {15,31}, minimum leaf size {5,20}.
RF: 250 trees, depth {14,None}, minimum leaf size {2,5}, n_jobs=1.
Seed=601. Choose each family's configuration by the arithmetic mean of
log P95 and log P99 q-error on unique tune subqueries; stable config order
breaks ties. No model seed search.

Native controls: PG unchanged; standalone PG-map replay; HGB root-ratio
broadcast (PG rows times the model's root PG-relative correction, including
root); clipped root-ratio broadcast; individual HGB subplan repair; individual
RF subplan repair. Root broadcast uses the same fitted HGB and label budget.
It is distinct from the earlier MSCN/PG transfer baseline. Choose symmetric
ratio cap from {2,4,10} using the same tune objective, counting each canonical
tune subquery once and averaging log predictions if it has several parents.
Subplan targets match the injection interface. Base relation estimates remain
native. PG-map is a substantive control, not assumed to be an identity.

## Two admission decisions

Estimator gate: per drift, at least 30 successful unique gate subqueries;
P90/P95/P99 reductions >=30/40/60 percent against PG, P95/P99 reductions
>=25/40 percent against the fit-only scalar offset, median ratio <=1.20
against PG. Use the fixed original numerical profile with explicitly changed
PG/subplan denominators. One grouped gate here, not the original 3-of-3 test.

Measure all six unconditional controls on runtime validation. For HGB and RF,
select runtime-only and full policies. Full requires that the drift passed
the estimator gate. Rules use nonempty drift subsets (7), minimum table
counts {2,3,4,5}, minimum native root estimates {0,100,1000,10000}, and root
correction direction {any,down,up}: 336 rules per family/policy before pruning.
Each region must have >=20 validation parents, all three baseline/candidate
executions correct, median per-query request ratio <=.98, P95 <=1.10,
fraction with ratio >1.01 <=.25, whole-policy P95 latency percentile ratio
<=1.01, and whole-policy sum of per-query medians below PG. Select maximum
validation total-time savings, then larger support, then lower subset P95,
then enumeration order. If no rule qualifies, select PG fallback. Freeze
rules before any final test timing/COUNT audit; no test-driven relaxation.

## Timing and cost boundary

PG13.1 native joinrel-row callback, same schema/catalog and existing helper
settings, 5-second statement timeout, JIT/parallel query off, automatic
prepared statements off. One connection, one process, no concurrent timing
workers. For each role, warm up once per SQL/condition, then three repetitions
with seeded randomized SQL and condition order. Validation has six controls;
test adds four actually executed policies (runtime-only/full x HGB/RF).
Fallback performs its actual selection, callback-off activation and SELECT.
Time Python policy lookup, GUC activation, planning and execution to returned
COUNT. Validate counts against the archived parent labels. Include timeouts,
errors and fallback costs; failed requests are not discarded or assigned
zero time. Report both statuses and measured request time.

Primary deployment mode caches native features and predicted estimate maps;
model inference, standalone EXPLAIN and map preparation are offline and
separately charged. There is no redundant candidate EXPLAIN in the request
because selection uses cached structural features, not candidate-plan features.
This measures the implemented warm-cache request path, not online inference,
COUNT acquisition, throughput under concurrency or map refresh. Do not use
these measurements to claim those unmeasured properties. Report setup and
validation cost and break-even repetitions only if a positive saving exists.

## Audit and interpretation

Save per-request timing, map hashes, model/tune/gate decisions, root-map checks
and chosen-plan join-node checks. Fresh standalone PG estimates may differ
from a join's estimate inside a parent, so report PG-map separately. After
freezing and evaluating policies, acquire final-test connected-subquery truth
for error analysis; unsuccessful COUNT attempts remain explicit. No further
model or policy fitting after test. Report errors by subplan size, coverage,
accepted subset median/P95/max, whole-workload sum and latency P50/P95/P99,
percentiles of ratios separately from ratios of percentiles. Use 2,000 paired
unique-parent bootstrap draws with fixed seed 601 for conditional intervals.
This single split does not estimate policy-selection uncertainty or transfer
to another workload. Nonzero admission and positive gains are not guaranteed.

Implementation note before any fit/new timing: the initial preparation stopped
at a cross-drift component assertion. No data acquisition or model fit ran.
The aborted start record is preserved as `prepare_attempt1_failed.json`;
the corrected pooled component allocation above is the executed protocol.
