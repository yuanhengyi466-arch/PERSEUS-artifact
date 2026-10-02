# Supplemental Native Policy Throughput Replay

Specified on 2026-09-29 before the new replay. This is separate from estimator training and uses the frozen policy and SQL population of the 2026-09-28 protocol-attribution experiment.

- Population: all 540 previously specified native test split-query cases (494 distinct canonical SQL), without outcome filtering.
- Policies: PostgreSQL default and the frozen full-HGB policy. Verify its selected rule is null for every outer split. The experiment measures the current all-fallback operating point; it does not introduce admitted repairs or demonstrate throughput under nonzero coverage.
- Workload: closed-loop replay with 1 or 4 clients, three paired repetitions. Within each pair both policies receive the same deterministic shuffled case list. Alternate policy order and concurrency order as encoded in the runner before measuring outcomes.
- Each client has a separate connection. Session settings match the earlier experiment, including disabled PostgreSQL parallel query and JIT and a 5-second statement timeout. Connections are established outside measurement. Run one sequential default warmup over the complete population before the timed blocks.
- Record every submitted request, status, count, client request time and wall-clock offset. No retry and no successful-case-only filtering of the main latency summaries. Report successful completions per measured block second as goodput, attempted requests per second separately, and timeouts/errors explicitly.
- A worker issues its next request after the current one completes. The case list is a replay backlog, not a simultaneous-arrival queue; latency is measured from request processing to result receipt, not from experiment start.
- Goodput denominator is the actual complete-block makespan with every worker included. Pair corresponding policy blocks. Report all three repetition ratios and their range; do not infer confidence from treating concurrent requests as independent throughput replications.
- Source predictions/maps remain offline/cached. This all-fallback policy skips repair inference by design. Do not label these measurements production-service throughput or a nonzero-admission speedup.
- Run after this task's training and statistical analysis have finished. Record host CPU/memory activity. Other user processes are not stopped.

Expected measured requests: 540 cases x 2 policies x 2 concurrency settings x 3 repetitions = 6,480, plus 540 separately recorded warmup requests. Freeze input and protocol hashes. Preserve raw request timings and original experiment files.
