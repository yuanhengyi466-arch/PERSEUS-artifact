# Why a correct root update can damage an intermediate subplan

This is an elementary error identity, not a new statistical safety theorem.
It explains the target/interface mismatch tested by this supplement.

Let a connected subplan have native estimate `p >= 1` and true cardinality
`c >= 1`, after the same floor used by q-error. Define its needed log correction
as `e = log(c/p)`. A broadcast root multiplier, with injection flooring, applies
an effective log correction `a = log(max(1, p * root_ratio) / p)`.

Then

```text
log q_error(native)    = |e|
log q_error(corrected) = |e - a|
```

If `e * a < 0`, the correction moves in the wrong direction. Consequently,

```text
log q_error(corrected) - log q_error(native) = |a|
q_error(corrected) / q_error(native)         = exp(|a|)
```

Symmetric ratio clipping reduces the magnitude of a harmful correction but
preserves its direction unless clipping or flooring makes the update zero.
It therefore cannot generally substitute for learning the subplan's own error.
This says nothing by itself about plan runtime: cardinality error and latency
are different outcomes, so native validation remains necessary.

## Limitation of any common nontrivial multiplier

Suppose two intermediate subplans under the same parent have opposite native
errors, `c1 > p1 > 1` and `p2 > c2 >= 1`. No common multiplier `k != 1` can
weakly improve both q-errors, including the injection floor:

- If `k < 1`, then `max(1, k*p1) < p1 < c1`, so the first q-error increases.
- If `k > 1`, then `k*p2 > p2 > c2`, so the second q-error increases.

Thus one shared multiplier has no nonzero update that is pointwise
non-degrading on both subplans. The strict `p1 > 1` assumption matters:
flooring can otherwise make a downward update on the first subplan inert.
This elementary incompatibility motivates subplan-specific targets. It is
not a bound on tail risk, a runtime theorem, or a claim that joint q-error
improvement is necessary for faster execution.

## Existing diagnostic check

In `native_subplan_diagnostic/subplans.csv`, 441 of the 1,201 successfully
labeled intermediate parent/subplan records have a strictly opposite
*effective* update. The identity holds numerically on all 441, with maximum
absolute log-space discrepancy `4.44e-16`. This differs from the earlier
40.4% statistic, which uses the unfloored full-query multiplier; flooring
can make some applied updates zero. These are development records and may
share canonical subqueries. The new held-out native experiment is separate.

Among the 171 label-complete development parents, **90** have opposite
intermediate error directions with both native estimates above one:
30 column, 58 table, and 2 value parents. These counts illustrate the
incompatibility condition on this development sample. They do not estimate
its frequency on all STATS SQL.

## Relation to the reviewer questions

The new controls hold the trained HGB and its supervision fixed while changing
how estimates are assigned: broadcast root ratio, clipped root ratio, or
subplan-specific residual. RF tests whether the conclusion depends on boosting.
The identity motivates this comparison; only the measured held-out results
can establish its practical value or any admission coverage.
