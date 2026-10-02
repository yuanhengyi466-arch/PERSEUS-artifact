# Structural final-test eligibility correction before test execution

The initial canonicalizer sorts literal join clauses but does not identify
equivalent spanning trees of equality constraints. An additional read-only
audit, while validation timing was running and before policy selection or
any final-test execution, found some shared subqueries across gate/test and
validation/test under equality closure. Fit/tune/gate/validation remain
disjoint under this stronger check.

Freeze the following correction before inspecting any test outcome:

1. Canonicalize table aliases, predicate numeric literals, and the equivalence
   classes induced by equality joins. Preserve all filter constraints.
2. Exclude a prospective test parent if any connected subquery shares this
   identity with development, runtime validation, or an earlier retained
   test parent. Traverse the original fixed population order. Do not refill.
3. Preserve the original manifests, trained models, prediction maps,
   validation requests and gate thresholds. No existing timing or input is
   overwritten. Unused map entries for excluded parents are harmless.
4. Record excluded IDs, role intersections and the reduced test size in
   `test_eligibility_amendment.json`. Native final-test timing and post-test
   COUNT acquisition must use exactly the retained parents. The new
   `run_final_test.py` driver implements this filter; the original runner
   and its phase hashes remain unchanged.
5. Report both canonical record counts and equality-normalized counts.
   The one within-fit equivalent duplicate does not cross a role; the
   already trained model remains fixed, and its supervision count is
   disclosed rather than silently deduplicated/refitted.

This is a pre-test leakage correction based only on SQL structure, not on
test cardinalities, latency or prediction quality. The narrower final-test
population is mandatory, regardless of whether it improves reported results.
Validation continues to enforce the original archived-label correctness
criterion; observed archived-label discrepancies do not trigger a gate change.
