# Summary

**Retained improvement applied: a seeded, whole-session, pairwise-disjoint train/val/test split.**

The prior audit found that cell 37 of the committed notebook has no held-out test set: at the
recorded 48 sessions x 12 trials = 576 trials, `test = 461..575` is a 115/115 subset of
`val = 460..575`, and the session block 456..467 straddles train/validation because
`460 % 12 == 4`. This run validated that record read-only (notebook parsed as JSON, never
executed) and shipped the retained repair as `seedv_split.py`, with pytest coverage and a
root-level regression test. The notebook and README were not modified.

At the goal's input size (48x12, seed 42) the new split is **348/120/108** over **29/10/9** whole
sessions: three non-empty, pairwise-disjoint sets covering `range(576)` exactly once, with no
session straddling a partition. That reproduces the audit's retained evidence exactly.

## Verification (pure Python; nothing executed from the notebook)

- Parent contract over `N=0..1000`: 6/1001 under disjointness+coverage only, **0/1001** once
  non-emptiness is required; test-in-val overlap 115/115.
- Candidate: 160/160 grid inputs pass the non-vacuous contract (n_sessions 3..59 at 1 and 2
  trials/session; 3..48 at 12); every partition non-empty.
- Determinism: repeated calls identical; output digest `1b8bc0aa…` under PYTHONHASHSEED
  0/1/12345; global RNG untouched; seed 7 changes membership.
- Cost at 48x12, best of 5: 0.0758 ms (parent 0.0042 ms), 42.9 KiB peak traced memory. At
  n=100000: 10.5176 ms vs 0.8738 ms. Timings are machine-specific and not correctness evidence.
- Tests: `python -m pytest -q` -> 40 passed; `pytest -q` -> 40 passed; plain-file runs 36 + 4.
- Notebook sha256 unchanged (`35d98aa4…a5142fc`); no tracked changes outside `research/` except
  the three added application files.

## Corrections carried forward

- The audit's "1001/1001" contract was vacuous on empty partitions; this implementation requires
  non-empty partitions and records the parent as 0/1001 under that stronger contract.
- NaN/Inf, negative, zero, `1.0`, bool and float inputs raise `ValueError`; decimal ratio triples
  summing to `0.9999999999999999` are accepted via `math.fsum` with an absolute tolerance.
- "Raising on overlapping sessions" is implemented as an internal `RuntimeError` guard plus a
  public `validate_partition` raising `ValueError`, since the required signature takes counts only.
- The 0.6/0.2/0.2 default is documented as differing from the notebook's `training_percent = 0.8`.
- The shipped repository still contains the leaky cell 37; this is an additive, tested extraction,
  not a notebook repair. Session-level disjointness is not subject-level disjointness.

## What is not claimed

No accuracy was measured or reproduced, no subject-level ordering is asserted, and no notebook
cell was executed.
