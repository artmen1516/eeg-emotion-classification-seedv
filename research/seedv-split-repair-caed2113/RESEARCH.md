# Research: seeded whole-session split repair

## Question

The prior audit (`research/evaluation-split-leakage-7e59d8a9`) found that the committed
notebook's evaluation split has no held-out test set and retained one minimal improvement: a
seeded, whole-session, pairwise-disjoint train/validation/test partition with a fail-fast
assertion. This run validates that record against the notebook (read-only) and, instead of
re-describing it, **applies** it as a stdlib-only, independently tested module.

## The parent defect (re-derived, notebook never executed)

`process_seedv_raw_files_with_mne_from_drive.ipynb` sha256
`35d98aa40078b0e6a60b65cf79e2f206827468bb95f1196deeff564cda5142fc`, 43 cells (33 code), all
`execution_count: null`. Cell 37:

```python
train_samples = math.floor((training_percent*total_array_samples))
trainList =  list(range(1, train_samples + 1, 1))
testList =  list(range(train_samples + 1, total_array_samples, 1))
...
train_set = windows_dataset
train_indices, val_indices = train_test_split(
    X_train.indices_, test_size=val_percent, shuffle=False)
```

At the recorded geometry (48 sessions x 12 trials from cell 32, `training_percent = 0.8`):

| partition | indices | size |
| --- | --- | --- |
| train | 0..459 | 460 |
| val | 460..575 | 116 |
| test | 461..575 | 115 |

`test & val` is **115/115**, i.e. every test trial is also a validation trial;
`val - test = {460}`. Since `460 % 12 == 4`, the session block 456..467 straddles
train/validation. Across `N = 0..1000` the parent satisfies disjointness + coverage on only
**6/1001** inputs (all with an empty test set); under the stronger non-empty contract it is
**0/1001**.

## Retained improvement (applied)

`seedv_split.py` implements

```python
split_indices(n_sessions, trials_per_session,
              train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
```

- Session `s` owns the contiguous block `[s*trials_per_session, (s+1)*trials_per_session)`.
- Ratios are validated as real, finite, strictly inside `(0, 1)` and summing to 1 within `1e-9`
  using `math.fsum` (so `0.7 + 0.2 + 0.1 = 0.9999999999999999` is accepted, while NaN is not).
- Whole-session quotas use floor + largest remainder with a deterministic tie-break
  (train, val, test), followed by a min-one-session-per-partition repair.
- Session ids are shuffled with a local `random.Random(seed)`; global RNG state is untouched.
- Outputs are ascending lists. An internal guard raises `RuntimeError` (not `assert`, which
  `python -O` strips) if any index duplicates, goes missing, or a session straddles partitions.
- `validate_partition(train, val, test, n_sessions, trials_per_session)` exposes the same check
  publicly with `ValueError`. Because the signature takes only counts, sessions cannot overlap at
  call time; this validator is how "raising on overlapping sessions" becomes observable and
  testable.

## Measurements

All numbers are pure Python on this machine; the notebook was parsed, never run.

| Quantity | Parent | Candidate |
| --- | --- | --- |
| Split at 48x12, seed 42 | 460 / 116 / 115 | 348 / 120 / 108 |
| Whole sessions | n/a (block 456..467 straddles) | 29 / 10 / 9 |
| Test-in-val overlap | 115/115 (100%) | 0 |
| Contract pass, non-empty | 0/1001 (N=0..1000) | 160/160 (grid) |
| Contract pass, disjoint+coverage only | 6/1001 | 160/160 |
| Time at 48x12, best of 5 | 0.0042 ms | 0.0758 ms |
| Time at n=100000, best of 5 | 0.8738 ms | 10.5176 ms |
| Peak traced memory at 48x12 | n/a | 42.9 KiB |
| Repeat-call output | n/a | identical |
| Cross-process output (PYTHONHASHSEED 0/1/12345) | n/a | identical digest `1b8bc0aa…` |
| Seed 7 vs seed 42 | n/a | different membership |

The time ratio is machine-specific and is not offered as correctness evidence. At the goal's
input size the candidate is under 0.1 ms, so the repair is free at this scale.

## Tests

- `tests/test_seedv_split.py` (36 tests): default 348/120/108 and 29/10/9; disjointness; exact
  coverage; whole-session integrity; a grid over n_sessions/trials; non-emptiness; one-session
  ratio tolerance; reproducibility; global-RNG invariance; seed sensitivity at 48 sessions;
  every documented `ValueError` path (including NaN, negative, zero, bool/float counts,
  `n_sessions < 3`); and `validate_partition` rejections.
- `test_repro_split_disjoint.py` (4 tests, repository root): reconstructs the cell-37 parent and
  asserts the historical failure (460/116/115, 115/115 overlap, `{460}`, 456..467 straddle,
  validator rejects it); asserts the replacement is 348/120/108, pairwise disjoint, covers
  0..575 exactly once, keeps whole sessions intact, is not the leaky subset relation, and is
  seed-reproducible.

Both `python -m pytest -q` and `pytest -q` report **40 passed**. No `conftest.py` is needed; each
test file bootstraps the repository root on `sys.path`.

## Scope and honest limitations

- The notebook and README are **not modified**: the repository's shipped pipeline remains leaky
  until cell 37 is changed to use this helper. This change is the tested, extractable kernel of
  the retained repair.
- Session-level disjointness is not subject-level disjointness. Under SEED-V's 16 subjects x
  3 sessions layout (external fact), a 29/10/9 session split can still share subjects across
  train and test.
- The rectangular `(n_sessions, trials_per_session)` model cannot represent the 7–15
  trials/session layout implied by cell 24; it mirrors the 48 x 12 layout recorded in cell 32.
- The default 0.6/0.2/0.2 deliberately differs from the notebook's `training_percent = 0.8`
  (348 vs 460 training trials at 48x12). The ratio is a parameter and the difference is
  documented, not silent.
- No accuracy was measured or reproduced; the tests are structural only.

## Files

Added: `seedv_split.py`, `tests/test_seedv_split.py`, `test_repro_split_disjoint.py`.
Research record: `research/seedv-split-repair-caed2113/` with `research.json`, `review.json`,
`report.json`, `charter.md`, `summary.md`, `round-1.md`, and the moved scout inputs.
