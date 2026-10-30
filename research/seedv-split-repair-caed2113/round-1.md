# Round 1

## Inputs

- `scout-explore.json`, `scout-synthesize.json`, `scout-challenge.json` (moved into this directory
  after the directory name was computed).
- `process_seedv_raw_files_with_mne_from_drive.ipynb`, parsed as JSON; `README.md`; git history.
- `research/evaluation-split-leakage-7e59d8a9/` (the parent audit record).
- No notebook execution, no network, no Drive, no SEED-V data.

## What was done

1. Read all three scouts and the parent audit record. Selected the single retained improvement:
   the seeded, whole-session, pairwise-disjoint split with a fail-fast assertion.
2. Re-derived the parent defect from committed literals only:
   - notebook sha256 `35d98aa40078b0e6a60b65cf79e2f206827468bb95f1196deeff564cda5142fc`,
     43 cells / 33 code, all `execution_count: null`;
   - cell 37: `train_samples = floor(0.8*576) = 460`, `testList = range(461, 576)`,
     `train_set = windows_dataset`, `train_test_split(..., test_size=0.2, shuffle=False)`;
   - therefore train 0..459, val 460..575 (116), test 461..575 (115), overlap 115/115,
     `val - test = {460}`, and `460 % 12 = 4` so block 456..467 straddles.
3. Applied the repair additively:
   - `seedv_split.py`: `split_indices(...)` with largest-remainder whole-session quotas, min-one
     repair, local seeded shuffle, `RuntimeError` fail-fast guard, public `validate_partition`
     (`ValueError`);
   - `tests/test_seedv_split.py`: 36 pytest cases;
   - `test_repro_split_disjoint.py` at the repository root: 4 regression cases including the
     parent reconstruction and a mutation guard.
4. Measured parent vs candidate at the goal input size (48x12 = 576) and on the correctness
   range, then created this directory (`research/seedv-split-repair-caed2113`) and wrote the
   record. The notebook and README were untouched.

The pre-existing `research/pointer.json` named the parent audit directory. That directory is this
run's *input*, so it was left intact and the pointer now names this run's record; the naming rule
(`cat research/scout-*.json | shasum -a 256`, hash8 `caed2113`) was computed from the flat scouts
before the move and still reproduces from this directory's scout files.

## Measurements

| Quantity | Value |
| --- | --- |
| Recorded trials | 576 = 48 sessions x 12 |
| Parent sizes (train/val/test) | 460 / 116 / 115 |
| Parent test-in-val overlap | 115/115 = 100% of test |
| Parent val-only index | 460 |
| Parent session straddle | block 456..467 (`460 mod 12 = 4`) |
| Parent contract, disjoint+coverage only | 6/1001 (N=0..1000) |
| Parent contract, non-empty required | 0/1001 |
| Candidate split at 48x12, seed 42 | 348 / 120 / 108 |
| Candidate whole sessions | 29 / 10 / 9 |
| Candidate contract, non-empty | 160/160 grid inputs |
| Candidate peak traced memory, 48x12 | 42.9 KiB |
| Time, 48x12, best of 5 | parent 0.0042 ms; candidate 0.0758 ms |
| Time, n=100000, best of 5 | parent 0.8738 ms; candidate 10.5176 ms |
| Repeat-call output | identical |
| Cross-process output (PYTHONHASHSEED 0/1/12345) | identical digest `1b8bc0aa…` |
| Seed 7 vs seed 42 | different membership |
| Suite | `python -m pytest -q` 40 passed; `pytest -q` 40 passed |
| Plain-file runs | 36 passed (`tests/test_seedv_split.py`), 4 passed (root regression) |

## Corrections addressed (from the challenge scout)

- **Vacuous contract**: non-empty partitions are required; the 3-session case is 1/1/1 via
  largest-remainder plus repair, and the parent is recorded as 0/1001 under the stronger contract.
- **Ratio validation both ways**: NaN/Inf rejected before the sum check; `math.fsum` with `1e-9`
  tolerance accepts decimal triples summing to `0.9999999999999999`; negatives, zero, `1.0`, bool
  and float counts raise.
- **Unstateable "overlapping sessions"**: implemented as an internal `RuntimeError` guard plus the
  public `validate_partition`; the rectangular-model limitation is documented.
- **Subject-level leakage**: only session-level disjointness is claimed.
- **Seed sensitivity**: asserted only at 48 sessions, where many partitions exist.
- **Timing methodology**: best of 5 with warm calls, labelled machine-specific, no correctness
  weight; memory recorded at the goal size.
- **6 vs 0.8 drift**: the 0.6/0.2/0.2 default is documented as a deliberate difference from the
  notebook's `training_percent = 0.8`, and the notebook is explicitly not repaired.
- **Invocation-dependent green**: tests bootstrap `sys.path`; both `pytest` and `python -m pytest`
  pass without a `conftest.py`.

## Outcome

Retained and applied the seeded whole-session disjoint split. It reproduces the audit's
348/120/108 replacement, passes a non-vacuous structural contract on 160/160 measured inputs,
is deterministic across calls and processes, costs under 0.1 ms at the goal size, and is covered
by 40 green tests including a regression test that reproduces the parent's 115/115 leak.
No accuracy was measured and no notebook cell was executed.
