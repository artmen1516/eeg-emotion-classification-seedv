# Charter

## Question

The prior audit (`research/evaluation-split-leakage-7e59d8a9`) retained one minimal improvement to
the committed notebook: a seeded, whole-session, pairwise-disjoint train/validation/test split with
a fail-fast assertion. Can that improvement be validated read-only against the committed notebook
and then shipped as a small, independently testable artifact without modifying the notebook or
README?

## Scope

- In scope: `process_seedv_raw_files_with_mne_from_drive.ipynb` (parsed as JSON only), the prior
  audit record, `README.md`, git history, and new additive files at the repository root and under
  `tests/`.
- Out of scope: executing the notebook, mounting Google Drive, reading SEED-V raw data, installing
  numpy/sklearn, or re-running the notebook pipeline.
- Application is additive: `seedv_split.py`, `tests/test_seedv_split.py`,
  `test_repro_split_disjoint.py`, and research artifacts. Existing tracked files are not modified.

## Constraints

- Every claim about the notebook must trace to a cell index and committed literal; no cell is
  executed.
- No accuracy, loss, or article-level claim may be made.
- The split contract is structural: non-empty, pairwise disjoint, exact-once coverage, and
  whole-session integrity.
- Tests must run in this environment (Python 3.12, no numpy/sklearn) with the standard library
  only, and be green under both `pytest` and `python -m pytest`.
- No `.fabro-reviews/` or `.zeroshot/` directories; no tracked-file changes outside `research/`
  except the three application files.

## Success criteria

1. The parent defect is re-derived from committed literals: at 48x12, test 461..575 is a 115/115
   subset of val 460..575, and block 456..467 straddles train/validation.
2. `split_indices(...)` reproduces the retained replacement at 48x12 with seed 42: 348/120/108
   over 29/10/9 whole sessions, deterministic per seed.
3. The candidate passes the non-vacuous contract across a grid of session/trial shapes; the parent
   is shown to fail it.
4. A regression test encodes the parent's failure and the replacement's repair, including a
   mutation guard, and is discovered by a root-level `test_repro*.py` glob.
5. The suite is green under both invocation styles; the notebook hash is unchanged; hygiene checks
   pass.
