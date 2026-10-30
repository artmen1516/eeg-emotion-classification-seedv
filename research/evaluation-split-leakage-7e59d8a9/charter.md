# Charter

## Question

Does the committed notebook (`process_seedv_raw_files_with_mne_from_drive.ipynb`, 43 cells, 33 code)
define an evaluation protocol that can support the repository's only recorded result
(`Test acc: 24.35%`, cell 39)? If not, what is the single smallest, evidence-adoptable change that
repairs it?

## Scope

- In scope: the committed notebook JSON, `README.md`, and git history.
- Out of scope: executing the notebook, installing packages, mounting Google Drive, or reading the
  SEED-V raw data (absent from the repository). The published article was not read in this round.
- Repository behavior is treated as read-only. Only files under `research/` may be written.

## Constraints

- Every claim must be traceable to a cell index and literal source or committed output.
- Recorded outputs are documentation of a past run, not reproduced measurements: all 43 cells have
  `execution_count: null`, and the cell-39 timestamp (`1724185416`, 2024-08-20) predates the notebook
  import commit (2024-08-31).
- No claim may be made about model accuracy, the article's reported numbers, or subject-level
  ordering that the committed artifact does not expose.

## Success criteria

1. One retained improvement, local to the notebook, with a parent construction and a measured
   replacement.
2. Correctness measured over inputs `N = 0..1000` and speed measured at `n = 100000` (best of 5),
   both by pure-Python re-derivation, never by running the notebook.
3. Evidence that the parent fails and the replacement passes the stated structural contract.
4. A record that separates the split defect from the loss, baseline, window-length, environment and
   Drive-coupling findings.
5. `git status --porcelain` shows no changes outside `research/`; the notebook hash is unchanged.
