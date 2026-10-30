# Does the committed validation split of the SEED-V notebook contain the test set?

**Record:** `research/test-val-overlap-b9ead542/`
**Notebook under test:** `process_seedv_raw_files_with_mne_from_drive.ipynb`, cell 37 (0-based), `#@title Model Training`
**Notebook blob:** `4c31f50269e1f994c8fe89ed4baa5ff10af6e237` (HEAD `203ae91`)
**Status:** supported (overlap ratio 1.0, exactly)

## Abstract

The committed split block of cell 37 builds a test set `testList = range(train_samples + 1, N)` and a validation set via `train_test_split(X_train.indices_, test_size=val_percent, shuffle=False)`. We reproduced this arithmetic statically, without executing the notebook, using the stipulated `N = 576` and `training_percent = 0.8`. Two lines earlier the notebook reassigns `train_set = windows_dataset`, so `X_train.indices_` is the full positional range `0..575`; the validation partition is therefore `460..575` (116 indices) while the test set is `461..575` (115 indices). All 115 test indices lie in the validation set, giving an overlap ratio of exactly `115/115 = 1.0`. The pre-registered support criterion is met. A control that computes the validation split over `trainList` (1..460) gives overlap `0/115 = 0.0`, so the result depends entirely on the reassignment: the notebook's "validation" set is, in effect, the whole dataset, and every test window participates in model selection. The verdict is unchanged at `N = 533`, the value the notebook's own committed score tables imply, and for every `N >= 6` tested.

## Hypothesis

In the committed code of cell 37 (`training_percent = 0.8`, `N = 576`, `val_percent = 1 - training_percent`), the validation index set produced by `train_test_split(X_train.indices_, test_size=val_percent, shuffle=False)` contains every index of `testList = range(floor(0.8*576)+1, 576) = 461..575`, so that

    |test ∩ validation| / |test| = 1.0.

Pre-registered decision rule (fixed before measurement):

| criterion | rule |
|---|---|
| supported | ratio == 1.0 exactly |
| refuted | ratio < 0.5 |
| inconclusive | 0.5 <= ratio < 1.0 |

Additionally inconclusive if the measurement cannot be formed: `|test| = 0`, the code path is non-executable as written, or no defensible `N` can be fixed.

## Method

The notebook mounts Google Drive and depends on SEED-V raw data, so it was **not executed**. Only its committed JSON text was read. The throwaway script `experiment_index_arithmetic.py` in the record directory reproduces the index arithmetic; its output is `experiment_results.json`.

1. **Constant extraction (E1).** Cell 37 was read as text and every constant quoted verbatim, including `training_percent = 0.8` from cell 35 (the `#@param` slider cell 37 consumes), `val_percent = 1 - training_percent`, `trainList = range(1, train_samples + 1)`, `testList = range(train_samples + 1, total_array_samples)`, the `train_set = windows_dataset` reassignment, and the `train_test_split` arguments. Cell 37 begins with `%%capture` and stores no outputs.
2. **Primary reproduction (E2).** With `N = 576`:
   - `train_samples = floor(0.8 * 576) = 460`
   - `trainList = range(1, 461)` -> 1..460
   - `testList = range(461, 576)` -> 461..575
   - `val_percent = 1 - 0.8 = 0.19999999999999996`
   - validation base: `train_set` was reassigned to `windows_dataset`, and skorch's `SliceDataset` sets `indices_ = np.arange(len(dataset))`, so the base is `np.arange(576)`; `train_test_split(np.arange(576), test_size=0.19999999999999996, shuffle=False)`.
   The ratio is computed as an exact rational (`fractions.Fraction`), not a rounded float. Environment: Python 3.12.15, numpy 2.5.4, scikit-learn 1.9.1.
3. **Base-sensitivity control (E3).** The same split recomputed over `trainList` (`np.arange(1, 461)`) and over its zero-based variant (`np.arange(0, 460)`), quantifying the discrepancy induced by the reassignment.
4. **Static consistency audit (E4).** Partition coverage, label alignment, the `SliceDataset.indices_` question (checked against skorch's `helper.py` source), the window geometry that determines `len(windows_dataset)`, and what `total_array_samples` counts, including the count of scores `>= 4` in the committed score tables.

The reporter stage re-ran the arithmetic independently (see `round-1.md`, "Reporter verification") and corrected the round-1 description of the window geometry; no measurement changed.

## Results

### Primary measurement (E2)

| quantity | value |
|---|---|
| `N` | 576 |
| `train_samples = floor(0.8·N)` | 460 |
| `testList` | 461..575, **115** indices |
| validation base (`X_train.indices_`) | 0..575, 576 indices |
| sklearn `n_train` / `n_val` | 460 (0..459) / 116 (460..575) |
| \|test\| | 115 |
| \|validation\| | 116 |
| \|test ∩ validation\| | **115** |
| ratio \|test ∩ validation\| / \|test\| | **115/115 = 1/1 = 1.0 exactly** |
| test \ validation | ∅ (empty) |
| validation \ test | {460} |

The pre-registered support criterion is satisfied exactly. The refute criterion (ratio < 0.5) and the inconclusive band (0.5 <= ratio < 1.0) are not reached.

### Robustness

- **Rounding.** scikit-learn computes the float `test_size` with `ceil`, so `n_val = ceil(0.19999999999999996 · 576) = ceil(115.2) = 116`. Forcing `floor` (115) or `round` (115) gives validation `461..575`; the ratio remains `115/115 = 1/1`.
- **Choice of N.** A sweep over 2000 values of `N` (2..2000 plus 540, 560, 570, 576, 580, 600, 720, 2160) yields ratio `1/1` for every `N >= 6`; `N = 2..5` produce an empty `testList`, so the ratio is undefined and excluded (1996 values with a defined ratio). The overlap is structural: `N - ceil(0.2N) <= floor(0.8N) + 1`, so the contiguous validation block always starts at or before the first test index.
- **The N the notebook itself implies.** `total_array_samples = len(y)` counts score-filtered trials. The 48 committed score tables in cell 24 (16 subjects × 3 sessions × 15 annotations = 720 entries) contain **533** values `>= 4`, so the notebook's own metadata implies `N = 533`. At `N = 533`:

  | quantity | value |
  |---|---|
  | `train_samples` | 426 |
  | `testList` | 427..532, 106 indices |
  | validation | 426..532, 107 indices |
  | intersection | 106 |
  | ratio | **106/106 = 1/1** |

  The stipulated `N = 576` and the metadata-implied `N = 533` agree on the verdict.

### Base-sensitivity control (E3)

| base | validation indices | \|validation\| | \|test ∩ validation\| | ratio |
|---|---|---|---|---|
| full dataset `0..575` (committed) | 460..575 | 116 | 115 | 1/1 = 1.0 |
| `trainList` `1..460` | 369..460 | 92 | 0 | 0/1 = 0.0 |
| zero-based train subset `0..459` | 368..459 | 92 | 0 | 0/1 = 0.0 |

The counterfactual and primary results differ by 1.0. The entire result therefore hinges on one line: `train_set = windows_dataset`, which overwrites the training split two lines after it is created. A reader who assumes `X_train.indices_` refers to the training split would obtain the opposite conclusion.

### Static consistency audit (E4)

| item | finding |
|---|---|
| (a) partition | `trainList` (1..460) and `testList` (461..575) are disjoint and cover 1..575; index 0 is never selected (575 of 576 slots used) |
| (b) test lengths | `len(testList) = 115` and `len(y_test_set) = len(y[461:]) = 115` agree in count |
| (c) train labels | `y_train_set = y[0:460]` pairs labels 0..459 with window indices 1..460 — a one-sample label shift |
| (d) `SliceDataset.indices_` | exists: `self.indices_ = (self.indices if self.indices is not None else np.arange(len(self.dataset)))`; the committed line does **not** raise `AttributeError`, and the base is the full dataset |
| (e) counted quantity and base size | `total_array_samples = len(y)` counts score-filtered trials (720 committed scores, 533 of them `>= 4`); each trial yields exactly one window (cell 27: `tmin=-0, tmax=50`; all 45 committed start/end pairs differ by 50 s; cell 31 output: `sfreq: 200.0 Hz`; cell 33: 10000-sample window with `drop_last_window=True`), so `len(windows_dataset) = len(y) = N` and the E2 base `np.arange(N)` is derived from the committed code, not merely stipulated. Cell 37 is `%%capture`, so the printed `Total Array Samples` is absent. |

## Threats to validity

1. **N is not recorded in the notebook.** `total_array_samples = len(y)` depends on how many trials survive the `score >= 4` filter, and cell 37's print is captured. The hypothesis stipulates 576; the committed score tables imply 533; neither is written in a cell output. Mitigation: the same ratio (1/1) holds at both values and for every `N >= 6` swept, so this threat does not change the verdict.
2. **Index identity and the scope of the leakage claim.** The split is unshuffled and positional, and the dataset is one row per trial, so `val_subset = Subset(windows_dataset, 460..575)` and `test_set = Subset(windows_dataset, 461..575)` reference the same underlying windows: this is dataset leakage, not merely numerical coincidence. What the result does not establish is the author's actual kernel state or the accuracy they obtained; it establishes the committed code's split semantics.
3. **Library version.** `n_val` at the boundary depends on scikit-learn's ceil rule (1.9.1 used here; the notebook pins no version). Both ceil and floor give ratio 1.0, so a one-index difference is visible but immaterial.
4. **Static reading only.** The notebook is a stateful Colab artifact with `%%capture` suppressing cell 37 output, so behaviour under the author's actual run cannot be observed from the repository. The committed logic is the only admissible evidence.
5. **Ordering dependence.** Because the split is contiguous and unshuffled, the finding is a property of these index ranges and does not generalise to shuffled or stratified splits. It does not rely on a peculiar ordering of the data, only on positional index order 0..N-1.
6. **Scope of the claim.** This paper tests the index arithmetic of the committed code. It does not re-fit any model, and it does not claim that every downstream reported result is invalid beyond the specific model-selection leakage identified.

## Conclusion

The hypothesis is **supported**. Under the notebook's committed constants and split code, the validation set returned by `train_test_split(X_train.indices_, test_size=1-0.8, shuffle=False)` contains every test index: `|test ∩ validation| / |test| = 115/115 = 1.0` exactly, with `test \ validation = ∅`.

The mechanism is a reassignment, not index luck: `train_set` is overwritten with the full `windows_dataset` before `X_train` is constructed, so the validation partition covers all `N` trials (the full dataset has one row per trial). The control over `trainList` yields ratio 0.0, a discrepancy of 1.0, showing that the notebook as committed performs no honest held-out validation - the entire test set participates in `predefined_split(val_subset)` during training. Independently of the overlap question, the split block drops index 0 and shifts training labels by one, which is worth correcting.

## Reproducibility

    python3 research/test-val-overlap-b9ead542/experiment_index_arithmetic.py

writes `experiment_results.json` in the same directory. The reporter's independent re-derivation (including `N = 533` and the sweep qualification) is recorded in `round-1.md`. The notebook itself is never executed and the target repository is unmodified outside `research/`.
