# Does the committed validation split of the SEED-V notebook contain the test set?

**Record:** `research/test-val-overlap-b9ead542/`
**Notebook under test:** `process_seedv_raw_files_with_mne_from_drive.ipynb`, cell 37 (0-based), `#@title Model Training`
**Notebook blob:** `4c31f50269e1f994c8fe89ed4baa5ff10af6e237` (HEAD `203ae91`)
**Status:** supported

## Abstract

The committed split block of cell 37 builds a test set `testList = range(train_samples + 1, N)` and a validation set via `train_test_split(X_train.indices_, test_size=val_percent, shuffle=False)`. We reproduced this arithmetic statically, without executing the notebook, using the stipulated `N = 576` and `training_percent = 0.8`. With the notebook's committed reassignment `train_set = windows_dataset`, `X_train.indices_` is the full positional range `0..575`, so the validation partition is `460..575` (116 indices) while the test set is `461..575` (115 indices). All 115 test indices lie in the validation set, giving an overlap ratio of exactly `115/115 = 1.0`. The pre-registered support criterion is met. A control in which validation is computed over `trainList` (1..460) gives overlap `0/115 = 0.0`, so the result depends entirely on the reassignment; the notebook's "validation" set is, in effect, the full dataset, and every test window is used during model selection.

## Hypothesis

In the committed code of cell 37 (`training_percent = 0.8`, `N = 576`, `val_percent = 1 - training_percent`), the validation index set produced by `train_test_split(X_train.indices_, test_size=val_percent, shuffle=False)` contains every index of `testList = range(floor(0.8*576)+1, 576) = 461..575`, so that

    |test ∩ validation| / |test| = 1.0.

Pre-registered decision rule (fixed before measurement):

| criterion | rule |
|---|---|
| supported | ratio == 1.0 exactly |
| refuted | ratio < 0.5 |
| inconclusive | 0.5 <= ratio < 1.0 |

Additionally inconclusive if the measurement cannot be formed (|test| = 0, the code path is non-executable as written, or no defensible N can be fixed).

## Method

The notebook mounts Google Drive and depends on SEED-V raw data, so it was **not executed**. Instead:

1. **Constant extraction (E1).** The notebook JSON was parsed and cell 37 read as text. Every constant used below is quoted verbatim from the file; `training_percent = 0.8` comes from cell 35 (the `#@param` slider that cell 37 consumes).
2. **Primary reproduction (E2).** A throwaway script (`experiment_index_arithmetic.py` in the record directory) reproduced the committed expressions with `N = 576`:
   - `train_samples = floor(0.8 * 576) = 460`
   - `trainList = range(1, 461)` → 1..460
   - `testList = range(461, 576)` → 461..575
   - `val_percent = 1 - 0.8 = 0.19999999999999996`
   - validation base: `train_set` was reassigned to `windows_dataset`, and skorch's `SliceDataset` sets `indices_ = np.arange(len(dataset))`, so the base is `np.arange(576)`; `train_test_split(np.arange(576), test_size=0.19999999999999996, shuffle=False)`.
   The ratio is computed as an exact rational (`fractions.Fraction`), not a rounded float. Environment: Python 3.12.15, numpy 2.5.4, scikit-learn 1.9.1.
3. **Base-sensitivity control (E3).** The same split recomputed over `trainList` (`np.arange(1, 461)`) and over its zero-based variant (`np.arange(0, 460)`), quantifying the discrepancy induced by the reassignment.
4. **Static consistency audit (E4).** Partition coverage, label alignment, the `SliceDataset.indices_` question (checked against skorch's `helper.py` source), and what `total_array_samples` actually counts.

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

- **Rounding.** scikit-learn computes a float `test_size` with `ceil`, so `n_val = ceil(0.19999999999999996 · 576) = ceil(115.2) = 116`. Forcing `floor` (115) or `round` (115) instead gives validation `461..575`; the ratio remains `115/115 = 1/1`.
- **Choice of N.** A supplementary sweep over 2000 values of `N` (2..2000 plus 540, 560, 570, 576, 580, 600, 720, 2160) produced the single distinct ratio `1/1`. The overlap is a structural property of `floor(0.8N)+1` versus `N - ceil(0.2N)`, not of the value 576.

### Base-sensitivity control (E3)

| base | validation indices | \|validation\| | \|test ∩ validation\| | ratio |
|---|---|---|---|---|
| full dataset `0..575` (committed) | 460..575 | 116 | 115 | 1/1 = 1.0 |
| `trainList` `1..460` | 369..460 | 92 | 0 | 0/1 = 0.0 |
| zero-based train subset `0..459` | 368..459 | 92 | 0 | 0/1 = 0.0 |

The counterfactual and primary results differ by 1.0. The entire result therefore hinges on one line: `train_set = windows_dataset`, which overwrites the training split only two lines after it is created. A reader who assumes `X_train.indices_` refers to the training split would obtain the opposite conclusion.

### Static consistency audit (E4)

| item | finding |
|---|---|
| (a) partition | `trainList` (1..460) and `testList` (461..575) are disjoint and cover 1..575; index 0 is never selected (575 of 576 slots used) |
| (b) test lengths | `len(testList) = 115` and `len(y_test_set) = len(y[461:]) = 115` agree in count |
| (c) train labels | `y_train_set = y[0:460]` pairs labels 0..459 with window indices 1..460 — a one-sample label shift |
| (d) `SliceDataset.indices_` | exists: `self.indices_ = (self.indices if self.indices is not None else np.arange(len(self.dataset)))`; the committed line does **not** raise `AttributeError`, and the base is the full dataset |
| (e) counted quantity | `total_array_samples = len(y)` counts score-filtered **epochs/trials** (at most 15 annotations × 3 sessions × 16 subjects = 720), whereas the index lists are applied to `windows_dataset`, whose rows are 50 s windows with 0.01 s stride; cell 37 is `%%capture`, so the actual count is not recorded |

## Threats to validity

1. **N is stipulated, not observed.** `total_array_samples = len(y)` depends on how many trials survive the `score >= 4` filter; the committed notebook stores no output for cell 37, so 576 cannot be confirmed from the file. Mitigation: the E2c sweep shows the ratio is 1.0 for every tested N, so this threat does not change the verdict.
2. **Index identity across different spaces.** The split lists are derived from a trial count but index a window-level dataset. "Index overlap" here is positional and assumes stable, sequential indices; it establishes overlap of the committed index sets, not that the same labelled examples were duplicated. Under the notebook's own semantics, `val_subset = Subset(windows_dataset, 460..575)` and `test_set = Subset(windows_dataset, 461..575)` do reference the same underlying windows, so the leakage claim is about the datasets, not merely the numbers.
3. **Library version.** `n_test` at the boundary depends on scikit-learn's ceil rule (1.9.1 used here, the notebook pinned no version). Both ceil and floor give ratio 1.0, so a one-index difference is visible but immaterial.
4. **Static reading only.** The committed logic is the only admissible evidence; the notebook is a stateful Colab artifact with `%%capture` suppressing cell 37 output, so behaviour under the author's actual kernel state (e.g. how many windows `windows_dataset` really contained) is unknowable from the repository.
5. **Ordering dependence.** The split is unshuffled and contiguous, so the finding is an artefact of sequential positional ordering and does not generalise to shuffled or stratified splits.
6. **Scope of the claim.** This paper tests the index arithmetic of the committed code. It does not claim that the author's reported results are invalid beyond the specific leakage mechanism identified, nor does it re-fit any model.

## Conclusion

The hypothesis is **supported**. Under the notebook's committed constants and split code, the validation set returned by `train_test_split(X_train.indices_, test_size=1-0.8, shuffle=False)` contains every test index: `|test ∩ validation| / |test| = 115/115 = 1.0` exactly, with `test \ validation = ∅`.

The mechanism is not subtle index luck but a reassignment: `train_set` is overwritten with the full `windows_dataset` before `X_train` is constructed, so the validation partition is taken over all 576 (stipulated) windows. The control over `trainList` yields a ratio of 0.0, a discrepancy of 1.0, and shows that the notebook as committed performs no honest held-out validation — the entire test set participates in `predefined_split(val_subset)` during training. Beyond the primary finding, the split block drops sample 0, shifts training labels by one, and indexes a window-level dataset with trial-level counts, which is worth correcting independently of the overlap question.

## Reproducibility

    python3 research/test-val-overlap-b9ead542/experiment_index_arithmetic.py

writes `experiment_results.json` in the same directory. The notebook itself is never executed and the target repository is unmodified outside `research/`.
