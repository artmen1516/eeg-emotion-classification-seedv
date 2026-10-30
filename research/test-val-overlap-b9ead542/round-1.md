# Round 1 — evidence log

**Plan:** `research/hypothesis-plan.json` (hash8 `b9ead542`)
**Record:** `research/test-val-overlap-b9ead542/`
**Experiment code:** `experiment_index_arithmetic.py` → `experiment_results.json`
**Notebook executed:** no. Only its committed JSON text was read; the SEED-V Drive mount was never touched.

## Environment

- Python 3.12.15, numpy 2.5.4, scikit-learn 1.9.1 (installed into the throwaway environment only)
- Notebook HEAD `203ae91`, blob `4c31f50269e1f994c8fe89ed4baa5ff10af6e237`
- Cell 37 (`#@title Model Training`) starts with `%%capture` and has no committed outputs.

## E1 — committed constants (verbatim)

```
training_percent = 0.8 # @param {type:"slider", min:0.5, max:0.8, step:0.1}   [cell 35]
total_array_samples = len(y)
train_samples = math.floor((training_percent*total_array_samples))
trainList =  list(range(1, train_samples + 1, 1))
testList =  list(range(train_samples + 1, total_array_samples, 1))
val_percent = 1 - training_percent
splits = windows_dataset.split({"train": trainList, "test": testList})
train_set = splits['train']
test_set = splits['test']
y_train_set = y[0:train_samples]
y_test_set = y[train_samples + 1:]
train_set = windows_dataset                      # <-- overwrite
X_train = SliceDataset(train_set, idx=0)
train_indices, val_indices = train_test_split(
    X_train.indices_, test_size=val_percent, shuffle=False
)
val_subset = Subset(train_set, val_indices)
```

## E2 — primary measurement

Stipulated `N = 576`; `val_percent = 1 - 0.8 = 0.19999999999999996` (exact rational `900719925474099/4503599627370496`).

| quantity | value |
|---|---|
| `train_samples` | `floor(0.8·576) = 460` |
| `trainList` | 1..460 (460) |
| `testList` | 461..575 (115) |
| validation base | `np.arange(576)` because `SliceDataset(windows_dataset).indices_ = arange(len(dataset))` |
| sklearn split | n_train = 460 (0..459), n_val = 116 (460..575) |
| \|test ∩ validation\| | 115 |
| ratio | **115/115 = 1/1 = 1.0** |
| test \ validation | ∅ |
| validation \ test | {460} |

`ceil(0.19999999999999996 · 576) = 116`; forcing `floor`/`round` gives validation 461..575 and still ratio `1/1`.

Supplementary N-sweep: 2000 values of N (2..2000 plus 540, 560, 570, 576, 580, 600, 720, 2160) → single distinct ratio `1/1`. Verdict is insensitive to N and to the ceil/floor boundary.

## E3 — base-sensitivity control

| base | validation | \|test ∩ validation\| | ratio |
|---|---|---|---|
| `np.arange(1, 461)` (trainList) | 369..460 (92) | 0 | 0/1 = 0.0 |
| `np.arange(0, 460)` (zero-based train subset) | 368..459 (92) | 0 | 0/1 = 0.0 |

Discrepancy versus E2: **1.0**. The primary result exists only because `train_set` was overwritten with `windows_dataset` two lines before `X_train` was built. This is recorded as a finding, not as a replacement for the primary verdict.

## E4 — static audit

- (a) `trainList` ∪ `testList` = 1..575, disjoint; index 0 never selected. 575/576 slots used.
- (b) `len(testList) = 115` equals `len(y_test_set) = len(y[461:]) = 115`; counts agree.
- (c) `y_train_set = y[0:460]` pairs labels 0..459 with windows 1..460 — one-sample label shift.
- (d) `SliceDataset.indices_` exists in skorch (`self.indices_ = self.indices if self.indices is not None else np.arange(len(self.dataset))`); no `AttributeError`. The base is the full dataset, confirming the E2 reading.
- (e) `total_array_samples = len(y)` counts score-filtered epochs/trials (`list_of_epochs`, `score >= 4`), at most 15 × 3 × 16 = 720; `windows_dataset` rows are 50 s windows (0.01 s stride, `drop_last_window=True`). The split lists derived from trial counts are applied to a window-indexed dataset. Cell 37 is `%%capture`, so N = 576 is not confirmed by any committed output.

## Criterion comparison

| criterion | threshold | measured | met |
|---|---|---|---|
| support | ratio == 1.0 | 1/1 = 1.0 | yes |
| refute | ratio < 0.5 | 1.0 | no |
| inconclusive | 0.5 <= ratio < 1.0 | 1.0 | no |
| inconclusive (unformable) | \|test\| = 0, non-executable, no defensible N | \|test\| = 115; `indices_` exists; ratio invariant over 2000 N | no |

**Status: supported.**
