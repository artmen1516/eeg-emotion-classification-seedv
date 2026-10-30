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

Committed score tables (cell 24): 48 lists (16 subjects × 3 sessions), 15 entries each, 720 entries total, **533** values `>= 4`.

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

## E2c — sensitivity to N, and the N the notebook implies

- Sweep over 2000 values of N (2..2000 plus 540, 560, 570, 576, 580, 600, 720, 2160): every N with a non-empty `testList` (N >= 6, 1996 values) gives ratio `1/1`. N = 2, 3, 4, 5 give `testList = []`, so the ratio is undefined; the script's `if r is not None` filter excludes them and the claim is restricted accordingly.
- Structural reason: validation starts at `N - ceil(0.2N) <= floor(0.8N) + 1`, the first test index, and both runs end at `N-1`.
- Metadata-implied `N = 533`: `train_samples = 426`, `testList = 427..532` (106), validation `426..532` (107), intersection 106, ratio `106/106 = 1/1`. Same verdict as `N = 576`.

## E3 — base-sensitivity control

| base | validation | \|test ∩ validation\| | ratio |
|---|---|---|---|
| `np.arange(1, 461)` (trainList) | 369..460 (92) | 0 | 0/1 = 0.0 |
| `np.arange(0, 460)` (zero-based train subset) | 368..459 (92) | 0 | 0/1 = 0.0 |

Discrepancy versus E2: **1.0**. The primary result exists only because `train_set` was overwritten with `windows_dataset` two lines before `X_train` was built. This is recorded as a finding, not as a replacement for the primary verdict. At `N = 533` the same control gives validation 341..426, intersection 0.

## E4 — static audit

- (a) `trainList` ∪ `testList` = 1..575, disjoint; index 0 never selected. 575/576 slots used.
- (b) `len(testList) = 115` equals `len(y_test_set) = len(y[461:]) = 115`; counts agree.
- (c) `y_train_set = y[0:460]` pairs labels 0..459 with windows 1..460 — one-sample label shift.
- (d) `SliceDataset.indices_` exists in skorch (`self.indices_ = self.indices if self.indices is not None else np.arange(len(self.dataset))`); no `AttributeError`. The base is the full dataset, confirming the E2 reading.
- (e) `total_array_samples = len(y)` counts score-filtered trials: cell 27 uses `mne.Epochs(..., tmin=-0, tmax=50, baseline=None)`; every committed start/end pair in cell 22 differs by exactly 50 s; cell 31's committed output gives `sfreq: 200.0 Hz`; cell 33 uses a 50 s (10000-sample) window with `drop_last_window=True`. A 10000-sample window over a 10000-sample epoch yields **exactly one window per trial**, so `len(windows_dataset) = len(y) = N` and the E2 base is derived from the committed code. N itself is not printed (cell 37 is `%%capture`); the score tables imply 533, the hypothesis stipulates 576, and both give ratio 1.0.

**Correction vs. the first draft of this log:** an earlier version described `windows_dataset` as having far more rows than `len(y)` because of the 0.01 s stride. That was wrong — the stride is inside a window equal to the full epoch length, so each trial contributes one window. No measurement changed; only the description of the mechanism was corrected.

## Reporter verification (independent re-derivation)

Run from the repository root, no notebook execution:

```python
import json, math, re
from fractions import Fraction
from sklearn.model_selection import train_test_split
import numpy as np, sklearn

nb = json.load(open('process_seedv_raw_files_with_mne_from_drive.ipynb'))
src = ''.join(nb['cells'][24]['source'])
lists = re.findall(r'^subject(\d+)_session(\d+)=\[([0-9,]+)\]', src, re.M)
total = sum(len(v.split(',')) for _, _, v in lists)
ge4 = sum(1 for _, _, v in lists for x in v.split(',') if int(x) >= 4)

for N in (576, 533):
    ts = math.floor(0.8 * N)
    test = list(range(ts + 1, N))
    tr, val = train_test_split(np.arange(N), test_size=1 - 0.8, shuffle=False)
    inter = len(set(test) & set(val.tolist()))
    print(N, ts, test[0], test[-1], len(test), val[0], val[-1], len(val),
          inter, Fraction(inter, len(test)))
```

Output:

```
lists 48
scores total 720 >=4 533
sklearn 1.9.1
576 train_samples 460 test 461 575 115 val 460 575 116 inter 115 ratio 1
   control base trainList: 369 460 92 inter 0
533 train_samples 426 test 427 532 106 val 426 532 107 inter 106 ratio 1
   control base trainList: 341 426 86 inter 0
```

Sweep qualification check: for N in 2..2000, no N with a non-empty testList has a ratio other than 1; undefined for N = 2, 3, 4, 5.

## Criterion comparison

| criterion | threshold | measured | met |
|---|---|---|---|
| support | ratio == 1.0 | 1/1 = 1.0 | yes |
| refute | ratio < 0.5 | 1.0 | no |
| inconclusive | 0.5 <= ratio < 1.0 | 1.0 | no |
| inconclusive (unformable) | \|test\| = 0, non-executable, no defensible N | \|test\| = 115 (576) / 106 (533); `indices_` exists; ratio 1.0 for all N >= 6 | no |

**Status: supported.**
