# Research: evaluation-split leakage in the SEED-V notebook

**Repository:** `artmen1516/eeg-emotion-classification-seedv`
**Artifact under review:** `process_seedv_raw_files_with_mne_from_drive.ipynb` (43 cells) + `README.md`
**Article:** Brain Sciences 14(12):1211, DOI 10.3390/brainsci14121211
**Status:** read-only study; the notebook was never executed and the SEED-V data was not accessed.

## Question

The repository ships one recorded measurement, `Test acc: 24.35%` (cell 39). Does the committed
notebook define an evaluation split that can support it, and if not, what is the smallest change that
repairs the split?

## Answer

No. The test set is a subset of the validation set, so there is no held-out measurement in the
repository. The retained improvement replaces the cell-37 index arithmetic with **one seeded,
whole-session, pairwise-disjoint train/val/test partition** and a fail-fast assertion.

## What the notebook actually does

Cell 32's committed output is 48 label vectors of length 12: 576 trials, class counts
Disgust 144 / Fear 96 / Sad 112 / Neutral 144 / Happy 80. Cell 37 then contains, verbatim:

```python
train_samples = math.floor((training_percent*total_array_samples))
trainList = list(range(1, train_samples + 1, 1))
testList = list(range(train_samples + 1, total_array_samples, 1))
splits = windows_dataset.split({"train": trainList, "test": testList})
train_set = splits['train']
test_set = splits['test']
y_test_set = y[train_samples + 1:]
train_set = windows_dataset                       # discards splits['train']
train_indices, val_indices = train_test_split(
    X_train.indices_, test_size=val_percent, shuffle=False)
```

With `training_percent = 0.8` and `N = 576`:

| Partition | Indices | Size |
| --- | --- | --- |
| declared test (`testList`) | 461..575 | 115 |
| validation (`train_test_split`) | 460..575 | 116 |
| training subset | 0..459 | 460 |

`test ∩ val = 115`, i.e. **100% of the test set is inside validation**; index 460 is validation-only.
`test \ val = ∅`. The line `train_set = windows_dataset` means validation is carved from all 576
trials, not from `splits['train']`. Because cell 38 displays validation accuracy while `model_option`
and `training_percent` are selected, every test trial has already influenced selection.

At the recorded block size of 12 trials per file, `460 mod 12 = 4`, so the block at indices 456..467
straddles the train/val boundary: at least one recording session contributes to both partitions. No
subject-level ordering claim is made, because cells 25 and 29 are `%%capture` and expose no file list.

## Why this is the highest-value fix

1. It is the precondition for any quantitative claim: without a held-out set, the only recorded number
   is a validation re-measurement.
2. It is local to one cell, needs no new data, and is independent of the loss and baseline defects.
3. It is exactly checkable from committed literals, without running the notebook.

## The retained change

Replace the cell-37 split block with a seeded whole-group partition built from the metadata that
cell 27 already writes (`subject`, `session`). The pure-Python analogue that was tested:

```python
import random, numpy as np

def seeded_grouped_split(metadata, seed=20240820):
    n = len(metadata)
    groups = list(zip(metadata["subject"], metadata["session"]))
    order = sorted(set(groups))
    random.Random(seed).shuffle(order)
    members = {g: [] for g in order}
    for i, g in enumerate(groups):
        members[g].append(i)
    train, val, test = [], [], []
    t_train, t_val = round(0.6 * n), round(0.2 * n)
    for pos, g in enumerate(order):
        remaining = len(order) - pos
        if len(train) < t_train and (remaining > 2 or train):
            train.extend(members[g])
        elif len(val) < t_val and (remaining > 1 or val):
            val.extend(members[g])
        else:
            test.extend(members[g])
    assert set(train) | set(val) | set(test) == set(range(n))
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    return train, val, test
```

`test_set` and `y_test_set` must then be built from this test partition, and `train_set` must not be
reassigned to the full `windows_dataset`.

## Experiment

Method: pure-Python re-derivation; the notebook was parsed as JSON and never executed. Inputs
`N = 0..1000` for correctness; `n = 100000` for speed, best of 5 wall-clock runs. Candidate bytes were
verified and restored: notebook sha256 `35d98aa40078b0e6a60b65cf79e2f206827468bb95f1196deeff564cda5142fc`
before and after, and no repository file was written (experiment script in `/tmp`).

Contract checked for every `N`: pairwise-disjoint train/val/test, union `{0..N-1}`, and no effective
group split across partitions (whole 12-trial session blocks when `N >= 36`, per-trial fallback below
that).

| Measurement | Parent (cell 37) | Replacement |
| --- | --- | --- |
| Contract pass, `N=0..1000` | 6/1001 | 1001/1001 |
| Disjointness failures | 995 | 0 |
| Group-integrity failures | 995 | 0 |
| Split at `N=576` | 460 / 116 / 115 | 348 / 120 / 108 |
| Fractions at `N=576` | 79.86 / 20.14 / 19.97% | 60.42 / 20.83 / 18.75% |
| Same-seed determinism | n/a (unseeded) | identical |
| Different-seed sensitivity | n/a | different |
| `n=100000`, best of 5 | 1.923 ms | 8.439 ms |

The parent fails for every `N >= 6` because `testList` is contained in the validation slice. The
replacement is ~4.4x slower in absolute terms but under 10 ms for 100k indices.

## The recorded result is null-compatible (separate finding)

On the recorded cell-32 labels, always-predict-Neutral scores:

- train 115/460 = 25.00%
- val 29/116 = 25.00%
- test 28/115 = 24.35%

These reproduce the recorded `0.25` / `0.25` (cell 38) and `24.35%` (cell 39) exactly. The test slice's
majority class is Disgust at 29/115 = 25.22%, so the pipeline is below the trivial baseline. Exact
binomial: p = 0.2441 vs 0.20, p = 0.9146 vs 29/115; Wilson 95% CI [17.42%, 32.94%]. This is a
quantitative identity about the recorded labels, not a claim about the model's internals.

## Other context findings (not bundled into the fix)

- **Loss:** cell 37 uses `torch.nn.NLLLoss`, cell 35's summary ends at raw logits (`[-1, 5]`, no
  log-softmax), and cell 38's 100-row intrinsic summary reports all-negative losses (train min
  -753759.28, valid min -955689.18). The objective is inconsistent with the stated criterion.
- **Provenance:** cell 24's score tables imply 533 score>=4 trials (7-15 per session; class counts
  98/116/91/134/94), not the 576 at 12/file in cell 32, so the recorded inputs cannot be regenerated
  from the committed preprocessing cells.
- **Reproducibility:** zero `manual_seed` / `np.random.seed` / `random.seed` / `random_state` calls;
  order from unsorted `os.listdir`; `mne` unpinned and `braindecode` installed at 0.8 then replaced by
  0.8.1; no requirements/lock file; six `/content/drive` paths; both `.save()` calls commented out;
  all 43 cells have `execution_count: null`; cell-39 timestamp 2024-08-20 predates the 2024-08-31
  notebook commit.
- **Windowing:** cell 27 epochs 50 s and cell 33 windows 50 s, so exactly one 10,000-sample window per
  trial is produced and the 0.01 s stride is inert.

## Scope and honesty

- Nothing was executed. No accuracy was reproduced; the SEED-V data is absent.
- N = 576 is inferred from committed outputs (48 x 12), not from a run.
- Test ⊆ validation is exact arithmetic from committed literals and holds for every `N >= 6`.
- The constant-Neutral identity is scoped to the recorded cell-32 label vectors.
- Only session-level leakage is claimed; no subject-group numbering and no listdir-order claim.
- No claim is made about the article's reported figures.
- All writes are under `research/`; the notebook and README are unchanged.

## Artifacts

- `research.json` - retained improvement with parent, experiment and evidence
- `review.json` - verdict and corrections
- `report.json` - findings for the retained result
- `charter.md`, `summary.md`, `round-1.md` - research records
- `scout-explore.json`, `scout-synthesize.json`, `scout-challenge.json` - transient inputs
