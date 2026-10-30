# Summary

**Retained improvement: one seeded, whole-session, disjoint train/val/test split in cell 37.**

The notebook's current split cannot produce a held-out number. Cell 37 sets
`train_samples = math.floor(0.8 * N)`, builds `testList = range(train_samples + 1, N)`, then carves
validation out of all `N` indices with an unseeded `train_test_split(..., test_size=0.2, shuffle=False)`
after reassigning `train_set = windows_dataset`. For the 576 recorded trials (48 vectors x 12 from
cell 32) that gives train 460, val 116 (460..575) and test 115 (461..575): test is entirely inside
validation (115/115), with only index 460 validation-only. Cell 39's `Test acc: 24.35%` is therefore a
validation re-measurement, and because validation accuracy is displayed while `model_option` and
`training_percent` are chosen, every test trial has already been used for selection.

The replacement groups trials by the subject/session metadata that cell 27 already writes, shuffles
whole groups with a fixed seed, assigns them at 60/20/20, and asserts pairwise disjointness plus full
coverage before building the subsets. It is a one-cell change, needs no new data, and does not depend
on the loss or baseline defects.

## Verification (read-only, nothing executed)

- Parent contract over `N = 0..1000`: 6/1001 pass; disjointness fails 995 times and session-group
  integrity fails 995 times (all `N >= 6`).
- Replacement: 1001/1001 pass; at `N = 576` it yields 348/120/108 (60.42/20.83/18.75%), deterministic
  under a fixed seed and different under another seed.
- Speed at `n = 100000`, best of 5: 1.923 ms (parent) vs 8.439 ms (replacement).
- Repository unchanged: notebook sha256 `35d98aa4...a5142fc` before and after; all writes under
  `research/`.

## Context findings (not part of the retained fix)

- The recorded accuracy equals the constant-Neutral baseline exactly (25.00%/25.00%/24.35% on
  train/val/test) and sits below the test-slice majority (Disgust 29/115 = 25.22%). The recorded
  result is null-compatible: exact binomial p=0.2441 vs chance, p=0.9146 vs the constant baseline;
  Wilson 95% CI [17.42%, 32.94%].
- Cell 38's intrinsic summary covers all 100 epochs and reports train/valid accuracy `std=0.0`,
  `min=max=0.25`; losses are negative (train min -753759.28, valid min -955689.18) although cell 37
  uses `torch.nn.NLLLoss` and cell 35 shows no log-softmax layer.
- The committed score tables (cell 24) imply 533 score>=4 trials (7-15 per session;
  98/116/91/134/94 by class), not the 576 at 12/file shown in cell 32, so the recorded epoch inputs
  cannot be regenerated from the committed preprocessing cells.
- No seeds anywhere, unpinned `mne`/`braindecode`, six `/content/drive` paths, commented-out
  `.save()` calls, and `execution_count: null` for all 43 cells.

## What is not claimed

Nothing was executed. No accuracy was reproduced, no subject-level ordering is asserted, and no claim
is made about the published article's reported figures.
