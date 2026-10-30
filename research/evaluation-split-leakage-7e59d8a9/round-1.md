# Round 1

## Inputs

- `scout-explore.json`, `scout-synthesize.json`, `scout-challenge.json` (moved into this directory).
- `process_seedv_raw_files_with_mne_from_drive.ipynb`, parsed as JSON; `README.md`; `git log`.
- No execution, no network, no Drive, no SEED-V data.

## What was done

1. Read all three scouts and the prior reviewer pass in this directory. The prior reviewer rejected
   an empty submission and left a correction list; this round addresses it rather than re-scouting.
2. Re-derived the committed evidence directly from the notebook JSON:
   - cell 32 output: 48 label vectors of length 12 = 576 trials, counts 144/96/112/144/80;
   - cell 37 source: `train_samples = math.floor((training_percent*total_array_samples))`,
     `trainList = range(1, train_samples + 1)`, `testList = range(train_samples + 1, total_array_samples)`,
     `train_set = windows_dataset`, unseeded `train_test_split(..., test_size=0.2, shuffle=False)`;
   - cell 38 intrinsic summary: 100 rows, train/valid accuracy `std=0.0`, `min=max=0.25`, losses all
     negative (train min -753759.28, valid min -955689.18);
   - cell 39: `Test acc: 24.35%`, metadata line, 115 preload lines, timestamp 1724185416.
3. Selected the parent `cell-37-index-split` and the experiment: pure-Python re-derivation of the
   cell-37 split plus a seeded whole-session replacement, correctness over `N = 0..1000`, timing at
   `n = 100000` best of 5.
4. Verified and restored candidate bytes: recorded the notebook sha256 before and after
   (`35d98aa4...a5142fc`); no repository file was written. The experiment script lives in `/tmp`,
   outside the repository.
5. Applied the prior reviewer's corrections: test-subset direction, 100-row accuracy scope,
   label-provenance mismatch, no subject-group numbering, external facts not treated as repository
   evidence, and unbundling of the loss/baseline/window findings.

## Measurements

| Quantity | Value |
| --- | --- |
| Recorded trials | 576 = 48 vectors x 12 |
| train / val / test sizes | 460 / 116 / 115 |
| test-in-validation overlap | 115/115 = 100% of test |
| val-only index | 460 |
| Session block straddling train/val | 456..467 (`460 mod 12 = 4`) |
| Constant-Neutral accuracy | 25.00% / 25.00% / 24.35% (train/val/test) |
| Test-slice majority | Disgust 29/115 = 25.22% |
| Exact binomial 28/115 vs 0.20 | p = 0.2441 |
| Wilson 95% CI | [17.42%, 32.94%] |
| Contract pass, parent | 6/1001 |
| Contract pass, replacement | 1001/1001 |
| Replacement split at N=576 | 348 / 120 / 108 (60.42 / 20.83 / 18.75%) |
| Runtime, n=100000, best of 5 | parent 1.923 ms; replacement 8.439 ms |

## Corrections carried forward

- The scouts' claim that the recorded 12-id vectors are "valid subsequences" consistent with cell 22/27
  is not a consistency proof. Cell 24's scores give 533 score>=4 trials (7-15 per session;
  98/116/91/134/94 by class); the recorded output is 576 at 12/file. The recorded inputs are not
  reproducible from the committed preprocessing cells.
- `test subset of val`, not the reverse.
- Accuracy claim is grounded in the 100-row intrinsic summary, not only the 10 displayed table rows.
- No subject-level ordering claim; only the session-block straddle is provable.

## Outcome

Retained `seeded-session-grouped-disjoint-split` as the single improvement. It is local to cell 37,
needs no new data, passes the structural contract on all tested inputs, and is the precondition for
any meaningful re-evaluation. No accuracy was measured and nothing was executed.
