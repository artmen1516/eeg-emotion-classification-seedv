#!/usr/bin/env python3
"""Throwaway experiment for hypothesis-test round 1.

Reproduces, WITHOUT executing the notebook, the committed split arithmetic of
cell 37 of process_seedv_raw_files_with_mne_from_drive.ipynb.

Run from the repository root:
    python3 research/test-val-overlap-b9ead542/experiment_index_arithmetic.py

Writes experiment_results.json next to this file.
"""
from __future__ import annotations

import json
import math
import platform
import re
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
NB_PATH = REPO / "process_seedv_raw_files_with_mne_from_drive.ipynb"
OUT = Path(__file__).resolve().parent / "experiment_results.json"

# ---------------------------------------------------------------- E1
nb = json.loads(NB_PATH.read_text())
cell = nb["cells"][37]
assert cell["cell_type"] == "code", "cell 37 is not a code cell"
src = "".join(cell["source"])
lines = src.splitlines()


src35_lines = "".join(nb["cells"][35]["source"]).splitlines()


def committed_line(pattern: str, pool=None) -> str:
    """Return the single committed line matching pattern (verbatim)."""
    pool = lines if pool is None else pool
    hits = [ln for ln in pool if re.search(pattern, ln)]
    assert len(hits) == 1, f"expected exactly one line for {pattern!r}, got {hits}"
    return hits[0]


e1 = {
    "cell_index_0based": 37,
    "cell_title": [ln for ln in lines if ln.startswith("#@title")][0],
    "cell_magic_first_line": lines[0],
    "committed_outputs_present": bool(cell.get("outputs")),
    "notebook_head_sha": subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip(),
    "notebook_blob_sha": subprocess.check_output(
        ["git", "-C", str(REPO), "hash-object", str(NB_PATH)], text=True
    ).strip(),
    "verbatim": {
        "training_percent": committed_line(r"^training_percent\s*=", src35_lines),
        "total_array_samples": committed_line(r"^total_array_samples\s*="),
        "train_samples": committed_line(r"^train_samples\s*="),
        "trainList": committed_line(r"^trainList\s*="),
        "testList": committed_line(r"^testList\s*="),
        "val_percent": committed_line(r"^val_percent\s*="),
        "split_call": committed_line(r"^splits\s*=.*windows_dataset\.split"),
        "train_set_reassign": committed_line(r"^train_set\s*=\s*windows_dataset"),
        "x_train": committed_line(r"^X_train\s*=\s*SliceDataset"),
        "y_train_set": committed_line(r"^y_train_set\s*="),
        "y_test_set": committed_line(r"^y_test_set\s*="),
        "train_test_split": committed_line(r"^train_indices,\s*val_indices"),
        "subsets": committed_line(r"^val_subset\s*="),
    },
    "training_percent_value": 0.8,  # cell 35 slider literal
    "training_percent_cell35": committed_line(r"^training_percent\s*=", src35_lines),
}

# ---------------------------------------------------------------- E2
training_percent = 0.8
N = 576
train_samples = math.floor(training_percent * N)
trainList = list(range(1, train_samples + 1, 1))
testList = list(range(train_samples + 1, N, 1))
val_percent = 1 - training_percent


def overlap(test, val):
    t, v = set(test), set(val)
    inter = sorted(t & v)
    ratio = Fraction(len(inter), len(t)) if t else None
    return {
        "n_test": len(t),
        "n_val": len(v),
        "n_intersection": len(inter),
        "ratio_exact": None if ratio is None else f"{ratio.numerator}/{ratio.denominator}",
        "ratio_float": None if ratio is None else float(ratio),
        "test_first_last": [t and min(t), t and max(t)],
        "val_first_last": [v and min(v), v and max(v)],
        "intersection_first_last": [inter[0], inter[-1]] if inter else None,
        "test_minus_val_first10": sorted(t - v)[:10],
        "val_minus_test_first10": sorted(v - t)[:10],
    }


# Committed code: train_set = windows_dataset; X_train = SliceDataset(train_set, idx=0).
# skorch's SliceDataset sets indices_ = np.arange(len(dataset)) when indices is None,
# so under the stipulated N = 576 the validation base is the full positional range.
X_train_indices_ = np.arange(N)
tr_idx, val_idx = train_test_split(
    X_train_indices_, test_size=val_percent, shuffle=False
)

e2 = {
    "N": N,
    "training_percent": training_percent,
    "train_samples_floor": train_samples,
    "val_percent_repr": repr(val_percent),
    "val_percent_exact": str(Fraction(val_percent)),
    "trainList_range": [trainList[0], trainList[-1], len(trainList)],
    "testList_range": [testList[0], testList[-1], len(testList)],
    "base_size": len(X_train_indices_),
    "sklearn_n_train": len(tr_idx),
    "sklearn_n_val": len(val_idx),
    "sklearn_train_first_last": [int(tr_idx[0]), int(tr_idx[-1])],
    "sklearn_val_first_last": [int(val_idx[0]), int(val_idx[-1])],
    "ceil_rule_check": {
        "float_test_size_times_N": val_percent * N,
        "ceil": math.ceil(val_percent * N),
        "floor": math.floor(val_percent * N),
    },
    **overlap(testList, val_idx.tolist()),
}

# E2b: same arithmetic under the floor/ceil alternatives for n_test (one-index robustness)
alt = {}
for name, ntest in {
    "ceil": math.ceil(val_percent * N),
    "floor": math.floor(val_percent * N),
    "round": round(val_percent * N),
}.items():
    v = list(range(N - ntest, N))
    alt[name] = {"n_val": len(v), "val_first_last": [v[0], v[-1]], **overlap(testList, v)}
e2["n_test_variants"] = alt

# E2c: sensitivity of the ratio to the stipulated N (threat: N is data-dependent).
sweep = {}
for n in list(range(2, 2001)) + [2160, 720, 600, 580, 570, 560, 540]:
    ts = math.floor(training_percent * n)
    tl = range(ts + 1, n)
    _, vi = train_test_split(np.arange(n), test_size=val_percent, shuffle=False)
    o = overlap(list(tl), vi.tolist())
    sweep[n] = (o["ratio_exact"], o["n_test"], o["n_intersection"])
ratios = {r for r, _, _ in sweep.values() if r is not None}
e2["N_sensitivity"] = {
    "N_values_tested": len(sweep),
    "distinct_ratios": sorted(ratios),
    "all_ratio_equal_1": ratios == {"1/1"},
    "N_576": sweep[576],
    "N_540": sweep[540],
    "N_560": sweep[560],
    "N_600": sweep[600],
    "N_720": sweep[720],
    "N_2160": sweep[2160],
}

# ---------------------------------------------------------------- E3
# Counterfactual base = trainList (1..460): what a reader who assumes X_train is the
# training split would compute.
_, val_trainList = train_test_split(
    np.arange(1, train_samples + 1, 1), test_size=val_percent, shuffle=False
)
e3_raw = {
    "base": "np.arange(1, 461)  # trainList",
    "base_size": train_samples,
    "sklearn_n_train": len(np.arange(1, train_samples + 1, 1)) - len(val_trainList),
    "sklearn_n_val": len(val_trainList),
    "sklearn_val_first_last": [int(val_trainList[0]), int(val_trainList[-1])],
    **overlap(testList, val_trainList.tolist()),
}
# Variant base 0..459 (a 460-element train subset indexed from 0)
_, val_0based = train_test_split(
    np.arange(0, train_samples, 1), test_size=val_percent, shuffle=False
)
e3 = {
    "primary_counterfactual": e3_raw,
    "zero_based_variant": {
        "base": "np.arange(0, 460)",
        "sklearn_n_val": len(val_0based),
        "sklearn_val_first_last": [int(val_0based[0]), int(val_0based[-1])],
        **overlap(testList, val_0based.tolist()),
    },
    "discrepancy_primary_minus_counterfactual": {
        "E2_ratio": e2["ratio_exact"],
        "E3_ratio": e3_raw["ratio_exact"],
        "absolute_difference": float(
            Fraction(e2["ratio_exact"]) - Fraction(e3_raw["ratio_exact"])
        ),
    },
}

# ---------------------------------------------------------------- E4
set_train, set_test = set(trainList), set(testList)
e4 = {
    "a_partition": {
        "quote_trainList": e1["verbatim"]["trainList"],
        "quote_testList": e1["verbatim"]["testList"],
        "len_train": len(set_train),
        "len_test": len(set_test),
        "intersection_train_test": len(set_train & set_test),
        "union_size": len(set_train | set_test),
        "missing_indices": sorted(set(range(N)) - (set_train | set_test)),
        "finding": "trainList and testList are disjoint and together cover 1..575; "
        "index 0 is never selected. 575 of the stipulated 576 positional slots are used.",
    },
    "b_length_consistency": {
        "quote_testList": e1["verbatim"]["testList"],
        "quote_y_test_set": e1["verbatim"]["y_test_set"],
        "len_testList": len(testList),
        "len_y_test_set_under_N_576": len(range(train_samples + 1, N)),
        "match": len(testList) == len(range(train_samples + 1, N)),
        "finding": "test_set and y_test_set both have 115 elements under N = 576; "
        "the count matches even though the objects index different spaces.",
    },
    "c_train_label_alignment": {
        "quote_trainList": e1["verbatim"]["trainList"],
        "quote_y_train_set": e1["verbatim"]["y_train_set"],
        "trainList_first_last": [trainList[0], trainList[-1]],
        "y_train_set_slice": "y[0:460] -> labels for positional samples 0..459",
        "len_match": len(trainList) == train_samples,
        "finding": "lengths match (460) but the offsets do not: window indices 1..460 are "
        "paired with labels 0..459, so every training label is shifted by one sample "
        "relative to its window; the training block is internally inconsistent.",
    },
    "d_slice_dataset_indices": {
        "quote_x_train": e1["verbatim"]["x_train"],
        "quote_train_set_reassign": e1["verbatim"]["train_set_reassign"],
        "notebook_import": "from skorch.helper import predefined_split, SliceDataset  (cell 35)",
        "upstream_source": "skorch/helper.py, SliceDataset.__init__",
        "upstream_quote": "self.indices_ = (self.indices if self.indices is not None "
        "else np.arange(len(self.dataset)))",
        "attribute_exists": True,
        "base_identity": "X_train.indices_ == np.arange(len(windows_dataset))",
        "finding": "SliceDataset does expose indices_ (its base class is not torch Subset but "
        "a Sequence that materialises arange(len(dataset))). The committed line does not "
        "raise AttributeError, and the validation split is taken over the FULL "
        "windows_dataset, not over trainList, because train_set was reassigned from "
        "splits['train'] to windows_dataset two lines earlier.",
    },
    "e_what_is_counted": {
        "quote_y": committed_line(r"^y\s*=\s*\[epoch\.metadata"),
        "quote_total": e1["verbatim"]["total_array_samples"],
        "quote_window_creation": [
            ln for ln in "".join(nb["cells"][33]["source"]).splitlines()
            if re.search(r"trial_window_(time|stride_time)\s*=|drop_last_window", ln)
        ],
        "quote_score_filter": [
            ln for ln in "".join(nb["cells"][30]["source"]).splitlines()
            if "score" in ln and "list_of_epochs" in ln
        ],
        "annotations_per_session_file": 15,
        "session_definitions": 3,
        "labeled_files_expected": "16 subjects x 3 sessions = 48",
        "max_epochs_before_filter": 15 * 3 * 16,
        "finding": "total_array_samples = len(y) where y concatenates "
        "epoch.metadata['target'] over list_of_epochs, i.e. it counts SCORE-FILTERED "
        "EPOCHS (trials), not 50 s windows. windows_dataset (cell 33, 50 s window, 0.01 s "
        "stride, drop_last_window=True) has one row per sliding window, which is far more "
        "than len(y). So the split lists derived from the trial count are applied to a "
        "window-indexed dataset. N = 576 is a plausible filtered-trial count "
        "(<= 15*3*16 = 720) but is not recorded anywhere in the committed notebook: "
        "cell 37 is %%capture, so no stored output confirms it.",
    },
}

results = {
    "environment": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    },
    "E1_committed_constants": e1,
    "E2_primary": e2,
    "E3_base_sensitivity": e3,
    "E4_static_audit": e4,
}

OUT.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({k: results[k] for k in ("environment",)}, indent=2))
print("E2 ratio:", e2["ratio_exact"], "|test|:", e2["n_test"], "|val|:", e2["n_val"],
      "intersection:", e2["n_intersection"], "val:", e2["sklearn_val_first_last"])
print("E2b n_test variants:", {k: v["ratio_exact"] for k, v in e2["n_test_variants"].items()})
print("E3 ratio:", e3_raw["ratio_exact"], "val:", e3_raw["sklearn_val_first_last"])
print("N-sweep all==1:", e2["N_sensitivity"]["all_ratio_equal_1"],
      "tested:", e2["N_sensitivity"]["N_values_tested"])
print("wrote", OUT)
