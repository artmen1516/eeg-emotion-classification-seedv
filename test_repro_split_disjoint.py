"""Regression test: the evaluation split must have a held-out test set.

The parent construction is a *static* re-derivation of
``process_seedv_raw_files_with_mne_from_drive.ipynb`` cell 37 (it is never
executed here).  At the recorded 48 sessions x 12 trials = 576 trials and
``training_percent = 0.8`` cell 37 computes::

    train = 0..459        (460)
    val   = 460..575      (116, sklearn shuffle=False, test_size=0.2)
    test  = 461..575      (115)

so ``test`` is entirely inside ``val`` (115/115 overlap) and the session block
456..467 straddles train/validation (``460 % 12 == 4``).

This test fails if the split regresses to that shape: it checks pairwise
disjointness, exact-once coverage, whole-session containment, and that the new
split is not a subset relation like the parent's.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# The file lives at the repository root; keep the import independent of how
# pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seedv_split import split_indices, validate_partition  # noqa: E402

N_SESSIONS = 48
TRIALS_PER_SESSION = 12
N_TRIALS = N_SESSIONS * TRIALS_PER_SESSION
TRAINING_PERCENT = 0.8


def _cell37_parent_split(n_trials, training_percent=TRAINING_PERCENT):
    """Re-derive cell 37's split from committed literals (no sklearn, no run).

    ``train_samples = floor(training_percent * N)``;
    ``testList = range(train_samples + 1, N)``;
    ``train_test_split(range(N), test_size=1 - training_percent, shuffle=False)``
    puts the last ``ceil((1 - training_percent) * N)`` indices in validation.
    """
    train_samples = math.floor(training_percent * n_trials)
    train = list(range(0, train_samples))
    test = list(range(train_samples + 1, n_trials))
    n_val = math.ceil((1.0 - training_percent) * n_trials)
    val = list(range(n_trials - n_val, n_trials))
    return train, val, test


def test_parent_cell37_split_has_no_held_out_test_set():
    """The historical defect: test is a subset of validation."""
    train, val, test = _cell37_parent_split(N_TRIALS)

    assert (len(train), len(val), len(test)) == (460, 116, 115)
    assert set(test) <= set(val)
    assert set(test) - set(val) == set()
    assert set(val) - set(test) == {460}
    assert len(set(test) & set(val)) == 115

    # The block 456..467 straddles train/validation.
    assert 460 % TRIALS_PER_SESSION == 4
    block = set(range(456, 468))
    assert block & set(train) and block & set(val)

    # The validator rejects the parent partition outright.
    with pytest.raises(ValueError):
        validate_partition(train, val, test, N_SESSIONS, TRIALS_PER_SESSION)


def test_repro_split_disjoint_covers_every_index_once():
    """The repair: three pairwise-disjoint sets covering 0..575 exactly once."""
    train, val, test = split_indices(N_SESSIONS, TRIALS_PER_SESSION)

    assert (len(train), len(val), len(test)) == (348, 120, 108)

    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert sorted(train + val + test) == list(range(N_TRIALS))
    assert len(train) + len(val) + len(test) == N_TRIALS


def test_repro_split_keeps_whole_sessions_together():
    """No 12-index session block may straddle two partitions."""
    train, val, test = split_indices(N_SESSIONS, TRIALS_PER_SESSION)

    partition_of_session = {}
    for name, indices in (("train", train), ("val", val), ("test", test)):
        assert len(indices) % TRIALS_PER_SESSION == 0
        for index in indices:
            session = index // TRIALS_PER_SESSION
            assert partition_of_session.setdefault(session, name) == name
            block = range(
                session * TRIALS_PER_SESSION,
                (session + 1) * TRIALS_PER_SESSION,
            )
            assert set(block) <= set(indices)

    assert len(partition_of_session) == N_SESSIONS
    assert len(train) // TRIALS_PER_SESSION == 29
    assert len(val) // TRIALS_PER_SESSION == 10
    assert len(test) // TRIALS_PER_SESSION == 9

    # The public validator agrees.
    assert validate_partition(
        train, val, test, N_SESSIONS, TRIALS_PER_SESSION
    ) is None


def test_repro_split_is_not_the_leaky_parent_and_is_seeded():
    """Mutation guard plus reproducibility: the fix cannot silently revert."""
    train, val, test = split_indices(N_SESSIONS, TRIALS_PER_SESSION)
    parent_train, parent_val, parent_test = _cell37_parent_split(N_TRIALS)

    # The defect pattern must be absent from the replacement.
    assert not set(test) <= set(val)
    assert (len(train), len(val), len(test)) != (
        len(parent_train),
        len(parent_val),
        len(parent_test),
    )

    # Same seed -> identical lists; another seed -> different membership.
    assert split_indices(N_SESSIONS, TRIALS_PER_SESSION, seed=42) == (
        train,
        val,
        test,
    )
    other = split_indices(N_SESSIONS, TRIALS_PER_SESSION, seed=7)
    assert other != (train, val, test)
    assert set(other[2]) != set(test)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    raise SystemExit(pytest.main([__file__, "-q"]))
