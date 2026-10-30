"""Correctness tests for :mod:`seedv_split`.

Run with ``python -m pytest -q`` or ``pytest -q`` from the repository root, or
directly with ``python tests/test_seedv_split.py`` (the file bootstraps the
repository root on ``sys.path`` because pytest's default prepend import mode
does not add it for tests living under ``tests/``).
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

# Make ``import seedv_split`` work regardless of the pytest invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import seedv_split  # noqa: E402
from seedv_split import split_indices, validate_partition  # noqa: E402

N_SESSIONS = 48
TRIALS = 12
N_TRIALS = N_SESSIONS * TRIALS


def _sessions(indices, trials_per_session=TRIALS):
    """Return the set of whole sessions touched by *indices*."""
    return {index // trials_per_session for index in indices}


def test_default_shape_reproduces_the_recorded_replacement():
    """48 sessions x 12 trials, seed 42 -> the audit's 348/120/108 split."""
    train, val, test = split_indices(N_SESSIONS, TRIALS)

    assert (len(train), len(val), len(test)) == (348, 120, 108)
    assert (len(train) // TRIALS, len(val) // TRIALS, len(test) // TRIALS) == (
        29,
        10,
        9,
    )
    assert train == sorted(train)
    assert val == sorted(val)
    assert test == sorted(test)


def test_partitions_are_disjoint_and_cover_every_index_exactly_once():
    train, val, test = split_indices(N_SESSIONS, TRIALS)

    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert set(train) | set(val) | set(test) == set(range(N_TRIALS))
    assert len(train) + len(val) + len(test) == N_TRIALS


def test_whole_sessions_do_not_straddle_partitions():
    train, val, test = split_indices(N_SESSIONS, TRIALS)

    train_sessions = _sessions(train)
    val_sessions = _sessions(val)
    test_sessions = _sessions(test)

    assert not (train_sessions & val_sessions)
    assert not (train_sessions & test_sessions)
    assert not (val_sessions & test_sessions)
    assert train_sessions | val_sessions | test_sessions == set(
        range(N_SESSIONS)
    )
    # Each partition is exactly a union of whole blocks.
    for indices in (train, val, test):
        for session in _sessions(indices):
            block = set(
                range(session * TRIALS, (session + 1) * TRIALS)
            )
            assert block <= set(indices)


def test_structural_contract_holds_across_a_grid():
    for trials in (1, 2, 12):
        for n_sessions in range(3, 41):
            train, val, test = split_indices(n_sessions, trials)
            n_trials = n_sessions * trials
            assert sorted(train + val + test) == list(range(n_trials))
            assert len(set(train + val + test)) == n_trials
            assert not (set(train) & set(val))
            assert not (set(train) & set(test))
            assert not (set(val) & set(test))
            for indices in (train, val, test):
                for session in _sessions(indices, trials):
                    block = range(session * trials, (session + 1) * trials)
                    assert set(block) <= set(indices)


def test_every_partition_is_non_empty_for_three_or_more_sessions():
    for n_sessions in range(3, 41):
        train, val, test = split_indices(n_sessions, 1)
        assert train and val and test


def test_realised_sizes_stay_within_one_session_of_the_target():
    # Whole-session allocation cannot hit 60/20/20 exactly; the deviation is
    # bounded by one session for every feasible input.
    for n_sessions in range(3, 41):
        train, val, test = split_indices(n_sessions, 10)
        for indices, ratio in (
            (train, 0.6),
            (val, 0.2),
            (test, 0.2),
        ):
            realised = len(indices) / (n_sessions * 10)
            assert abs(realised - ratio) <= 1 / n_sessions + 1e-12


def test_same_seed_is_reproducible_and_order_is_stable():
    first = split_indices(N_SESSIONS, TRIALS, seed=42)
    second = split_indices(N_SESSIONS, TRIALS, seed=42)
    assert first == second
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_different_seed_changes_the_partition():
    base = split_indices(N_SESSIONS, TRIALS, seed=42)
    other = split_indices(N_SESSIONS, TRIALS, seed=7)
    assert base != other
    # Whole-session counts need not change, but the membership must.
    assert set(base[2]) != set(other[2])


def test_global_random_state_is_untouched():
    random.seed(1234)
    before = random.getstate()
    split_indices(N_SESSIONS, TRIALS, seed=42)
    assert random.getstate() == before


def test_interleaved_seed_does_not_perturb_the_result():
    first = split_indices(N_SESSIONS, TRIALS, seed=42)
    split_indices(N_SESSIONS, TRIALS, seed=7)
    second = split_indices(N_SESSIONS, TRIALS, seed=42)
    assert first == second


def test_seed_accepts_positional_and_keyword_forms():
    assert split_indices(48, 12, 0.6, 0.2, 0.2, 42) == split_indices(
        48, 12, seed=42
    )
    assert split_indices(48, 12, 0.6, 0.2, 0.2, 42) == split_indices(
        48, 12, 0.6, 0.2, 0.2, seed=42
    )


@pytest.mark.parametrize(
    "ratios",
    [
        (0.7, 0.2, 0.1),
        (0.3, 0.35, 0.35),
        (0.6, 0.3, 0.1),
        (0.2, 0.7, 0.1),
    ],
)
def test_decimal_ratio_triples_that_sum_to_one_in_decimal_are_accepted(ratios):
    # Some of these float sums are 0.9999999999999999; an exact == 1.0 check
    # would wrongly reject them.
    assert math.isclose(math.fsum(ratios), 1.0, abs_tol=1e-9)
    train, val, test = split_indices(30, 4, *ratios)
    assert len(train) + len(val) + len(test) == 30 * 4


@pytest.mark.parametrize(
    "ratios",
    [
        (0.6, 0.2, 0.3),          # sum 1.1
        (0.5, 0.2, 0.2),          # sum 0.9
        (0.8, 0.3, -0.1),         # negative but sums to 1
        (1.2, -0.1, -0.1),        # negative but sums to 1
        (1.0, 0.0, 0.0),          # not strictly inside (0, 1)
        (0.0, 0.5, 0.5),          # an empty partition is the defect
        (float("nan"), 0.2, 0.8),  # NaN passes a naive tolerance check
        (float("inf"), 0.0, 0.0),  # non-finite
    ],
)
def test_impossible_ratios_raise_value_error(ratios):
    with pytest.raises(ValueError):
        split_indices(30, 4, *ratios)


@pytest.mark.parametrize("n_sessions", [0, 1, 2, -1])
def test_too_few_sessions_raise_value_error(n_sessions):
    with pytest.raises(ValueError):
        split_indices(n_sessions, 12)


def test_bool_and_float_counts_are_rejected():
    with pytest.raises(ValueError):
        split_indices(True, 12)
    with pytest.raises(ValueError):
        split_indices(4.0, 12)
    with pytest.raises(ValueError):
        split_indices(4, True)
    with pytest.raises(ValueError):
        split_indices(4, 12.0)
    with pytest.raises(ValueError):
        split_indices(4, 12, True, 0.4, 0.6)


@pytest.mark.parametrize("trials", [0, -1])
def test_non_positive_trials_raise_value_error(trials):
    with pytest.raises(ValueError):
        split_indices(10, trials)


def test_validate_partition_accepts_the_generated_split():
    assert validate_partition(
        *split_indices(N_SESSIONS, TRIALS), N_SESSIONS, TRIALS
    ) is None


def test_validate_partition_rejects_a_duplicated_index():
    train, val, test = split_indices(10, 4)
    broken_val = val + [train[0]]
    with pytest.raises(ValueError):
        validate_partition(train, broken_val, test, 10, 4)


def test_validate_partition_rejects_a_missing_index():
    train, val, test = split_indices(10, 4)
    with pytest.raises(ValueError):
        validate_partition(train, val, test[:-1], 10, 4)


def test_validate_partition_rejects_a_session_straddling_partitions():
    train, val, test = split_indices(10, 4)
    # Find a whole session in train and move its last index into val.
    session_zero = min(train) // 4
    block = list(range(session_zero * 4, (session_zero + 1) * 4))
    assert set(block) <= set(train)
    broken_train = [i for i in train if i != block[-1]]
    broken_val = val + [block[-1]]
    with pytest.raises(ValueError):
        validate_partition(broken_train, broken_val, test, 10, 4)


def test_validate_partition_rejects_an_out_of_range_index():
    train, val, test = split_indices(10, 4)
    with pytest.raises(ValueError):
        validate_partition(train[:-1], val, test + [40], 10, 4)


def test_module_exposes_the_documented_api():
    assert callable(seedv_split.split_indices)
    assert callable(seedv_split.validate_partition)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    raise SystemExit(pytest.main([__file__, "-q"]))
