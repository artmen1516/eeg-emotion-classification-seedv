"""Seeded, whole-session, pairwise-disjoint train/validation/test splits.

Why this module exists
----------------------
The committed notebook ``process_seedv_raw_files_with_mne_from_drive.ipynb``
(cell 37) builds an evaluation split that has no held-out test set.  With the
recorded shape of 48 sessions x 12 trials (576 trials, cell 32) and
``training_percent = 0.8`` it computes

    train = 0..459        (460 indices)
    val   = 460..575      (116 indices, sklearn shuffle=False test_size=0.2)
    test  = 461..575      (115 indices)

so ``test`` is a subset of ``val``: the overlap is 115/115, i.e. 100% of the
test set, and the only validation-only index is 460.  Because the session
block size is 12 and ``460 % 12 == 4``, the block 456..467 also straddles the
train/validation boundary.  The notebook is deliberately **not** modified by
this change; this module is the extractable, independently tested kernel of the
repair retained by ``research/evaluation-split-leakage-7e59d8a9``.

The contract
------------
``split_indices(n_sessions, trials_per_session, train_ratio=0.6,
val_ratio=0.2, test_ratio=0.2, seed=42)`` returns three ascending lists of
integer trial indices.  Session ``s`` owns the contiguous block
``[s * trials_per_session, (s + 1) * trials_per_session)``.  Whole sessions are
shuffled with ``random.Random(seed)`` and assigned to partitions, so:

* the three index lists are pairwise disjoint and their union is exactly
  ``range(n_sessions * trials_per_session)``;
* no session straddles two partitions;
* the result is reproducible for a fixed seed and independent of the global
  :mod:`random` state and of ``PYTHONHASHSEED``.

"Raising on overlapping sessions"
---------------------------------
The required signature takes only counts, so sessions are implicit
(``index // trials_per_session``) and cannot overlap at call time.  The
requirement is therefore implemented two ways:

* :func:`split_indices` runs a fail-fast structural guard on its own output and
  raises ``RuntimeError`` (not ``assert``, which ``python -O`` strips) if any
  index is duplicated, missing, or if a session lands in two partitions -- i.e.
  if overlapping sessions were ever produced;
* :func:`validate_partition` exposes the same check publicly and raises
  ``ValueError`` for a caller-supplied partition that has overlapping indices,
  missing/extra indices, or a session that straddles partitions.

Honest limitations
------------------
* Whole-session disjointness is *not* subject disjointness.  Under SEED-V's
  16 subjects x 3 sessions layout (external fact, not in this repository) a
  session-level split can still share subjects across train and test.
* The rectangular ``(n_sessions, trials_per_session)`` model cannot represent
  the 7-15-trials-per-session layout implied by the notebook's cell 24; it
  mirrors the 48 x 12 layout recorded in cell 32.
* The default ``0.6/0.2/0.2`` intentionally differs from the notebook's
  user-visible ``training_percent = 0.8`` (460 vs 348 training trials at the
  recorded shape).  The ratio is a parameter; the default is the retained
  60/20/20 target, not a silent claim that cell 37 was repaired.
* Exact ratios are usually unreachable at session granularity: the realised
  split deviates from the requested ratio by at most one session, because
  every partition must contain a whole number of sessions.
"""

from __future__ import annotations

import math
import random

__all__ = ["split_indices", "validate_partition"]

#: Names used only in error messages / quota tie-breaking order.
_PARTITIONS = ("train", "val", "test")
#: Absolute tolerance for "the ratios sum to one".
_RATIO_SUM_TOL = 1e-9


def _positive_int(value, name):
    """Return *value* as an ``int`` or raise ``ValueError``.

    ``bool`` is rejected explicitly even though ``isinstance(True, int)`` is
    true, so ``n_sessions=True`` cannot silently mean one session.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be a positive integer, got {value!r} "
            f"({type(value).__name__})"
        )
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validated_ratio(value, name):
    """Return *value* as a ``float`` in ``(0, 1)`` or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a real number in (0, 1), got {value!r} "
            f"({type(value).__name__})"
        )
    ratio = float(value)
    if not math.isfinite(ratio):
        # NaN/Inf slip through naive tolerance checks: abs(nan - 1) > tol is
        # False, so reject them before any sum comparison.
        raise ValueError(f"{name} must be finite, got {value!r}")
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"{name} must be strictly between 0 and 1, got {value!r}"
        )
    return ratio


def _session_quotas(n_sessions, ratios):
    """Allocate whole sessions to train/val/test by largest remainder.

    Raw targets are ``ratio * n_sessions``; each partition gets the floor and
    the leftover sessions go to the largest fractional parts (ties broken by
    partition order: train, then val, then test).  A repair pass then gives
    every partition at least one session, stealing from the largest partition,
    so a three-way split is always feasible once ``n_sessions >= 3``.
    """
    raw = [ratio * n_sessions for ratio in ratios]
    quotas = [math.floor(value) for value in raw]
    leftover = n_sessions - sum(quotas)

    order = sorted(range(len(ratios)), key=lambda i: (-(raw[i] - quotas[i]), i))
    for position in range(leftover):
        quotas[order[position % len(order)]] += 1

    # Repair: no empty partition when three whole-session partitions exist.
    while min(quotas) == 0:
        empty = quotas.index(0)
        donor = max(range(len(quotas)), key=lambda i: (quotas[i], -i))
        if quotas[donor] < 2:  # pragma: no cover - unreachable for n_sessions >= 3
            raise ValueError(
                "cannot give every partition at least one whole session with "
                f"n_sessions={n_sessions}"
            )
        quotas[donor] -= 1
        quotas[empty] += 1

    return quotas


def _check_structure(train, val, test, n_sessions, trials_per_session, exc_type):
    """Raise *exc_type* unless the partition is disjoint, complete, whole-session."""
    n_trials = n_sessions * trials_per_session
    partitions = (train, val, test)
    names = _PARTITIONS

    for name, indices in zip(names, partitions):
        if len(indices) == 0:
            # An empty partition is exactly the failure being repaired: a
            # "test" set with no trials is not a held-out test set.
            raise exc_type(
                f"{name} is empty; a three-way split must have non-empty "
                "partitions"
            )

    seen = {}
    for name, indices in zip(names, partitions):
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise exc_type(
                    f"{name} contains a non-integer index {index!r} "
                    f"({type(index).__name__})"
                )
            if index < 0 or index >= n_trials:
                raise exc_type(
                    f"{name} contains out-of-range index {index} "
                    f"(expected 0 <= index < {n_trials})"
                )
            if index in seen:
                raise exc_type(
                    f"index {index} appears in both {seen[index]} and {name}: "
                    "the partitions must be pairwise disjoint"
                )
            seen[index] = name

    missing = [index for index in range(n_trials) if index not in seen]
    if missing:
        preview = ", ".join(str(index) for index in missing[:5])
        raise exc_type(
            f"the partitions miss {len(missing)} index/indices (e.g. {preview}); "
            f"the union must cover range({n_trials}) exactly once"
        )

    # Whole-session integrity: a session's block may not straddle partitions.
    session_partition = {}
    for name, indices in zip(names, partitions):
        for index in indices:
            session = index // trials_per_session
            previous = session_partition.setdefault(session, name)
            if previous != name:
                raise exc_type(
                    f"session {session} straddles two partitions "
                    f"({previous} and {name}); whole sessions must not be split"
                )


def validate_partition(train, val, test, n_sessions, trials_per_session):
    """Validate a train/val/test partition; raise ``ValueError`` if invalid.

    Checks that ``train``, ``val`` and ``test`` are pairwise disjoint, that
    their union is exactly ``range(n_sessions * trials_per_session)``, and
    that every contiguous session block of ``trials_per_session`` indices lies
    entirely inside one partition.

    This is how "raising on overlapping sessions" is made observable from
    outside: it rejects a caller-supplied partition in which a whole session
    (or any single index) appears in more than one partition.
    """
    n_sessions = _positive_int(n_sessions, "n_sessions")
    trials_per_session = _positive_int(
        trials_per_session, "trials_per_session"
    )
    _check_structure(
        train, val, test, n_sessions, trials_per_session, ValueError
    )


def split_indices(
    n_sessions,
    trials_per_session,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=42,
):
    """Return a seeded, whole-session, pairwise-disjoint index partition.

    Parameters
    ----------
    n_sessions : int
        Number of contiguous sessions (``>= 3`` for a three-way split).
    trials_per_session : int
        Trials in each session block (``>= 1``).
    train_ratio, val_ratio, test_ratio : float
        Strictly positive fractions summing to 1 within ``1e-9``.
    seed : int
        Seed for a local ``random.Random``; never touches global RNG state.

    Returns
    -------
    (train, val, test) : tuple of list of int
        Ascending index lists, pairwise disjoint, covering
        ``range(n_sessions * trials_per_session)`` exactly once, with no
        session block straddling two partitions.

    Raises
    ------
    ValueError
        For non-integer/bool counts, ``n_sessions < 3``,
        ``trials_per_session < 1``, non-finite or non-positive ratios, or
        ratios whose sum is not 1 within tolerance.
    RuntimeError
        If the constructed partition ever violates pairwise disjointness,
        exact coverage, or whole-session integrity (a fail-fast guard that
        survives ``python -O``).
    """
    n_sessions = _positive_int(n_sessions, "n_sessions")
    trials_per_session = _positive_int(
        trials_per_session, "trials_per_session"
    )
    if n_sessions < 3:
        raise ValueError(
            "n_sessions must be >= 3 so that train, val and test can each "
            f"receive at least one whole session, got {n_sessions}"
        )

    ratios = [
        _validated_ratio(train_ratio, "train_ratio"),
        _validated_ratio(val_ratio, "val_ratio"),
        _validated_ratio(test_ratio, "test_ratio"),
    ]
    total = math.fsum(ratios)
    if abs(total - 1.0) > _RATIO_SUM_TOL:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1 "
            f"(tolerance {_RATIO_SUM_TOL}), got {total!r}"
        )

    quotas = _session_quotas(n_sessions, ratios)

    order = list(range(n_sessions))
    random.Random(seed).shuffle(order)

    assigned = ([], [], [])
    cursor = 0
    for partition, quota in enumerate(quotas):
        sessions = sorted(order[cursor:cursor + quota])
        cursor += quota
        indices = assigned[partition]
        for session in sessions:
            start = session * trials_per_session
            indices.extend(range(start, start + trials_per_session))
        indices.sort()

    train, val, test = assigned
    _check_structure(
        train, val, test, n_sessions, trials_per_session, RuntimeError
    )
    return train, val, test
