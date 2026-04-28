"""Functional contracts for step17 `_acq_prob_at_step` dispatch.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

These tests pin the input/output behavior of the per-step acquisition-
probability dispatch. They are functional tests, not branch-coverage
exercises: each test asserts what the function should return for a
given input, not just that the code path was executed.

Two contracts pinned here:

1. **Dispatch correctness** (task #2 in 2026-04-27 cleanup): each named
   acquisition mode returns the documented per-step probability;
   unknown modes raise ValueError. The pre-task-#2 code used an
   `if` chain with an unconditional fall-through to `valence_inverted`
   for any mode it didn't explicitly match — so a typo silently ran
   the wrong branch. The dispatch-table refactor + ValueError fixes
   that, and these tests pin it.

2. **Periodic step=0 gate** (task #19 in 2026-04-27 cleanup): the
   `periodic` mode used to fire at step=0 because `step %
   PERIODIC_INTERVAL_STEPS == 0` is True at step=0. The initial pool
   is fully active so no actual acquisition happened, but the trigger
   was a spurious RNG-draw + Bernoulli computation. The fix added a
   `step > 0` guard. These tests pin the gate so a future refactor
   can't reintroduce the spurious step-0 fire.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

# pylint: disable=wrong-import-position,import-error
import step17_structural_growth as s17  # noqa: E402

# pylint: disable=protected-access


# Functional contract A: dispatch correctness for each named mode.

def test_acq_prob_off_returns_zero() -> None:
    """`off` mode returns 0.0 regardless of step or valence."""
    result = s17._acq_prob_at_step(
        "off", jnp.int32(100), jnp.float32(0.7),
    )
    assert float(result) == 0.0


def test_acq_prob_stochastic_returns_constant() -> None:
    """`stochastic` returns ACQ_PROB_STOCHASTIC regardless of step/valence."""
    result = s17._acq_prob_at_step(
        "stochastic", jnp.int32(100), jnp.float32(0.7),
    )
    # Tolerance because float32 can't exactly represent 0.001.
    assert abs(float(result) - s17.ACQ_PROB_STOCHASTIC) < 1e-7


def test_acq_prob_valence_gated_scales_with_valence() -> None:
    """`valence_gated` returns ACQ_PROB_VALENCE_MAX * valence (exploit)."""
    valence = 0.6
    result = s17._acq_prob_at_step(
        "valence_gated", jnp.int32(100), jnp.float32(valence),
    )
    expected = s17.ACQ_PROB_VALENCE_MAX * valence
    assert abs(float(result) - expected) < 1e-7


def test_acq_prob_valence_inverted_scales_with_inverse_valence() -> None:
    """`valence_inverted` returns ACQ_PROB_VALENCE_MAX * (1-valence)."""
    valence = 0.6
    result = s17._acq_prob_at_step(
        "valence_inverted", jnp.int32(100), jnp.float32(valence),
    )
    expected = s17.ACQ_PROB_VALENCE_MAX * (1.0 - valence)
    assert abs(float(result) - expected) < 1e-7


def test_acq_prob_unknown_mode_raises_value_error() -> None:
    """Unknown mode raises ValueError with debug-useful message.

    Pre-task-#2 code fell through to `valence_inverted` for any mode
    not matching the explicit branches; a typo silently ran the wrong
    branch. The dispatch-table version raises so a typo fails loudly
    at config-load time.

    The error message contract has two parts the user needs to debug:
    (1) the bad mode value, so they can find it in their config; and
    (2) the list of valid modes, so they know what to change it to.
    Without both, the error message would land and not actually help
    debug the issue (per the "error paths must produce useful output"
    rule).
    """
    with pytest.raises(ValueError) as exc_info:
        s17._acq_prob_at_step(
            "bogus", jnp.int32(100), jnp.float32(0.5),
        )
    msg = str(exc_info.value)
    # The bad mode must appear in the message (debug part 1).
    assert "bogus" in msg, (
        f"error message should include the bad mode 'bogus' for "
        f"debugging; got: {msg!r}"
    )
    # The list of valid modes must appear (debug part 2).
    for valid_mode in (
        "off", "stochastic", "periodic", "valence_gated",
        "valence_inverted",
    ):
        assert valid_mode in msg, (
            f"error message should include valid mode '{valid_mode}' "
            f"so the user knows what to change to; got: {msg!r}"
        )


# Functional contract B: periodic step=0 gate.

def test_acq_prob_periodic_does_not_fire_at_step_zero() -> None:
    """step=0 must NOT trigger periodic acquisition.

    Without the `step > 0` guard, `step % PERIODIC_INTERVAL_STEPS == 0`
    is True at step=0 — fires a spurious initial trigger that does
    nothing useful (initial pool is fully active so acquisition is a
    no-op) but still incurs the RNG-draw + Bernoulli cost.
    """
    result = s17._acq_prob_at_step(
        "periodic", jnp.int32(0), jnp.float32(0.0),
    )
    assert float(result) == 0.0


def test_acq_prob_periodic_fires_at_first_interval() -> None:
    """First periodic fire is at step=PERIODIC_INTERVAL_STEPS."""
    result = s17._acq_prob_at_step(
        "periodic",
        jnp.int32(s17.PERIODIC_INTERVAL_STEPS),
        jnp.float32(0.0),
    )
    assert float(result) == 1.0


def test_acq_prob_periodic_silent_just_before_first_fire() -> None:
    """One step before the first interval, periodic must NOT fire."""
    result = s17._acq_prob_at_step(
        "periodic",
        jnp.int32(s17.PERIODIC_INTERVAL_STEPS - 1),
        jnp.float32(0.0),
    )
    assert float(result) == 0.0


def test_acq_prob_periodic_silent_just_after_first_fire() -> None:
    """One step after a fire, periodic must NOT fire again."""
    result = s17._acq_prob_at_step(
        "periodic",
        jnp.int32(s17.PERIODIC_INTERVAL_STEPS + 1),
        jnp.float32(0.0),
    )
    assert float(result) == 0.0


def test_acq_prob_periodic_fires_at_second_interval() -> None:
    """Periodic fires at every multiple of PERIODIC_INTERVAL_STEPS."""
    result = s17._acq_prob_at_step(
        "periodic",
        jnp.int32(2 * s17.PERIODIC_INTERVAL_STEPS),
        jnp.float32(0.0),
    )
    assert float(result) == 1.0
