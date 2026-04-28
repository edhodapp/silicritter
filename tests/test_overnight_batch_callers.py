"""Regression test: overnight_batch.py honors changed step16/step17 signatures.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

When step16/step17 helper signatures change (return type, parameter list),
cross-file callers in `experiments/overnight_batch.py` must be updated to
match. Without this test, the only way to detect a stale caller is to
actually run a long-duration overnight batch, which means the regression
ships and is caught on a real run.

Two regressions caught and fixed by this test's existence:

1. Task #1 (raster-fix cleanup): changed `_measure_fitness` from
   `tuple[float, jax.Array, jax.Array]` to `float`. The in-file callers
   were updated; the cross-file callers in overnight_batch.py
   (`fit_before, _, _ = s16_measure_fitness(...)`) were missed and
   would have raised `TypeError: cannot unpack non-iterable float
   object` on the next overnight run.

2. Task #4 (cross_e_mask refactor): added `a_is_inh` parameter to
   `_describe_pool`. The in-file callers were updated; the
   overnight_batch.py call site (`s16_describe_pool(trained_pool)`)
   was missed and would have raised `TypeError: missing 1 required
   positional argument: 'a_is_inh'`.

The pytest gate runs only `tests/`, but tests don't import or exercise
`overnight_batch.py` by default, so even the wide test suite missed
both regressions until manual review caught them.

This test imports overnight_batch's `_step16_once` private helper
(which contains the affected call sites), monkey-patches the heavy
step-count module constants down to test-cheap values, and runs it
end-to-end. If any future signature change to step16's helpers leaves
overnight_batch.py stale, this test fails loudly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

# pylint: disable=wrong-import-position,import-error
import overnight_batch  # noqa: E402

# pylint: disable=protected-access


# Step counts small enough for a unit-test run but still exercise the
# full call chain (build drives -> training scan -> measure fitness ->
# describe pool). Constraints:
# - TEST_MEASURE_STEPS must be divisible by WINDOW_STEPS (100).
# - TEST_MEASURE_STEPS must give n_windows >= 2 so the fitness formula
#   `b_rate[:-1] - a_rate[1:]` averages over a non-empty array; at
#   n_windows=1 that subtraction is shape (0,) and mean is NaN.
# - TEST_TRAIN_STEPS must be divisible by len(A_DRIVE_PROFILE) (4).
TEST_MEASURE_STEPS = 200
TEST_TRAIN_STEPS = 200


@pytest.fixture(name="cheap_steps")
def _cheap_steps_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monkey-patch step-count constants to test-cheap values."""
    monkeypatch.setattr(
        overnight_batch, "S16_MEASURE_STEPS", TEST_MEASURE_STEPS,
    )
    monkeypatch.setattr(
        overnight_batch, "S16_TRAIN_STEPS", TEST_TRAIN_STEPS,
    )


def test_overnight_batch_step16_once_honors_signatures(
    cheap_steps: None,
) -> None:
    """`_step16_once` must run end-to-end and return a metrics dict.

    Pre-fix state: raises `TypeError` either at the
    `fit_before, _, _ = s16_measure_fitness(...)` unpack (regression
    from task #1's float return) or at `s16_describe_pool(trained_pool)`
    (regression from task #4's added `a_is_inh` parameter).

    Post-fix state: returns the documented metrics dict.
    """
    del cheap_steps  # fixture applied; arg present for pytest dependency
    result = overnight_batch._step16_once(
        seed=0,
        plasticity_rate=1.0,
        init_v_mean=1.0,
        init_v_std=0.3,
    )

    assert isinstance(result, dict), (
        f"_step16_once must return a dict; got {type(result).__name__}"
    )
    expected_keys = {
        "fit_before", "fit_after", "train_time",
        "v_mean", "v_std", "cross_e_frac", "valence_mean",
    }
    assert expected_keys.issubset(result.keys()), (
        f"missing keys in result: {expected_keys - result.keys()}"
    )
    fit_before = result["fit_before"]
    fit_after = result["fit_after"]
    cross_e_frac = result["cross_e_frac"]
    assert isinstance(fit_before, float), (
        f"fit_before must be float (would be tuple under old "
        f"_measure_fitness signature); got {type(fit_before)}"
    )
    assert isinstance(fit_after, float), (
        f"fit_after must be float; got {type(fit_after)}"
    )
    assert isinstance(cross_e_frac, float), (
        f"cross_e_frac must be float; got {type(cross_e_frac)}"
    )
    # Value-sanity (catches NaN / inf / out-of-range silent regressions
    # that types alone don't catch — RNG-immune).
    assert math.isfinite(fit_before), (
        f"fit_before must be finite (NaN/inf indicates broken "
        f"simulation); got {fit_before}"
    )
    assert math.isfinite(fit_after), (
        f"fit_after must be finite; got {fit_after}"
    )
    assert 0.0 <= cross_e_frac <= 1.0, (
        f"cross_e_frac is a fraction-of-pool, must be in [0, 1]; "
        f"got {cross_e_frac}"
    )
