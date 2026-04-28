"""Memory-bound contract for the long-training scans in step16/step17.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

The training scans must NOT retain per-step (T, N_NEURONS) spike rasters.
At T=10M, N=256 those rasters are ~5 GB and exceed the 4 GB GPU ceiling
on the development laptop. The scan only needs scalar per-step outputs
(mean rates, valence) for downstream diagnostics; full rasters are dead
weight that gates long-duration revalidation runs.

These tests pin the contract: the second tuple of returned arrays from
each _training_scan must contain only scalar-per-step traces, not
N_NEURONS-shaped rasters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

# pylint: disable=wrong-import-position,import-error
import step16_stdp_learning as s16  # noqa: E402
import step17_structural_growth as s17  # noqa: E402
from silicritter.paired import PairedState  # noqa: E402
from silicritter.slotpool import assign_ei_identity  # noqa: E402

# Tests legitimately exercise the private _training_scan helpers and their
# private dependencies in the experiment modules — this is the system under
# test, not an encapsulation violation.
# pylint: disable=protected-access


T_TEST = 400  # must be divisible by len(A_DRIVE_PROFILE) = 4


def test_step16_training_scan_returns_no_per_step_raster() -> None:
    """step16 _training_scan must not return (T, N_NEURONS) spike rasters."""
    pool_b = s16._random_b_pool(seed=0, plasticity_rate=1.0)
    state = s16._build_state(pool_b)
    a_is_inh = assign_ei_identity(s16.N_NEURONS, s16.INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(s16.N_NEURONS, s16.INHIBITORY_FRACTION)
    i_ext_a, i_ext_b = s16._build_drives(T_TEST)

    result = s16._training_scan(
        state, a_is_inh, b_is_inh, i_ext_a, i_ext_b,
    )
    # result[0] is final_state (PairedState pytree); arrays inside it are
    # carry-state, not per-step traces, so they are exempt. Asserting the
    # type pins the tuple shape — a future refactor that reorders the
    # return tuple would fail here rather than silently exempt a real
    # per-step trace.
    assert isinstance(result[0], PairedState), (
        f"expected result[0] to be PairedState (carry); got {type(result[0])}"
    )
    per_step_outputs = result[1:]
    # Pin the count too: a future refactor that adds a 4th per-step
    # trace alongside (rate_a, rate_b, valence_b) would otherwise pass
    # silently because the new output isn't in the per-element loop's
    # expectation set.
    assert len(per_step_outputs) == 3, (
        f"expected 3 per-step outputs (rate_a, rate_b, valence_b); "
        f"got {len(per_step_outputs)}"
    )
    for i, arr in enumerate(per_step_outputs):
        assert isinstance(arr, jax.Array), (
            f"output {i} is not a jax.Array: {type(arr)}"
        )
        # Per-step scalar traces stack to shape (T,). A retained raster
        # would be (T, N_NEURONS) — rank 2 — so pinning the exact scalar
        # shape catches the regression unambiguously even if T_TEST ever
        # coincides with N_NEURONS.
        assert arr.shape == (T_TEST,), (
            f"output {i} has shape {arr.shape}; expected ({T_TEST},) — "
            f"non-scalar per-step output indicates a retained raster"
        )


def test_step17_training_scan_returns_no_per_step_raster() -> None:
    """step17 _training_scan must not return (T, N_NEURONS) spike rasters."""
    pool_b = s17._random_b_pool(seed=0)
    state = s17._build_state(pool_b)
    a_is_inh = assign_ei_identity(s17.N_NEURONS, s17.INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(s17.N_NEURONS, s17.INHIBITORY_FRACTION)
    i_ext_a, i_ext_b = s17._build_drives(T_TEST)
    config = s17.Config(
        name="test",
        acq_mode="off",
        pre_id_source="uniform",
        acq_initial_v=1.0,
        release_threshold=0.05,
        release_duration=200,
    )

    result = s17._training_scan(
        state, a_is_inh, b_is_inh, i_ext_a, i_ext_b, config, seed=0,
    )
    assert isinstance(result[0], PairedState), (
        f"expected result[0] to be PairedState (carry); got {type(result[0])}"
    )
    per_step_outputs = result[1:]
    assert len(per_step_outputs) == 2, (
        f"expected 2 per-step outputs (valence_b, n_active); "
        f"got {len(per_step_outputs)}"
    )
    for i, arr in enumerate(per_step_outputs):
        assert isinstance(arr, jax.Array), (
            f"output {i} is not a jax.Array: {type(arr)}"
        )
        assert arr.shape == (T_TEST,), (
            f"output {i} has shape {arr.shape}; expected ({T_TEST},) — "
            f"non-scalar per-step output indicates a retained raster"
        )
