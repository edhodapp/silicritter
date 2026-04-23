"""Behavioral tests for the closed-loop adrenaline controller.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.closedloop import (
    ControllerParams,
    ControllerState,
    init_controller,
    simulate_closedloop,
    step_closedloop,
)
from silicritter.paired import init_paired_state
from silicritter.plasticity import default_params


def _high_drive_params() -> ControllerParams:
    """Gain-50, range [0.5, 3.0]: step 10 defaults."""
    return ControllerParams(
        decay=0.98, baseline=1.0, gain=50.0, adr_min=0.5, adr_max=3.0,
    )


def test_init_controller_starts_at_baseline() -> None:
    """Zero rate EMAs; adrenaline at baseline."""
    ctrl = init_controller(baseline=1.25)
    assert float(ctrl.rate_a_ema) == 0.0
    assert float(ctrl.rate_b_ema) == 0.0
    assert float(ctrl.adrenaline_b) == 1.25


def test_step_closedloop_updates_state_and_adrenaline() -> None:
    """One step advances the PairedState and produces a new ControllerState."""
    n = 4
    k = 2
    state = init_paired_state(n, k, jax.random.PRNGKey(0))
    ctrl = init_controller()
    params = _high_drive_params()
    zero_vec = jnp.zeros((n,), dtype=jnp.float32)
    nxt_state, nxt_ctrl = step_closedloop(
        state, ctrl, params,
        i_ext_a=zero_vec, i_ext_b=zero_vec,
        valence_a=jnp.float32(1.0), valence_b=jnp.float32(1.0),
        adrenaline_a=jnp.float32(1.0),
        stdp_params=default_params(),
    )
    assert nxt_state.a.lif.v.shape == (n,)
    assert nxt_state.b.lif.v.shape == (n,)
    assert isinstance(nxt_ctrl, ControllerState)
    # Adrenaline stays within clip range.
    assert params.adr_min <= float(nxt_ctrl.adrenaline_b) <= params.adr_max


def test_simulate_closedloop_returns_traces_of_right_shape() -> None:
    """simulate_closedloop returns (spikes_a, spikes_b, adr) with (T, ...)."""
    n = 4
    k = 2
    t = 20
    state = init_paired_state(n, k, jax.random.PRNGKey(1))
    params = _high_drive_params()
    i_ext_a = jnp.full((t, n), 22.0, dtype=jnp.float32)
    i_ext_b = jnp.full((t, n), 16.0, dtype=jnp.float32)
    val_trace = jnp.ones((t,), dtype=jnp.float32)
    adr_a = jnp.ones((t,), dtype=jnp.float32)
    spikes_a, spikes_b, adr_trace = simulate_closedloop(
        state, params,
        i_ext_a, i_ext_b, val_trace, val_trace, adr_a,
        default_params(),
    )
    assert spikes_a.shape == (t, n)
    assert spikes_b.shape == (t, n)
    assert adr_trace.shape == (t,)
    # Adrenaline trace never escapes the clip range.
    assert bool(jnp.all(adr_trace >= params.adr_min))
    assert bool(jnp.all(adr_trace <= params.adr_max))


def test_closedloop_pushes_b_above_open_loop_when_a_is_ahead() -> None:
    """When A fires and B doesn't, controller drives adr_b above baseline."""
    n = 32
    k = 4
    t = 200
    state = init_paired_state(n, k, jax.random.PRNGKey(2))
    # A gets strong drive, B gets weak drive -- A fires, B lags.
    i_ext_a = jnp.full((t, n), 25.0, dtype=jnp.float32)
    i_ext_b = jnp.full((t, n), 10.0, dtype=jnp.float32)
    val_trace = jnp.ones((t,), dtype=jnp.float32)
    adr_a = jnp.ones((t,), dtype=jnp.float32)
    params = _high_drive_params()
    _, _, adr_trace = simulate_closedloop(
        state, params,
        i_ext_a, i_ext_b, val_trace, val_trace, adr_a,
        default_params(),
    )
    # After the EMA has had time to build, adrenaline on B should
    # sit above baseline (controller pushing B to catch up).
    assert float(adr_trace[-1]) > params.baseline


def test_closedloop_clips_output_at_adr_max() -> None:
    """With very high gain and a sustained error, adr_b rails at adr_max."""
    n = 16
    k = 4
    t = 400
    state = init_paired_state(n, k, jax.random.PRNGKey(3))
    i_ext_a = jnp.full((t, n), 30.0, dtype=jnp.float32)
    i_ext_b = jnp.full((t, n), 5.0, dtype=jnp.float32)
    val_trace = jnp.ones((t,), dtype=jnp.float32)
    adr_a = jnp.ones((t,), dtype=jnp.float32)
    # Gain=10000 so any error immediately rails the clip.
    params = ControllerParams(
        decay=0.98, baseline=1.0, gain=10000.0, adr_min=0.5, adr_max=3.0,
    )
    _, _, adr_trace = simulate_closedloop(
        state, params,
        i_ext_a, i_ext_b, val_trace, val_trace, adr_a,
        default_params(),
    )
    assert float(adr_trace[-1]) == params.adr_max


def test_closedloop_clips_output_at_adr_min() -> None:
    """When B outpaces A, negative error should floor adr_b at adr_min."""
    n = 16
    k = 4
    t = 400
    state = init_paired_state(n, k, jax.random.PRNGKey(4))
    i_ext_a = jnp.full((t, n), 5.0, dtype=jnp.float32)
    i_ext_b = jnp.full((t, n), 30.0, dtype=jnp.float32)
    val_trace = jnp.ones((t,), dtype=jnp.float32)
    adr_a = jnp.ones((t,), dtype=jnp.float32)
    params = ControllerParams(
        decay=0.98, baseline=1.0, gain=10000.0, adr_min=0.5, adr_max=3.0,
    )
    _, _, adr_trace = simulate_closedloop(
        state, params,
        i_ext_a, i_ext_b, val_trace, val_trace, adr_a,
        default_params(),
    )
    assert float(adr_trace[-1]) == params.adr_min
