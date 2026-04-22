"""Behavioral tests for the LIF forward simulator.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import jax.numpy as jnp

from silicritter.lif import (
    DT_MS,
    TAU_M_MS,
    V_RESET_MV,
    V_REST_MV,
    V_THRESH_MV,
    LIFState,
    init_state,
    simulate,
    step,
)


def test_init_state_shape_and_values() -> None:
    """Initial state has V at rest and no spikes for every neuron."""
    n = 7
    s = init_state(n)
    assert s.v.shape == (n,)
    assert s.spikes.shape == (n,)
    assert bool(jnp.all(s.v == V_REST_MV))
    assert not bool(jnp.any(s.spikes))


def test_step_with_zero_input_stays_at_rest() -> None:
    """With zero input current and V at rest, the cell remains at rest."""
    n = 5
    state = init_state(n)
    weights = jnp.zeros((n, n), dtype=jnp.float32)
    i_ext = jnp.zeros((n,), dtype=jnp.float32)
    next_state = step(state, weights, i_ext)
    assert bool(jnp.all(next_state.v == V_REST_MV))
    assert not bool(jnp.any(next_state.spikes))


def test_step_crosses_threshold_and_resets() -> None:
    """A strong-enough input drives V over threshold in one step; V is reset."""
    n = 3
    state = init_state(n)
    weights = jnp.zeros((n, n), dtype=jnp.float32)
    # Single-step dv = i_ext * (dt / tau_m). To cross V_thresh from V_rest in
    # one step, need i_ext * (dt / tau_m) >= V_thresh - V_rest.
    min_current = (V_THRESH_MV - V_REST_MV) * (TAU_M_MS / DT_MS)
    i_ext = jnp.full((n,), min_current + 1.0, dtype=jnp.float32)
    next_state = step(state, weights, i_ext)
    assert bool(jnp.all(next_state.spikes))
    assert bool(jnp.all(next_state.v == V_RESET_MV))


def test_step_subthreshold_integrates_without_spiking() -> None:
    """Subthreshold input leaves V between rest and threshold."""
    n = 4
    state = init_state(n)
    weights = jnp.zeros((n, n), dtype=jnp.float32)
    i_ext = jnp.full((n,), 5.0, dtype=jnp.float32)
    next_state = step(state, weights, i_ext)
    assert not bool(jnp.any(next_state.spikes))
    assert bool(jnp.all(next_state.v > V_REST_MV))
    assert bool(jnp.all(next_state.v < V_THRESH_MV))


def test_simulate_returns_expected_shapes() -> None:
    """simulate() returns a final state and a (T, N) spike trace."""
    n, t = 6, 20
    state = init_state(n)
    weights = jnp.zeros((n, n), dtype=jnp.float32)
    i_ext_trace = jnp.zeros((t, n), dtype=jnp.float32)
    final_state, spike_trace = simulate(state, weights, i_ext_trace)
    assert isinstance(final_state, LIFState)
    assert final_state.v.shape == (n,)
    assert final_state.spikes.shape == (n,)
    assert spike_trace.shape == (t, n)


def test_simulate_periodic_spiking_under_constant_drive() -> None:
    """Constant suprathreshold drive produces regular repeated spiking."""
    n = 2
    t = 200
    state = init_state(n)
    weights = jnp.zeros((n, n), dtype=jnp.float32)
    # Drive of 50 mV with tau_m=20ms, dt=1ms gives a per-step voltage
    # increment ~2.5 mV from near-rest, so V crosses V_thresh about every
    # 6 ms. Over 200 ms per cell we expect ~30 spikes, ~60 total across
    # two cells. Bounds are loose enough to absorb boundary-step effects
    # but tight enough to catch single-spike-and-die failure modes.
    i_ext_trace = jnp.full((t, n), 50.0, dtype=jnp.float32)
    _, spike_trace = simulate(state, weights, i_ext_trace)
    per_cell_counts = spike_trace.sum(axis=0)
    assert bool(jnp.all(per_cell_counts >= 10))
    total_spikes = int(spike_trace.sum())
    assert 40 <= total_spikes <= 200
