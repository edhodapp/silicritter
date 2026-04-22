"""Behavioral tests for three-factor STDP on slot-pool synapses.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.lif import LIFState, init_state
from silicritter.plasticity import (
    PlasticNetState,
    STDPParams,
    Traces,
    default_params,
    init_traces,
    simulate_plastic,
    step_plastic,
)
from silicritter.slotpool import SlotPool, init_random


def _pool_with(
    pre_ids: jax.Array,
    v: jax.Array,
    plasticity_rate: jax.Array,
    active: jax.Array,
) -> SlotPool:
    """Construct a SlotPool from explicit arrays (for hand-built tests)."""
    return SlotPool(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=plasticity_rate,
        active=active,
    )


def test_init_traces_shapes_and_zero() -> None:
    """Fresh traces have the requested shapes and are all zero."""
    traces = init_traces(n_pre=5, n_post=7)
    assert traces.pre.shape == (5,)
    assert traces.post.shape == (7,)
    assert bool(jnp.all(traces.pre == 0.0))
    assert bool(jnp.all(traces.post == 0.0))


def test_traces_decay_toward_zero_without_spikes() -> None:
    """With no spikes emitted, traces decay geometrically toward zero."""
    n = 3
    pre_ids = jnp.zeros((n, 1), dtype=jnp.int32)
    v = jnp.full((n, 1), 0.1, dtype=jnp.float32)
    pool = _pool_with(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=jnp.ones_like(v),
        active=jnp.ones_like(v, dtype=jnp.bool_),
    )
    # Start with non-zero traces and a V that can't cross threshold.
    start_traces = Traces(
        pre=jnp.full((n,), 1.0, dtype=jnp.float32),
        post=jnp.full((n,), 1.0, dtype=jnp.float32),
    )
    lif = LIFState(
        v=jnp.full((n,), -65.0, dtype=jnp.float32),
        spikes=jnp.zeros((n,), dtype=jnp.bool_),
    )
    state = PlasticNetState(lif=lif, pool=pool, traces=start_traces)
    params = default_params()

    next_state = step_plastic(
        state,
        i_ext=jnp.zeros((n,), dtype=jnp.float32),
        valence=jnp.float32(1.0),
        params=params,
    )
    # Decay factor exp(-dt/tau) with defaults (dt=1, tau=20) is ~0.9512.
    expected = jnp.exp(jnp.float32(-1.0 / 20.0))
    assert jnp.allclose(next_state.traces.pre, expected, atol=1e-6)
    assert jnp.allclose(next_state.traces.post, expected, atol=1e-6)


def test_zero_valence_freezes_weights() -> None:
    """With valence = 0 no slot v changes, even with spiking activity."""
    n = 4
    pool = init_random(n, n, 3, jax.random.PRNGKey(0))
    v0 = pool.v.copy()
    lif = init_state(n)
    traces = init_traces(n, n)
    state = PlasticNetState(lif=lif, pool=pool, traces=traces)
    params = default_params()

    t = 50
    i_ext_trace = jnp.full((t, n), 25.0, dtype=jnp.float32)
    valence_trace = jnp.zeros((t,), dtype=jnp.float32)

    final_state, spikes = simulate_plastic(
        state, i_ext_trace, valence_trace, params
    )
    # Confirm the network actually spiked - this rules out a vacuous pass.
    assert int(spikes.sum()) > 0
    assert jnp.allclose(final_state.pool.v, v0, atol=0.0)


def test_zero_plasticity_rate_freezes_weights() -> None:
    """Slots with plasticity_rate = 0 do not update under any activity."""
    n = 4
    pool = init_random(n, n, 3, jax.random.PRNGKey(1))
    innate_pool = pool._replace(plasticity_rate=jnp.zeros_like(pool.v))
    v0 = innate_pool.v.copy()
    state = PlasticNetState(
        lif=init_state(n),
        pool=innate_pool,
        traces=init_traces(n, n),
    )
    params = default_params()

    t = 50
    i_ext_trace = jnp.full((t, n), 25.0, dtype=jnp.float32)
    valence_trace = jnp.ones((t,), dtype=jnp.float32)

    final_state, spikes = simulate_plastic(
        state, i_ext_trace, valence_trace, params
    )
    assert int(spikes.sum()) > 0
    assert jnp.allclose(final_state.pool.v, v0, atol=0.0)


def test_ltp_from_pretrace_and_postspike() -> None:
    """A built-up pre_trace with a fresh post spike produces LTP (Δv > 0).

    Checks the LTP magnitude equals the expected
    valence * plasticity_rate * a_plus * pre_trace_decayed within a
    tight tolerance, not just "grew."
    """
    # Two-cell net; slot (post=1, k=0) is bound to pre=0.
    pre_ids = jnp.array([[1], [0]], dtype=jnp.int32)
    v_init = jnp.array([[0.1], [0.1]], dtype=jnp.float32)
    pool = _pool_with(
        pre_ids=pre_ids,
        v=v_init,
        plasticity_rate=jnp.ones_like(v_init),
        active=jnp.ones_like(v_init, dtype=jnp.bool_),
    )
    lif = LIFState(
        v=jnp.array([-65.0, -51.0], dtype=jnp.float32),
        spikes=jnp.array([False, False], dtype=jnp.bool_),
    )
    # Pre-existing pre_trace at pre=0 from earlier activity.
    pre_trace_init = 0.9
    traces = Traces(
        pre=jnp.array([pre_trace_init, 0.0], dtype=jnp.float32),
        post=jnp.zeros((2,), dtype=jnp.float32),
    )
    state = PlasticNetState(lif=lif, pool=pool, traces=traces)
    params = default_params()

    i_ext = jnp.array([0.0, 50.0], dtype=jnp.float32)
    nxt = step_plastic(state, i_ext, jnp.float32(1.0), params)

    # Post=1 spiked; slot (1, 0) sees pre=0's decayed pre_trace.
    assert bool(nxt.lif.spikes[1])
    decay_pre = float(jnp.exp(jnp.float32(-1.0 / params.tau_pre_ms)))
    expected_dv = params.a_plus * (pre_trace_init * decay_pre)
    actual_dv = float(nxt.pool.v[1, 0] - pool.v[1, 0])
    assert abs(actual_dv - expected_dv) < 1e-6


def test_ltd_decreases_weight_when_post_leads_pre() -> None:
    """Post trace is high; a pre spike causes LTD (negative dv)."""
    # Slot at post=0, k=0 is bound to pre=1. We want pre=1 to spike this
    # step while post=0 did not spike recently enough for zero post_trace.
    pre_ids = jnp.array([[1], [0]], dtype=jnp.int32)
    v_init = jnp.array([[0.2], [0.2]], dtype=jnp.float32)
    pool = _pool_with(
        pre_ids=pre_ids,
        v=v_init,
        plasticity_rate=jnp.ones_like(v_init),
        active=jnp.ones_like(v_init, dtype=jnp.bool_),
    )
    lif = LIFState(
        v=jnp.array([-65.0, -65.0], dtype=jnp.float32),
        spikes=jnp.array([False, False], dtype=jnp.bool_),
    )
    # post=0 has a built-up post_trace from recent activity; pre trace 0.
    traces = Traces(
        pre=jnp.zeros((2,), dtype=jnp.float32),
        post=jnp.array([0.9, 0.0], dtype=jnp.float32),
    )
    state = PlasticNetState(lif=lif, pool=pool, traces=traces)
    params = default_params()

    # Strong drive on post=1 so it spikes this step. That makes pre=1
    # (the pre bound by slot (0, 0)) a "pre spike" under our convention
    # (pre_spike_slot = spike_post[pool.pre_ids]). Post=0's post_trace
    # of 0.9 then causes LTD on slot (0, 0).
    i_ext = jnp.array([0.0, 500.0], dtype=jnp.float32)
    post_trace_init = 0.9
    nxt = step_plastic(state, i_ext, jnp.float32(1.0), params)
    assert bool(nxt.lif.spikes[1])
    assert not bool(nxt.lif.spikes[0])
    decay_post = float(jnp.exp(jnp.float32(-1.0 / params.tau_post_ms)))
    expected_dv = -params.a_minus * (post_trace_init * decay_post)
    actual_dv = float(nxt.pool.v[0, 0] - pool.v[0, 0])
    assert abs(actual_dv - expected_dv) < 1e-6


def test_weights_stay_within_v_min_v_max() -> None:
    """Over a long simulation with strong drive, weights respect the clip."""
    n = 6
    pool = init_random(n, n, 3, jax.random.PRNGKey(2))
    params = STDPParams(
        tau_pre_ms=20.0,
        tau_post_ms=20.0,
        a_plus=0.1,  # aggressive to push against the ceiling
        a_minus=0.1,
        v_min=-0.05,
        v_max=0.25,
    )
    state = PlasticNetState(
        lif=init_state(n),
        pool=pool,
        traces=init_traces(n, n),
    )
    t = 200
    i_ext_trace = jnp.full((t, n), 25.0, dtype=jnp.float32)
    valence_trace = jnp.ones((t,), dtype=jnp.float32)

    final_state, _ = simulate_plastic(
        state, i_ext_trace, valence_trace, params
    )
    assert bool(jnp.all(final_state.pool.v >= params.v_min))
    assert bool(jnp.all(final_state.pool.v <= params.v_max))


def test_valence_trace_modulates_weight_trajectories() -> None:
    """Different valence traces produce distinct final weight patterns."""
    n = 4
    pool = init_random(n, n, 2, jax.random.PRNGKey(10))
    v0 = pool.v.copy()
    state = PlasticNetState(
        lif=init_state(n),
        pool=pool,
        traces=init_traces(n, n),
    )
    params = default_params()

    t = 100
    i_ext_trace = jnp.full((t, n), 25.0, dtype=jnp.float32)
    v_pos = jnp.ones((t,), dtype=jnp.float32)
    v_neg = -jnp.ones((t,), dtype=jnp.float32)
    v_zero = jnp.zeros((t,), dtype=jnp.float32)

    final_pos, _ = simulate_plastic(state, i_ext_trace, v_pos, params)
    final_neg, _ = simulate_plastic(state, i_ext_trace, v_neg, params)
    final_zero, _ = simulate_plastic(state, i_ext_trace, v_zero, params)

    # Zero valence: weights unchanged.
    assert jnp.allclose(final_zero.pool.v, v0)
    # Positive and negative valence: both diverge from initial.
    assert not jnp.allclose(final_pos.pool.v, v0)
    assert not jnp.allclose(final_neg.pool.v, v0)
    # Positive and negative valence produce distinct trajectories.
    assert not jnp.allclose(final_pos.pool.v, final_neg.pool.v)


def test_simulate_plastic_returns_expected_shapes() -> None:
    """simulate_plastic returns a final state and a (T, N_post) spike trace."""
    n = 5
    t = 25
    pool = init_random(n, n, 2, jax.random.PRNGKey(3))
    state = PlasticNetState(
        lif=init_state(n),
        pool=pool,
        traces=init_traces(n, n),
    )
    i_ext_trace = jnp.full((t, n), 18.0, dtype=jnp.float32)
    valence_trace = jnp.ones((t,), dtype=jnp.float32)
    final_state, spike_trace = simulate_plastic(
        state, i_ext_trace, valence_trace, default_params()
    )
    assert isinstance(final_state, PlasticNetState)
    assert final_state.pool.v.shape == (n, 2)
    assert spike_trace.shape == (t, n)
