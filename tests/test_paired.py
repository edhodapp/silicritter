"""Behavioral tests for paired-agent simulation.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.lif import LIFState
from silicritter.paired import (
    PairedState,
    init_paired_state,
    make_pool_for_partner,
    simulate_paired,
    step_paired,
)
from silicritter.plasticity import (
    PlasticNetState,
    default_params,
    init_traces,
)
from silicritter.slotpool import SlotPool
from silicritter.structural import StructuralParams


def test_init_paired_state_has_combined_pre_space() -> None:
    """Both agents' pools have pre_ids in [0, 2*n_neurons) and right traces."""
    n = 6
    k = 3
    state = init_paired_state(n, k, jax.random.PRNGKey(0))
    assert state.a.pool.pre_ids.shape == (n, k)
    assert state.b.pool.pre_ids.shape == (n, k)
    assert bool(jnp.all(state.a.pool.pre_ids < 2 * n))
    assert bool(jnp.all(state.b.pool.pre_ids < 2 * n))
    # Trace pre vector spans the combined raster; post stays over own n.
    assert state.a.traces.pre.shape == (2 * n,)
    assert state.a.traces.post.shape == (n,)
    assert state.b.traces.pre.shape == (2 * n,)
    assert state.b.traces.post.shape == (n,)


def test_step_paired_preserves_shapes() -> None:
    """step_paired returns a PairedState with shape-invariant fields."""
    n = 5
    k = 2
    state = init_paired_state(n, k, jax.random.PRNGKey(1))
    stdp_params = default_params()
    zero_vec = jnp.zeros((n,), dtype=jnp.float32)
    nxt = step_paired(
        state,
        i_ext_a=zero_vec,
        i_ext_b=zero_vec,
        valence_a=jnp.float32(1.0),
        valence_b=jnp.float32(1.0),
        adrenaline_a=jnp.float32(1.0),
        adrenaline_b=jnp.float32(1.0),
        stdp_params=stdp_params,
    )
    assert isinstance(nxt, PairedState)
    assert nxt.a.lif.v.shape == (n,)
    assert nxt.b.lif.v.shape == (n,)
    assert nxt.a.pool.pre_ids.shape == (n, k)
    assert nxt.a.traces.pre.shape == (2 * n,)
    assert nxt.b.traces.post.shape == (n,)


def test_simulate_paired_returns_both_spike_traces() -> None:
    """simulate_paired returns per-agent spike traces of shape (T, n)."""
    n = 4
    k = 2
    t = 30
    state = init_paired_state(n, k, jax.random.PRNGKey(2))
    stdp_params = default_params()
    i_ext_a_trace = jnp.full((t, n), 20.0, dtype=jnp.float32)
    i_ext_b_trace = jnp.full((t, n), 20.0, dtype=jnp.float32)
    valence_trace = jnp.ones((t,), dtype=jnp.float32)
    adrenaline_trace = jnp.ones((t,), dtype=jnp.float32)
    final_state, spikes_a, spikes_b = simulate_paired(
        state,
        i_ext_a_trace, i_ext_b_trace,
        valence_trace, valence_trace,
        adrenaline_trace, adrenaline_trace,
        stdp_params,
    )
    assert isinstance(final_state, PairedState)
    assert spikes_a.shape == (t, n)
    assert spikes_b.shape == (t, n)


def test_cross_agent_synaptic_influence() -> None:
    """Cross-slot v moves B's V relative to a zero-cross-v control.

    Pairs a large cross-weight against a zero cross-weight under otherwise
    identical dynamics; the only difference between the two runs is the
    cross-slot v. B's v[0] must differ between them by at least the
    expected analytic synaptic contribution, ruling out a test that
    would pass purely via membrane leak.
    """
    n = 2
    # Slot (0, 0) is bound to pre=2 which, under the [own_b, partner_a]
    # convention for B's combined pre-raster, is A's neuron 0. A's
    # neuron 0 fires at step entry, so this slot's v determines how
    # much synaptic current B's neuron 0 sees this step.
    pre_ids = jnp.array([[2, 0], [0, 0]], dtype=jnp.int32)
    plasticity_rate = jnp.ones((n, 2), dtype=jnp.float32)
    active = jnp.ones((n, 2), dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)

    def _pool(cross_v: float) -> SlotPool:
        v = jnp.array([[cross_v, 0.0], [0.0, 0.0]], dtype=jnp.float32)
        return SlotPool(
            pre_ids=pre_ids,
            v=v,
            plasticity_rate=plasticity_rate,
            active=active,
            release_counter=release_counter,
        )

    lif_a = LIFState(
        v=jnp.full((n,), -65.0, dtype=jnp.float32),
        spikes=jnp.array([True, False], dtype=jnp.bool_),
    )
    lif_b = LIFState(
        v=jnp.array([-50.1, -65.0], dtype=jnp.float32),
        spikes=jnp.zeros((n,), dtype=jnp.bool_),
    )

    def _run(cross_v: float) -> float:
        state = PairedState(
            a=PlasticNetState(
                lif=lif_a,
                pool=make_pool_for_partner(n, 2, jax.random.PRNGKey(11)),
                traces=init_traces(n_pre=2 * n, n_post=n),
            ),
            b=PlasticNetState(
                lif=lif_b,
                pool=_pool(cross_v),
                traces=init_traces(n_pre=2 * n, n_post=n),
            ),
        )
        zero_vec = jnp.zeros((n,), dtype=jnp.float32)
        nxt = step_paired(
            state,
            i_ext_a=zero_vec, i_ext_b=zero_vec,
            valence_a=jnp.float32(0.0), valence_b=jnp.float32(0.0),
            adrenaline_a=jnp.float32(1.0),
            adrenaline_b=jnp.float32(1.0),
            stdp_params=default_params(),
        )
        return float(nxt.b.lif.v[0])

    v_without_cross = _run(cross_v=0.0)
    v_with_cross = _run(cross_v=0.5)
    # Expected analytic contribution at this step:
    # dv = (-(v - v_rest) + i_total) * dt/tau_m
    # With cross_v=0.5 and tau_m=20ms, dt=1ms:
    #   extra_dv = 0.5 * (1/20) = 0.025 mV
    # So v_with_cross should be >= v_without_cross + 0.025 - numeric slop.
    expected_extra = 0.5 * (1.0 / 20.0)
    assert v_with_cross - v_without_cross > expected_extra - 1e-5
    assert v_with_cross - v_without_cross < expected_extra + 1e-5


def test_paired_agents_independent_with_zero_cross_weights() -> None:
    """With all cross-bound slot weights zero, agents evolve independently."""
    n = 3
    k = 2
    t = 20
    state = init_paired_state(n, k, jax.random.PRNGKey(4))
    # Zero out weights on cross slots (pre_ids >= n) for both agents so
    # cross-connections exist structurally but contribute zero current.
    mask_a = state.a.pool.pre_ids < n
    mask_b = state.b.pool.pre_ids < n
    pool_a = state.a.pool._replace(
        v=jnp.where(mask_a, state.a.pool.v, jnp.float32(0.0))
    )
    pool_b = state.b.pool._replace(
        v=jnp.where(mask_b, state.b.pool.v, jnp.float32(0.0))
    )
    state = PairedState(
        a=state.a._replace(pool=pool_a),
        b=state.b._replace(pool=pool_b),
    )

    # Drive only A; B gets zero input and has only zeroed-out cross
    # connections so it should stay silent for the whole run.
    i_ext_a = jnp.full((t, n), 25.0, dtype=jnp.float32)
    i_ext_b = jnp.zeros((t, n), dtype=jnp.float32)
    zeros_t = jnp.zeros((t,), dtype=jnp.float32)
    ones_t = jnp.ones((t,), dtype=jnp.float32)
    _, spikes_a, spikes_b = simulate_paired(
        state,
        i_ext_a, i_ext_b,
        zeros_t, zeros_t,  # valence = 0: freeze weights
        ones_t, ones_t,
        default_params(),
    )
    # A spikes under the drive; B stays silent because its only inputs
    # (self-recurrent from silent B + cross from A through zero weights)
    # are all zero.
    assert int(spikes_a.sum()) > 0
    assert int(spikes_b.sum()) == 0


def test_step_paired_ei_substrate_changes_synaptic_current() -> None:
    """Providing E/I substrate negates+scales I-sourced contributions.

    Two identical runs differing only in whether `a_is_inhibitory` /
    `b_is_inhibitory` are provided; B receives synaptic current from
    A's spikes, and with partner A having an I-neuron that fires, the
    E/I run must produce a visibly different B membrane potential than
    the no-E/I run.
    """
    n = 4
    k = 2
    state = init_paired_state(n, k, jax.random.PRNGKey(100))
    # Force A's neurons [2, 3] to fire (I-neurons under the fraction
    # convention: last 50% at this small n).
    spikes_a_forced = jnp.array([False, False, True, True], dtype=jnp.bool_)
    state = PairedState(
        a=state.a._replace(lif=state.a.lif._replace(spikes=spikes_a_forced)),
        b=state.b,
    )
    # Make B's slots all bind to A-side (indices n..2n) so the A-spike
    # signal clearly drives B's synaptic input.
    n_pre = 2 * n
    cross_pre_ids = jnp.full((n, k), n, dtype=jnp.int32)
    _ = n_pre  # suppress lint
    pool_b_cross = SlotPool(
        pre_ids=cross_pre_ids,
        v=jnp.full((n, k), 0.3, dtype=jnp.float32),
        plasticity_rate=jnp.ones((n, k), dtype=jnp.float32),
        active=jnp.ones((n, k), dtype=jnp.bool_),
        release_counter=jnp.zeros((n, k), dtype=jnp.int32),
    )
    state = PairedState(
        a=state.a,
        b=state.b._replace(pool=pool_b_cross),
    )

    stdp_params = default_params()
    zero_vec = jnp.zeros((n,), dtype=jnp.float32)
    # Reference run: no E/I.
    nxt_no_ei = step_paired(
        state,
        i_ext_a=zero_vec, i_ext_b=zero_vec,
        valence_a=jnp.float32(0.0), valence_b=jnp.float32(0.0),
        adrenaline_a=jnp.float32(1.0), adrenaline_b=jnp.float32(1.0),
        stdp_params=stdp_params,
    )
    # E/I run: A has last 50% inhibitory (so all firing A-neurons
    # are I), B is all-excitatory.
    a_is_inh = jnp.array([False, False, True, True], dtype=jnp.bool_)
    b_is_inh = jnp.zeros((n,), dtype=jnp.bool_)
    nxt_with_ei = step_paired(
        state,
        i_ext_a=zero_vec, i_ext_b=zero_vec,
        valence_a=jnp.float32(0.0), valence_b=jnp.float32(0.0),
        adrenaline_a=jnp.float32(1.0), adrenaline_b=jnp.float32(1.0),
        stdp_params=stdp_params,
        a_is_inhibitory=a_is_inh,
        b_is_inhibitory=b_is_inh,
        i_weight_multiplier=4.0,
    )
    # No-EI run: B's neuron 0 receives positive synaptic current
    # (A neurons 2,3 firing, K=2 slots bound to pre=n (A neuron 0,
    # not firing), wait -- let me re-check. Slots bind to pre=n,
    # which is A-neuron 0. A-neuron 0 is NOT firing (only 2,3 are).
    # So I need slots bound to firing A-neurons.
    # Rebinding: bind to pre=n+2 (A-neuron 2, which IS firing).
    cross_pre_ids_firing = jnp.full((n, k), n + 2, dtype=jnp.int32)
    pool_b_firing = pool_b_cross._replace(pre_ids=cross_pre_ids_firing)
    state_firing = PairedState(
        a=state.a,
        b=state.b._replace(pool=pool_b_firing),
    )
    nxt_no_ei = step_paired(
        state_firing,
        i_ext_a=zero_vec, i_ext_b=zero_vec,
        valence_a=jnp.float32(0.0), valence_b=jnp.float32(0.0),
        adrenaline_a=jnp.float32(1.0), adrenaline_b=jnp.float32(1.0),
        stdp_params=stdp_params,
    )
    nxt_with_ei = step_paired(
        state_firing,
        i_ext_a=zero_vec, i_ext_b=zero_vec,
        valence_a=jnp.float32(0.0), valence_b=jnp.float32(0.0),
        adrenaline_a=jnp.float32(1.0), adrenaline_b=jnp.float32(1.0),
        stdp_params=stdp_params,
        a_is_inhibitory=a_is_inh,
        b_is_inhibitory=b_is_inh,
        i_weight_multiplier=4.0,
    )
    # A-neuron 2 (the pre for every B slot) is inhibitory, so with
    # E/I enabled, B's synaptic input flips sign and scales by 4x.
    # V after one step: no-E/I gets positive dV, E/I gets negative
    # (V goes further below rest, or less above).
    assert float(nxt_no_ei.b.lif.v[0]) > float(nxt_with_ei.b.lif.v[0])


def test_step_paired_compatible_with_structural_release() -> None:
    """Structural release applies per agent when structural_params set."""
    n = 4
    k = 3
    # Seed weak weights on both agents so some slots drift to release.
    state = init_paired_state(n, k, jax.random.PRNGKey(5))
    weak_v = jnp.full_like(state.a.pool.v, 0.005)
    state = PairedState(
        a=state.a._replace(pool=state.a.pool._replace(v=weak_v)),
        b=state.b._replace(pool=state.b.pool._replace(v=weak_v)),
    )
    structural_params = StructuralParams(
        v_release_threshold=0.01, release_dwell_steps=5
    )
    stdp_params = default_params()

    # Run long enough for dwell to trigger.
    t = 20
    i_ext_a = jnp.zeros((t, n), dtype=jnp.float32)
    i_ext_b = jnp.zeros((t, n), dtype=jnp.float32)
    valence = jnp.zeros((t,), dtype=jnp.float32)
    adrenaline = jnp.ones((t,), dtype=jnp.float32)
    final_state, _, _ = simulate_paired(
        state,
        i_ext_a, i_ext_b,
        valence, valence,
        adrenaline, adrenaline,
        stdp_params,
        structural_params=structural_params,
    )
    # Both agents' pools should have released some (or all) slots.
    assert int(final_state.a.pool.active.sum()) < n * k
    assert int(final_state.b.pool.active.sum()) < n * k
