"""Behavioral tests for the slot-pool synapse representation.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.lif import (
    V_REST_MV,
    LIFState,
    init_state,
)
from silicritter.lif import simulate as simulate_dense
from silicritter.lif import step as step_dense
from silicritter.slotpool import (
    SlotPool,
    assign_ei_identity,
    effective_weights,
    init_random,
    simulate,
    step,
    synaptic_current,
)


def test_init_random_shapes_and_types() -> None:
    """init_random produces correctly-shaped and typed SlotPool arrays."""
    n_post, n_pre, k = 7, 13, 4
    pool = init_random(n_post, n_pre, k, jax.random.PRNGKey(0))
    assert pool.pre_ids.shape == (n_post, k)
    assert pool.v.shape == (n_post, k)
    assert pool.plasticity_rate.shape == (n_post, k)
    assert pool.active.shape == (n_post, k)
    assert pool.pre_ids.dtype == jnp.int32
    assert pool.v.dtype == jnp.float32
    assert pool.plasticity_rate.dtype == jnp.float32
    assert pool.active.dtype == jnp.bool_
    assert bool(jnp.all(pool.active))
    assert bool(jnp.all(pool.pre_ids >= 0))
    assert bool(jnp.all(pool.pre_ids < n_pre))


def test_synaptic_current_no_spikes_gives_zero() -> None:
    """With no presynaptic spikes, every postsynaptic current is zero."""
    pool = init_random(5, 8, 3, jax.random.PRNGKey(1))
    spikes = jnp.zeros((8,), dtype=jnp.bool_)
    i_syn = synaptic_current(pool, spikes)
    assert i_syn.shape == (5,)
    assert bool(jnp.all(i_syn == 0.0))


def test_synaptic_current_sums_active_slots() -> None:
    """Synaptic current sums active-slot v for firing presynapses."""
    # Hand-built pool: 2 post neurons, 3 slots each, 4 possible pre.
    pre_ids = jnp.array([[0, 1, 2], [1, 1, 3]], dtype=jnp.int32)
    v = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    plasticity_rate = jnp.ones_like(v)
    active = jnp.array(
        [[True, True, False], [True, False, True]], dtype=jnp.bool_
    )
    release_counter = jnp.zeros_like(pre_ids)
    pool = SlotPool(pre_ids, v, plasticity_rate, active, release_counter)

    # Pre 1 and pre 3 fire.
    spikes = jnp.array([False, True, False, True], dtype=jnp.bool_)
    i_syn = synaptic_current(pool, spikes)
    # Post 0: slots bound to (0, 1, 2), active = (T, T, F); firing = (F, T, F)
    #   contributions: 0 + 2.0 + 0 = 2.0
    # Post 1: slots bound to (1, 1, 3), active = (T, F, T); firing = (T, T, T)
    #   contributions: 4.0 + 0 + 6.0 = 10.0
    assert float(i_syn[0]) == 2.0
    assert float(i_syn[1]) == 10.0


def test_effective_weights_accumulates_duplicate_bindings() -> None:
    """Multiple slots bound to the same (pre, post) sum their v values."""
    # Post 0 has two slots both bound to pre 1 with v=1.5 and v=2.5.
    pre_ids = jnp.array([[1, 1]], dtype=jnp.int32)
    v = jnp.array([[1.5, 2.5]], dtype=jnp.float32)
    plasticity_rate = jnp.ones_like(v)
    active = jnp.ones_like(v, dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = SlotPool(pre_ids, v, plasticity_rate, active, release_counter)

    w = effective_weights(pool, n_pre=3)
    assert w.shape == (1, 3)
    assert float(w[0, 0]) == 0.0
    assert float(w[0, 1]) == 4.0  # 1.5 + 2.5
    assert float(w[0, 2]) == 0.0


def test_effective_weights_matches_synaptic_current_on_random_pool() -> None:
    """Dense (W @ spikes) equals synaptic_current(pool, spikes)."""
    n_post, n_pre, k = 12, 20, 6
    pool = init_random(n_post, n_pre, k, jax.random.PRNGKey(2))
    spikes = jax.random.bernoulli(
        jax.random.PRNGKey(3), p=0.3, shape=(n_pre,)
    )
    via_pool = synaptic_current(pool, spikes)
    w = effective_weights(pool, n_pre)
    via_dense = w @ spikes.astype(jnp.float32)
    assert jnp.allclose(via_pool, via_dense, atol=1e-5)


def test_step_matches_dense_step_for_equivalent_weights() -> None:
    """Slot-pool step equals dense-weight step when both encode the same W."""
    n = 6
    k = 4
    pool = init_random(n, n, k, jax.random.PRNGKey(4))
    weights = effective_weights(pool, n)

    state = init_state(n)
    # Give the starting state some prior spikes so i_syn is non-trivial.
    state = LIFState(
        v=jnp.full((n,), V_REST_MV, dtype=jnp.float32),
        spikes=jax.random.bernoulli(
            jax.random.PRNGKey(5), p=0.5, shape=(n,)
        ),
    )
    i_ext = jnp.full((n,), 10.0, dtype=jnp.float32)

    next_pool = step(state, pool, i_ext)
    next_dense = step_dense(state, weights, i_ext)
    assert jnp.allclose(next_pool.v, next_dense.v, atol=1e-5)
    assert bool(jnp.all(next_pool.spikes == next_dense.spikes))


def test_assign_ei_identity_default_fraction() -> None:
    """Default 20% inhibitory, last indices marked."""
    n = 10
    is_inh = assign_ei_identity(n)
    assert is_inh.shape == (n,)
    assert is_inh.dtype == jnp.bool_
    # 20% of 10 = 2 inhibitory neurons, at the last indices.
    assert int(jnp.sum(is_inh)) == 2
    assert bool(is_inh[-1])
    assert bool(is_inh[-2])
    assert not bool(is_inh[-3])
    assert not bool(is_inh[0])


def test_assign_ei_identity_custom_fraction() -> None:
    """Custom fractions scale correctly."""
    is_inh_25 = assign_ei_identity(100, inhibitory_fraction=0.25)
    assert int(jnp.sum(is_inh_25)) == 25
    is_inh_50 = assign_ei_identity(100, inhibitory_fraction=0.5)
    assert int(jnp.sum(is_inh_50)) == 50


def test_synaptic_current_inhibitory_flips_sign_and_scales() -> None:
    """I-sourced contributions are negated and scaled by i_multiplier."""
    # Two-post hand-built: post 0 has slot bound to pre=0 (E), post 1
    # has slot bound to pre=1 (I). Both slots have v=0.5.
    pre_ids = jnp.array([[0], [1]], dtype=jnp.int32)
    v = jnp.full((2, 1), 0.5, dtype=jnp.float32)
    pool = SlotPool(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=jnp.ones_like(v),
        active=jnp.ones_like(v, dtype=jnp.bool_),
        release_counter=jnp.zeros_like(pre_ids),
    )
    # Pre 0 is excitatory, pre 1 is inhibitory.
    pre_is_inh = jnp.array([False, True], dtype=jnp.bool_)
    # Both pres fire.
    spikes = jnp.ones((2,), dtype=jnp.bool_)

    i_syn = synaptic_current(
        pool, spikes, pre_is_inhibitory=pre_is_inh,
        i_weight_multiplier=4.0,
    )
    # Post 0: E-sourced, contribution = 0.5 * 1 = +0.5
    # Post 1: I-sourced, contribution = 0.5 * -4.0 = -2.0
    assert float(i_syn[0]) == 0.5
    assert float(i_syn[1]) == -2.0


def test_synaptic_current_no_inhibitory_matches_default_behavior() -> None:
    """pre_is_inhibitory=None matches an all-False inhibitory array."""
    pool = init_random(8, 8, 4, jax.random.PRNGKey(100))
    spikes = jax.random.bernoulli(
        jax.random.PRNGKey(101), p=0.4, shape=(8,)
    )
    # Reference: no E/I handling.
    i_syn_default = synaptic_current(pool, spikes)
    # All-False inhibitory array with any multiplier: no I contributions
    # to scale, so result should match.
    all_excitatory = jnp.zeros((8,), dtype=jnp.bool_)
    i_syn_explicit = synaptic_current(
        pool, spikes, pre_is_inhibitory=all_excitatory,
        i_weight_multiplier=4.0,
    )
    assert jnp.allclose(i_syn_default, i_syn_explicit)


def test_simulate_equivalence_over_trace() -> None:
    """simulate() on slot pool matches simulate() on dense weights."""
    n = 5
    k = 3
    t = 50
    pool = init_random(n, n, k, jax.random.PRNGKey(6))
    weights = effective_weights(pool, n)
    state = init_state(n)
    i_ext_trace = jnp.full((t, n), 20.0, dtype=jnp.float32)

    _, spikes_pool = simulate(state, pool, i_ext_trace)
    _, spikes_dense = simulate_dense(state, weights, i_ext_trace)
    assert bool(jnp.all(spikes_pool == spikes_dense))
