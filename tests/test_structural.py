"""Behavioral tests for structural plasticity (slot release).

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.lif import LIFState, init_state
from silicritter.plasticity import (
    PlasticNetState,
    Traces,
    default_params,
    init_traces,
    simulate_plastic,
    step_plastic,
)
from silicritter.slotpool import SlotPool, init_random
from silicritter.structural import (
    StructuralParams,
    apply_acquisition,
    apply_release,
    default_structural_params,
)


def _acq_params(
    prob: float = 1.0,
    initial_v: float = 0.2,
    plasticity_rate: float = 1.0,
) -> StructuralParams:
    """Acquisition-focused params with release disabled."""
    return StructuralParams(
        v_release_threshold=0.0,
        release_dwell_steps=1_000_000,
        acquisition_prob=prob,
        acquisition_initial_v=initial_v,
        acquisition_plasticity_rate=plasticity_rate,
    )


def _pool_with(
    pre_ids: jax.Array,
    v: jax.Array,
    plasticity_rate: jax.Array,
    active: jax.Array,
    release_counter: jax.Array,
) -> SlotPool:
    return SlotPool(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=plasticity_rate,
        active=active,
        release_counter=release_counter,
    )


def test_default_structural_params_in_reasonable_range() -> None:
    """default_structural_params returns non-pathological defaults."""
    params = default_structural_params()
    assert params.v_release_threshold > 0.0
    assert params.v_release_threshold < 0.1
    assert params.release_dwell_steps > 0
    assert params.release_dwell_steps < 10_000


def test_apply_release_increments_counter_below_threshold() -> None:
    """A slot with v below threshold has its counter incremented by one."""
    pre_ids = jnp.array([[0, 1]], dtype=jnp.int32)
    v = jnp.array([[0.005, 0.5]], dtype=jnp.float32)
    plasticity_rate = jnp.ones_like(v)
    active = jnp.ones_like(v, dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = StructuralParams(
        v_release_threshold=0.01, release_dwell_steps=100
    )
    new_pool = apply_release(pool, params)
    # Slot 0 (v=0.005 < 0.01): counter increments to 1.
    # Slot 1 (v=0.5): counter stays at 0.
    assert int(new_pool.release_counter[0, 0]) == 1
    assert int(new_pool.release_counter[0, 1]) == 0
    # Neither slot should be released yet.
    assert bool(jnp.all(new_pool.active))


def test_apply_release_resets_counter_when_v_rises() -> None:
    """A slot whose v crosses back above threshold has its counter cleared."""
    pre_ids = jnp.array([[0]], dtype=jnp.int32)
    v = jnp.array([[0.2]], dtype=jnp.float32)
    plasticity_rate = jnp.ones_like(v)
    active = jnp.ones_like(v, dtype=jnp.bool_)
    # Pre-existing counter at 50 (half-way to release).
    release_counter = jnp.full_like(pre_ids, 50)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = StructuralParams(
        v_release_threshold=0.01, release_dwell_steps=100
    )
    new_pool = apply_release(pool, params)
    assert int(new_pool.release_counter[0, 0]) == 0
    assert bool(new_pool.active[0, 0])


def test_apply_release_deactivates_slot_at_dwell() -> None:
    """A slot whose counter reaches dwell is released: active=False, v=0."""
    pre_ids = jnp.array([[7]], dtype=jnp.int32)
    v = jnp.array([[0.005]], dtype=jnp.float32)
    plasticity_rate = jnp.full_like(v, 0.5)
    active = jnp.ones_like(v, dtype=jnp.bool_)
    # Counter at dwell - 1 so the next step releases.
    release_counter = jnp.full_like(pre_ids, 99)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = StructuralParams(
        v_release_threshold=0.01, release_dwell_steps=100
    )
    new_pool = apply_release(pool, params)
    assert not bool(new_pool.active[0, 0])
    assert float(new_pool.v[0, 0]) == 0.0
    assert int(new_pool.release_counter[0, 0]) == 0
    # pre_ids and plasticity_rate are preserved across release.
    assert int(new_pool.pre_ids[0, 0]) == 7
    assert float(new_pool.plasticity_rate[0, 0]) == 0.5


def test_apply_release_ignores_inactive_slots() -> None:
    """Already-inactive slots stay inactive; their counter does not grow."""
    pre_ids = jnp.array([[0, 1]], dtype=jnp.int32)
    v = jnp.array([[0.0, 0.005]], dtype=jnp.float32)
    plasticity_rate = jnp.ones_like(v)
    active = jnp.array([[False, True]], dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = StructuralParams(
        v_release_threshold=0.01, release_dwell_steps=100
    )
    new_pool = apply_release(pool, params)
    assert int(new_pool.release_counter[0, 0]) == 0
    assert int(new_pool.release_counter[0, 1]) == 1


def test_step_plastic_without_structural_params_is_identity_on_pool() -> None:
    """structural_params=None preserves step-4/5 byte-exact behaviour."""
    n = 4
    pool = init_random(n, n, 3, jax.random.PRNGKey(40))
    state_with_structural = PlasticNetState(
        lif=init_state(n),
        pool=pool,
        traces=init_traces(n, n),
    )
    state_without = PlasticNetState(
        lif=init_state(n),
        pool=pool,
        traces=init_traces(n, n),
    )
    i_ext = jnp.full((n,), 20.0, dtype=jnp.float32)
    stdp_params = default_params()

    next_a = step_plastic(
        state_with_structural,
        i_ext,
        jnp.float32(1.0),
        jnp.float32(1.0),
        stdp_params,
    )
    next_b = step_plastic(
        state_without,
        i_ext,
        jnp.float32(1.0),
        jnp.float32(1.0),
        stdp_params,
        structural_params=None,
    )
    assert bool(jnp.all(next_a.pool.v == next_b.pool.v))
    assert bool(jnp.all(next_a.pool.active == next_b.pool.active))


def test_simulate_plastic_releases_weak_slots_over_long_sim() -> None:
    """With LTD-biased drive, a meaningful fraction of slots release."""
    n = 6
    t = 400
    pool = init_random(n, n, 4, jax.random.PRNGKey(41))
    state = PlasticNetState(
        lif=init_state(n),
        pool=pool,
        traces=init_traces(n, n),
    )
    i_ext_trace = jnp.full((t, n), 20.0, dtype=jnp.float32)
    valence_trace = -jnp.ones((t,), dtype=jnp.float32)
    adrenaline_trace = jnp.ones((t,), dtype=jnp.float32)
    params = default_params()
    structural_params = StructuralParams(
        v_release_threshold=0.01, release_dwell_steps=50
    )

    final_state, _ = simulate_plastic(
        state,
        i_ext_trace,
        valence_trace,
        adrenaline_trace,
        params,
        structural_params=structural_params,
    )
    initial_active = int(pool.active.sum())
    final_active = int(final_state.pool.active.sum())
    released = initial_active - final_active
    # Under LTD-biased drive with dwell=50 and 400 steps, expect at
    # least 3 slots to release out of 24 total. Tight enough to catch
    # a mechanism regression where release-dwell never triggers, loose
    # enough to absorb noise and a_plus / a_minus asymmetry effects.
    assert released >= 3, f"expected >= 3 releases, got {released}"


def test_step_plastic_ordering_stdp_before_release() -> None:
    """Release check sees POST-STDP v, not PRE-STDP v.

    Constructs a minimal scenario where a slot's v is sub-threshold
    *before* the STDP update but is lifted above threshold *by* the
    STDP update on this step. With the current STDP-first-then-release
    ordering, the slot's release_counter should stay at 0 (because the
    post-update v > threshold). If the ordering were reversed (release
    first, STDP second), the counter would increment to 1 this step.

    This test *pins* the architectural choice: swapping the order of
    `apply_release` and the weight update in step_plastic will fail
    this test. Without it, the ordering is architecturally free.
    """
    # Two-cell net; slot (post=1, k=0) bound to pre=0.
    pre_ids = jnp.array([[1], [0]], dtype=jnp.int32)
    # v=0.005 is sub-threshold (below v_release_threshold=0.01).
    v_init = jnp.array([[0.5], [0.005]], dtype=jnp.float32)
    plasticity_rate = jnp.ones_like(v_init)
    active = jnp.ones_like(v_init, dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = _pool_with(
        pre_ids, v_init, plasticity_rate, active, release_counter,
    )

    lif = LIFState(
        v=jnp.array([-65.0, -51.0], dtype=jnp.float32),
        spikes=jnp.array([False, False], dtype=jnp.bool_),
    )
    # Pre_trace at pre=0 primed high so the LTP increment on slot (1, 0)
    # this step is large: dv = a_plus * pre_trace_decayed ~ 0.01 * 0.86
    # = 0.0086, which lifts v from 0.005 to 0.0136 > threshold 0.01.
    traces = Traces(
        pre=jnp.array([0.9, 0.0], dtype=jnp.float32),
        post=jnp.zeros((2,), dtype=jnp.float32),
    )
    state = PlasticNetState(lif=lif, pool=pool, traces=traces)
    stdp_params = default_params()
    structural_params = StructuralParams(
        v_release_threshold=0.01, release_dwell_steps=100
    )

    # Post=1 is about to spike (V = -51 + dv = -49.25 on this drive).
    i_ext = jnp.array([0.0, 50.0], dtype=jnp.float32)
    nxt = step_plastic(
        state,
        i_ext,
        jnp.float32(1.0),
        jnp.float32(1.0),
        stdp_params,
        structural_params=structural_params,
    )
    # Sanity: post=1 spiked and slot (1, 0)'s v did grow.
    assert bool(nxt.lif.spikes[1])
    assert float(nxt.pool.v[1, 0]) > 0.01, (
        "STDP did not lift v above threshold; test setup is broken"
    )
    # Ordering pin: release_counter stayed at 0 because post-update v
    # exceeds threshold. Would be 1 if release ran before STDP.
    assert int(nxt.pool.release_counter[1, 0]) == 0


def test_apply_release_boundary_at_threshold() -> None:
    """v == v_release_threshold exactly does NOT count toward release.

    The release rule uses strict `<`, so a slot sitting exactly at the
    threshold is above (by the rule's definition) and its counter
    stays at 0.
    """
    pre_ids = jnp.array([[0]], dtype=jnp.int32)
    v = jnp.array([[0.01]], dtype=jnp.float32)
    plasticity_rate = jnp.ones_like(v)
    active = jnp.ones_like(v, dtype=jnp.bool_)
    release_counter = jnp.full_like(pre_ids, 50)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = StructuralParams(
        v_release_threshold=0.01, release_dwell_steps=100
    )
    new_pool = apply_release(pool, params)
    assert int(new_pool.release_counter[0, 0]) == 0


def test_default_structural_params_has_zero_acquisition() -> None:
    """Backward-compat: default params disable acquisition."""
    params = default_structural_params()
    assert params.acquisition_prob == 0.0


def test_apply_acquisition_leaves_active_slots_untouched() -> None:
    """Active slots are not candidates for rebinding."""
    pre_ids = jnp.array([[7, 8, 9]], dtype=jnp.int32)
    v = jnp.array([[0.5, 0.5, 0.5]], dtype=jnp.float32)
    plasticity_rate = jnp.full_like(v, 0.3)
    active = jnp.ones_like(v, dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = _acq_params(prob=1.0, initial_v=0.2, plasticity_rate=1.0)
    new_pool = apply_acquisition(
        pool, params, jax.random.PRNGKey(0), n_pre=16,
    )
    assert bool(jnp.all(new_pool.pre_ids == pool.pre_ids))
    assert bool(jnp.all(new_pool.v == pool.v))
    assert bool(jnp.all(new_pool.plasticity_rate == pool.plasticity_rate))
    assert bool(jnp.all(new_pool.active))


def test_apply_acquisition_zero_prob_is_noop() -> None:
    """prob=0 preserves the pool exactly, even when inactive slots exist."""
    pre_ids = jnp.array([[0, 1]], dtype=jnp.int32)
    v = jnp.array([[0.0, 0.5]], dtype=jnp.float32)
    plasticity_rate = jnp.array([[0.0, 0.3]], dtype=jnp.float32)
    active = jnp.array([[False, True]], dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = _acq_params(prob=0.0)
    new_pool = apply_acquisition(
        pool, params, jax.random.PRNGKey(0), n_pre=16,
    )
    assert bool(jnp.all(new_pool.pre_ids == pool.pre_ids))
    assert bool(jnp.all(new_pool.v == pool.v))
    assert bool(jnp.all(new_pool.active == pool.active))


def test_apply_acquisition_prob_one_rebinds_all_inactive() -> None:
    """With prob=1, every inactive slot gets rebound."""
    pre_ids = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    v = jnp.array([[0.0, 0.0, 0.5]], dtype=jnp.float32)
    plasticity_rate = jnp.array([[0.0, 0.0, 0.3]], dtype=jnp.float32)
    active = jnp.array([[False, False, True]], dtype=jnp.bool_)
    release_counter = jnp.array([[0, 0, 0]], dtype=jnp.int32)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = _acq_params(prob=1.0, initial_v=0.25, plasticity_rate=0.7)
    new_pool = apply_acquisition(
        pool, params, jax.random.PRNGKey(1), n_pre=32,
    )
    assert bool(new_pool.active[0, 0])
    assert bool(new_pool.active[0, 1])
    assert abs(float(new_pool.v[0, 0]) - 0.25) < 1e-5
    assert abs(float(new_pool.v[0, 1]) - 0.25) < 1e-5
    assert abs(float(new_pool.plasticity_rate[0, 0]) - 0.7) < 1e-5
    assert abs(float(new_pool.plasticity_rate[0, 1]) - 0.7) < 1e-5
    # Slot 2 (already active) is untouched.
    assert float(new_pool.v[0, 2]) == 0.5
    assert abs(float(new_pool.plasticity_rate[0, 2]) - 0.3) < 1e-5
    assert int(new_pool.pre_ids[0, 0]) >= 0
    assert int(new_pool.pre_ids[0, 0]) < 32
    assert int(new_pool.pre_ids[0, 1]) >= 0
    assert int(new_pool.pre_ids[0, 1]) < 32


def test_apply_acquisition_resets_release_counter() -> None:
    """A rebound slot restarts its release counter at zero."""
    pre_ids = jnp.array([[0]], dtype=jnp.int32)
    v = jnp.array([[0.0]], dtype=jnp.float32)
    plasticity_rate = jnp.array([[0.0]], dtype=jnp.float32)
    active = jnp.array([[False]], dtype=jnp.bool_)
    release_counter = jnp.array([[42]], dtype=jnp.int32)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = _acq_params(prob=1.0)
    new_pool = apply_acquisition(
        pool, params, jax.random.PRNGKey(2), n_pre=8,
    )
    assert int(new_pool.release_counter[0, 0]) == 0


def test_apply_acquisition_deterministic_rng() -> None:
    """Same rng and pool gives byte-identical output."""
    pre_ids = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    v = jnp.zeros_like(pre_ids, dtype=jnp.float32)
    plasticity_rate = jnp.zeros_like(v)
    active = jnp.zeros_like(v, dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = _acq_params(prob=0.5)
    new_pool_a = apply_acquisition(
        pool, params, jax.random.PRNGKey(3), n_pre=16,
    )
    new_pool_b = apply_acquisition(
        pool, params, jax.random.PRNGKey(3), n_pre=16,
    )
    assert bool(jnp.all(new_pool_a.pre_ids == new_pool_b.pre_ids))
    assert bool(jnp.all(new_pool_a.active == new_pool_b.active))


def test_apply_acquisition_uniform_pre_id_coverage() -> None:
    """Uniform-random acquisition produces ~uniform pre_id distribution.

    With n_pre=10 and many inactive slots all acquiring, the sampled
    pre_ids should cover the full range roughly uniformly.
    """
    n_post = 200
    k_slots = 16
    n_pre = 10
    pre_ids = jnp.zeros((n_post, k_slots), dtype=jnp.int32)
    v = jnp.zeros((n_post, k_slots), dtype=jnp.float32)
    plasticity_rate = jnp.zeros_like(v)
    active = jnp.zeros_like(v, dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = _acq_params(prob=1.0)
    new_pool = apply_acquisition(
        pool, params, jax.random.PRNGKey(4), n_pre=n_pre,
    )
    counts = jnp.bincount(new_pool.pre_ids.flatten(), length=n_pre)
    min_count = int(counts.min())
    max_count = int(counts.max())
    assert min_count > 200
    assert max_count < 500


def test_apply_acquisition_hebbian_bias_favors_active_pre() -> None:
    """With pre_activity concentrated on a subset, the sampled pre_ids
    are biased toward that subset.
    """
    n_post = 100
    k_slots = 16
    n_pre = 20
    pre_activity = jnp.concatenate([
        jnp.full((15,), 0.01, dtype=jnp.float32),
        jnp.full((5,), 1.0, dtype=jnp.float32),
    ])
    pre_ids = jnp.zeros((n_post, k_slots), dtype=jnp.int32)
    v = jnp.zeros((n_post, k_slots), dtype=jnp.float32)
    plasticity_rate = jnp.zeros_like(v)
    active = jnp.zeros_like(v, dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    params = _acq_params(prob=1.0)
    new_pool = apply_acquisition(
        pool, params, jax.random.PRNGKey(5), n_pre=n_pre,
        pre_activity=pre_activity,
    )
    n_total = n_post * k_slots
    n_hot = int((new_pool.pre_ids >= 15).sum())
    frac_hot = n_hot / n_total
    # Expected ~5*1.0 / (15*0.01 + 5*1.0) = 5/5.15 ~ 0.97. Allow 0.90.
    assert frac_hot > 0.90


def test_apply_acquisition_shape_preservation() -> None:
    """Output arrays have the same shape as input."""
    n_post = 8
    k_slots = 4
    pre_ids = jnp.arange(n_post * k_slots, dtype=jnp.int32).reshape(
        n_post, k_slots
    )
    v = jnp.full_like(pre_ids, 0.0, dtype=jnp.float32)
    plasticity_rate = jnp.zeros_like(v)
    active = jnp.zeros_like(v, dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids)
    pool = _pool_with(pre_ids, v, plasticity_rate, active, release_counter)
    new_pool = apply_acquisition(
        pool, _acq_params(prob=1.0), jax.random.PRNGKey(6), n_pre=16,
    )
    assert new_pool.pre_ids.shape == pool.pre_ids.shape
    assert new_pool.v.shape == pool.v.shape
    assert new_pool.plasticity_rate.shape == pool.plasticity_rate.shape
    assert new_pool.active.shape == pool.active.shape
    assert new_pool.release_counter.shape == pool.release_counter.shape
