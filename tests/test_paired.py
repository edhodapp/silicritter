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
    cross_e_partner_mask,
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
    # Bind every B slot to A-neuron 2 (an I-neuron under the
    # convention "last 50% inhibitory at n=4"), which IS firing
    # under spikes_a_forced. That makes B's synaptic current
    # depend visibly on whether E/I-handling is enabled.
    cross_pre_ids = jnp.full((n, k), n + 2, dtype=jnp.int32)
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
    a_is_inh = jnp.array([False, False, True, True], dtype=jnp.bool_)
    b_is_inh = jnp.zeros((n,), dtype=jnp.bool_)
    # Reference run: no E/I substrate. A's I-neuron contributes
    # positively to B's synaptic current (E behavior).
    nxt_no_ei = step_paired(
        state,
        i_ext_a=zero_vec, i_ext_b=zero_vec,
        valence_a=jnp.float32(0.0), valence_b=jnp.float32(0.0),
        adrenaline_b=jnp.float32(1.0),
        stdp_params=stdp_params,
    )
    # E/I run: A's I-neuron's contribution flips sign and scales by 4x.
    nxt_with_ei = step_paired(
        state,
        i_ext_a=zero_vec, i_ext_b=zero_vec,
        valence_a=jnp.float32(0.0), valence_b=jnp.float32(0.0),
        adrenaline_b=jnp.float32(1.0),
        stdp_params=stdp_params,
        a_is_inhibitory=a_is_inh,
        b_is_inhibitory=b_is_inh,
        i_weight_multiplier=4.0,
    )
    # V after one step: no-E/I gets positive dV (A-neuron 2 fires +
    # treated as E), E/I gets negative dV (same fire + treated as I).
    assert float(nxt_no_ei.b.lif.v[0]) > float(nxt_with_ei.b.lif.v[0])


def test_step_paired_partial_ei_a_only_raises_value_error() -> None:
    """Partial E/I (a set, b None) must fail loudly, not silently degrade.

    Previous behavior: `_combine_ei_if_set` returned None when either
    side was None, silently treating the whole pair as having no E/I
    info. This hides programmer mistakes (forgot to thread one of the
    masks through). The contract is now: pass both masks or neither;
    partial E/I raises ValueError with a debug-useful message.
    """
    import pytest  # pylint: disable=import-outside-toplevel
    n = 4
    k = 2
    state = init_paired_state(n, k, jax.random.PRNGKey(0))
    stdp_params = default_params()
    zero_vec = jnp.zeros((n,), dtype=jnp.float32)
    a_is_inh = jnp.array([False, False, True, True], dtype=jnp.bool_)

    with pytest.raises(ValueError) as exc_info:
        step_paired(
            state,
            i_ext_a=zero_vec, i_ext_b=zero_vec,
            valence_a=jnp.float32(0.0), valence_b=jnp.float32(0.0),
            adrenaline_a=jnp.float32(1.0), adrenaline_b=jnp.float32(1.0),
            stdp_params=stdp_params,
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=None,
        )
    msg = str(exc_info.value)
    # Error message must name the SUPPLIED side (so the user can find
    # the call in their config). Catches a regression where the
    # message drops the supplied-side name.
    assert "a_is_inhibitory" in msg, (
        f"error message must name the supplied side; got: {msg!r}"
    )
    # Error message must name the MISSING side (so the user knows
    # what to add).
    assert "b_is_inhibitory" in msg, (
        f"error message must name the missing side; got: {msg!r}"
    )
    # Error message must direct the fix.
    assert "both" in msg.lower() or "neither" in msg.lower(), (
        f"error message must direct user to pass both or neither; "
        f"got: {msg!r}"
    )


def test_step_paired_partial_ei_b_only_raises_value_error() -> None:
    """Partial E/I (b set, a None): symmetric case must also raise."""
    import pytest  # pylint: disable=import-outside-toplevel
    n = 4
    k = 2
    state = init_paired_state(n, k, jax.random.PRNGKey(0))
    stdp_params = default_params()
    zero_vec = jnp.zeros((n,), dtype=jnp.float32)
    b_is_inh = jnp.array([False, False, True, True], dtype=jnp.bool_)

    with pytest.raises(ValueError) as exc_info:
        step_paired(
            state,
            i_ext_a=zero_vec, i_ext_b=zero_vec,
            valence_a=jnp.float32(0.0), valence_b=jnp.float32(0.0),
            adrenaline_a=jnp.float32(1.0), adrenaline_b=jnp.float32(1.0),
            stdp_params=stdp_params,
            a_is_inhibitory=None,
            b_is_inhibitory=b_is_inh,
        )
    msg = str(exc_info.value)
    # Both sides named (supplied + missing) for debug completeness.
    assert "b_is_inhibitory" in msg, (
        f"error message must name the supplied side; got: {msg!r}"
    )
    assert "a_is_inhibitory" in msg, (
        f"error message must name the missing side; got: {msg!r}"
    )
    assert "both" in msg.lower() or "neither" in msg.lower(), (
        f"error message must direct user to pass both or neither; "
        f"got: {msg!r}"
    )


def test_step_paired_adrenaline_a_default_is_one() -> None:
    """step_paired's `adrenaline_a` defaults to 1.0 (no gain modulation).

    Most experiments run agent A open-loop while only B is under
    closed-loop adrenaline control. Forcing every callsite to pass
    `adrenaline_a=jnp.float32(1.0)` is busywork that hides the
    "open-loop" intent in plumbing. The default makes the open-loop
    case the silent case.

    Contract pinned: omitting `adrenaline_a` produces the same result
    as explicitly passing `1.0`.
    """
    n = 4
    k = 2
    state = init_paired_state(n, k, jax.random.PRNGKey(7))
    stdp_params = default_params()
    zero_vec = jnp.zeros((n,), dtype=jnp.float32)
    common_kwargs = {
        "i_ext_a": zero_vec, "i_ext_b": zero_vec,
        "valence_a": jnp.float32(0.0),
        "valence_b": jnp.float32(0.0),
        "adrenaline_b": jnp.float32(1.0),
        "stdp_params": stdp_params,
    }
    nxt_default = step_paired(state, **common_kwargs)
    nxt_explicit = step_paired(
        state, **common_kwargs, adrenaline_a=jnp.float32(1.0),
    )
    # Same input, default vs explicit-1.0 — outputs must match exactly.
    assert jnp.array_equal(nxt_default.a.lif.v, nxt_explicit.a.lif.v), (
        "step_paired's adrenaline_a default must equal 1.0; "
        "outputs differ from explicit 1.0 call"
    )
    assert jnp.array_equal(nxt_default.b.lif.v, nxt_explicit.b.lif.v)


def test_cross_e_partner_mask_classifies_correctly() -> None:
    """`cross_e_partner_mask` marks True iff slot is cross AND points to E.

    Constructs a 4-slot pool with pre_ids spanning the [own, partner]
    raster boundary, plus a partner_is_inh that picks out specific
    indices as I-neurons. Asserts the mask is True only for slots
    whose pre_id is in [n_own, 2*n_own) AND whose partner index is E
    (partner_is_inh[partner_idx] is False).
    """
    n_own = 4
    # pool.pre_ids: slot 0 self-recurrent E target, slot 1 self I,
    # slot 2 cross-partner E (partner index 0), slot 3 cross-partner
    # I (partner index 2). With partner_is_inh = [F, F, T, F]:
    #   slot 0: pre=1 (self), cross_mask False -> result False
    #   slot 1: pre=2 (self), cross_mask False -> result False
    #   slot 2: pre=4 (=n_own+0), cross True, partner_is_inh[0]=F -> True
    #   slot 3: pre=6 (=n_own+2), cross True, partner_is_inh[2]=T -> False
    pre_ids = jnp.array([[1, 2, 4, 6]], dtype=jnp.int32)
    pool = SlotPool(
        pre_ids=pre_ids,
        v=jnp.full((1, 4), 0.5, dtype=jnp.float32),
        plasticity_rate=jnp.ones((1, 4), dtype=jnp.float32),
        active=jnp.ones((1, 4), dtype=jnp.bool_),
        release_counter=jnp.zeros((1, 4), dtype=jnp.int32),
    )
    partner_is_inh = jnp.array(
        [False, False, True, False], dtype=jnp.bool_,
    )
    mask = cross_e_partner_mask(pool, n_own, partner_is_inh)
    assert mask.shape == pre_ids.shape
    assert not bool(mask[0, 0]), "self-recurrent E target must not be cross-E"
    assert not bool(mask[0, 1]), "self-recurrent I target must not be cross-E"
    assert bool(mask[0, 2]), "cross to partner E must be True"
    assert not bool(mask[0, 3]), "cross to partner I must be False"


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
