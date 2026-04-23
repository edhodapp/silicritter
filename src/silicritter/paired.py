"""Paired-agent simulation: two silicritter instances with cross-connections.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 7 of the silicritter implementation ladder: two plastic slot-pool
networks coupled through cross-agent synaptic connections. Each agent's
slot pool binds pre_ids into a *combined* pre-raster of length
2 * n_neurons -- indices [0, n_neurons) refer to the agent's own
neurons (recurrent self-connections), indices [n_neurons, 2*n_neurons)
refer to the partner's neurons (cross-agent connections). This is the
minimal "paired-agent" social-dynamics primitive.

Each agent has its own private modulators (valence, adrenaline),
private plasticity state (pool, traces), and private external drive.
Agents do NOT share chemistry -- biological endocrine systems are
private per animal and silicritter mirrors that convention. Cross-agent
communication is purely through the spike raster.

Timing: a two-phase step per scan iteration. Phase 1 computes both
agents' new LIF spike_posts using each agent's previous-step combined
pre-raster. Phase 2 runs each agent's STDP update using the newly
computed combined spike_post as the pre-spike source for the traces.
The phases run in parallel (no sequential dependency within a phase);
this matches the biological view that neurons fire at roughly the same
wall-clock moment without a preferred ordering.

All existing single-agent machinery (plasticity.step_plastic,
structural.apply_release, ga primitives) applies unchanged. Paired is
additive; nothing in the step 1-6 codebase is disturbed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from silicritter.lif import (
    DT_MS,
    TAU_M_MS,
    LIFState,
    init_state,
    integrate_and_spike,
)
from silicritter.plasticity import (
    GAIN_MODULATORS,
    GainMode,
    PlasticNetState,
    STDPParams,
    init_traces,
    stdp_update,
)
from silicritter.slotpool import SlotPool, init_random, synaptic_current
from silicritter.structural import StructuralParams, apply_release


class PairedState(NamedTuple):
    """Two paired-agent PlasticNetStates.

    Each member is a full PlasticNetState in which `pool.pre_ids` index
    into a combined pre-raster of length (2 * n_neurons). The paired
    step combines both agents' previous spikes into that raster before
    calling synaptic_current.
    """

    a: PlasticNetState
    b: PlasticNetState


def init_paired_state(
    n_neurons: int,
    slots_per_post: int,
    rng: jax.Array,
    weight_scale: float = 0.05,
) -> PairedState:
    """Build a fresh paired-agent state with independent random pools.

    Each agent has `n_neurons` neurons and `slots_per_post` slots per
    neuron; slot pre_ids are drawn uniformly from [0, 2*n_neurons).
    Trace vectors are sized for the combined pre-raster (pre length =
    2*n_neurons) and own post-raster (post length = n_neurons).
    """
    k_a, k_b = jax.random.split(rng, 2)
    pool_a = init_random(
        n_post=n_neurons,
        n_pre=2 * n_neurons,
        slots_per_post=slots_per_post,
        rng=k_a,
        weight_scale=weight_scale,
    )
    pool_b = init_random(
        n_post=n_neurons,
        n_pre=2 * n_neurons,
        slots_per_post=slots_per_post,
        rng=k_b,
        weight_scale=weight_scale,
    )
    return PairedState(
        a=PlasticNetState(
            lif=init_state(n_neurons),
            pool=pool_a,
            traces=init_traces(n_pre=2 * n_neurons, n_post=n_neurons),
        ),
        b=PlasticNetState(
            lif=init_state(n_neurons),
            pool=pool_b,
            traces=init_traces(n_pre=2 * n_neurons, n_post=n_neurons),
        ),
    )


def _lif_forward(
    own_state: PlasticNetState,
    combined_prev: jax.Array,
    i_ext: jax.Array,
    adrenaline: jax.Array,
    gain_mode: GainMode,
    combined_is_inhibitory: jax.Array | None,
    i_weight_multiplier: float,
    dt_ms: float,
    tau_m_ms: float,
) -> LIFState:
    """Run one agent's LIF forward pass under the combined pre-raster."""
    i_syn = synaptic_current(
        own_state.pool,
        combined_prev,
        pre_is_inhibitory=combined_is_inhibitory,
        i_weight_multiplier=i_weight_multiplier,
    )
    base_i_total = i_ext + i_syn
    modulator = GAIN_MODULATORS[gain_mode]
    i_total, tau_eff = modulator(base_i_total, adrenaline, tau_m_ms, dt_ms)
    v_next, spike_post = integrate_and_spike(
        own_state.lif.v, i_total, dt_ms, tau_eff
    )
    return LIFState(v=v_next, spikes=spike_post)


def _apply_agent_plasticity(
    own_state: PlasticNetState,
    new_lif: LIFState,
    combined_new_spikes: jax.Array,
    valence: jax.Array,
    params: STDPParams,
    structural_params: StructuralParams | None,
    dt_ms: float,
) -> PlasticNetState:
    """Run STDP update and optional structural release for one agent."""
    new_pool, new_traces = stdp_update(
        own_state.pool,
        own_state.traces,
        pre_spike_source=combined_new_spikes,
        post_spikes=new_lif.spikes,
        valence=valence,
        params=params,
        dt_ms=dt_ms,
    )
    if structural_params is not None:
        new_pool = apply_release(new_pool, structural_params)
    return PlasticNetState(lif=new_lif, pool=new_pool, traces=new_traces)


def _combine_ei_if_set(
    own_ei: jax.Array | None,
    partner_ei: jax.Array | None,
) -> jax.Array | None:
    """Return combined [own, partner] E/I array, or None if either missing."""
    if own_ei is None or partner_ei is None:
        return None
    return jnp.concatenate([own_ei, partner_ei])


def step_paired(
    state: PairedState,
    i_ext_a: jax.Array,
    i_ext_b: jax.Array,
    valence_a: jax.Array,
    valence_b: jax.Array,
    adrenaline_a: jax.Array,
    adrenaline_b: jax.Array,
    stdp_params: STDPParams,
    gain_mode: GainMode = "multiplicative",
    structural_params: StructuralParams | None = None,
    a_is_inhibitory: jax.Array | None = None,
    b_is_inhibitory: jax.Array | None = None,
    i_weight_multiplier: float = 4.0,
    dt_ms: float = DT_MS,
    tau_m_ms: float = TAU_M_MS,
) -> PairedState:
    """Advance a paired-agent simulation by one timestep.

    Phase 1 computes both agents' new spike_posts in parallel using
    each agent's previous-step combined pre-raster [own, partner].
    Phase 2 runs each agent's STDP update using the newly computed
    combined spike_post as the pre-spike source.

    E/I substrate is opt-in. When `a_is_inhibitory` and
    `b_is_inhibitory` are provided (bool arrays of length N_A, N_B),
    synaptic_current applies the cortical-balanced-network convention:
    contributions sourced from inhibitory pre-neurons are negated and
    scaled by `i_weight_multiplier` (default 4.0). When either E/I
    array is None, every pre is treated as excitatory and behavior
    is byte-exact to the step 2-8 code path.
    """
    # Phase 1: combined prev spikes for each agent's synaptic input.
    combined_prev_a = jnp.concatenate(
        [state.a.lif.spikes, state.b.lif.spikes]
    )
    combined_prev_b = jnp.concatenate(
        [state.b.lif.spikes, state.a.lif.spikes]
    )
    combined_ei_a = _combine_ei_if_set(a_is_inhibitory, b_is_inhibitory)
    combined_ei_b = _combine_ei_if_set(b_is_inhibitory, a_is_inhibitory)
    new_lif_a = _lif_forward(
        state.a, combined_prev_a, i_ext_a, adrenaline_a,
        gain_mode, combined_ei_a, i_weight_multiplier, dt_ms, tau_m_ms,
    )
    new_lif_b = _lif_forward(
        state.b, combined_prev_b, i_ext_b, adrenaline_b,
        gain_mode, combined_ei_b, i_weight_multiplier, dt_ms, tau_m_ms,
    )

    # Phase 2: combined new spikes for STDP pre-spike source.
    combined_new_a = jnp.concatenate(
        [new_lif_a.spikes, new_lif_b.spikes]
    )
    combined_new_b = jnp.concatenate(
        [new_lif_b.spikes, new_lif_a.spikes]
    )
    new_a = _apply_agent_plasticity(
        state.a, new_lif_a, combined_new_a, valence_a,
        stdp_params, structural_params, dt_ms,
    )
    new_b = _apply_agent_plasticity(
        state.b, new_lif_b, combined_new_b, valence_b,
        stdp_params, structural_params, dt_ms,
    )

    return PairedState(a=new_a, b=new_b)


class _PairedDrive(NamedTuple):
    """Per-step drive bundle for both agents (scan input)."""

    i_ext_a: jax.Array
    i_ext_b: jax.Array
    valence_a: jax.Array
    valence_b: jax.Array
    adrenaline_a: jax.Array
    adrenaline_b: jax.Array


def simulate_paired(
    initial_state: PairedState,
    i_ext_a_trace: jax.Array,
    i_ext_b_trace: jax.Array,
    valence_a_trace: jax.Array,
    valence_b_trace: jax.Array,
    adrenaline_a_trace: jax.Array,
    adrenaline_b_trace: jax.Array,
    stdp_params: STDPParams,
    gain_mode: GainMode = "multiplicative",
    structural_params: StructuralParams | None = None,
    a_is_inhibitory: jax.Array | None = None,
    b_is_inhibitory: jax.Array | None = None,
    i_weight_multiplier: float = 4.0,
) -> tuple[PairedState, jax.Array, jax.Array]:
    """Simulate a paired-agent sim over T steps.

    Returns the final state plus both agents' spike traces, each of
    shape (T, n_neurons). E/I substrate is opt-in via the
    `*_is_inhibitory` arguments; see step_paired docstring.
    """

    def scan_step(
        carry: PairedState, drive: _PairedDrive,
    ) -> tuple[PairedState, tuple[jax.Array, jax.Array]]:
        next_state = step_paired(
            carry,
            drive.i_ext_a, drive.i_ext_b,
            drive.valence_a, drive.valence_b,
            drive.adrenaline_a, drive.adrenaline_b,
            stdp_params,
            gain_mode=gain_mode,
            structural_params=structural_params,
            a_is_inhibitory=a_is_inhibitory,
            b_is_inhibitory=b_is_inhibitory,
            i_weight_multiplier=i_weight_multiplier,
        )
        return next_state, (next_state.a.lif.spikes, next_state.b.lif.spikes)

    drive = _PairedDrive(
        i_ext_a=i_ext_a_trace,
        i_ext_b=i_ext_b_trace,
        valence_a=valence_a_trace,
        valence_b=valence_b_trace,
        adrenaline_a=adrenaline_a_trace,
        adrenaline_b=adrenaline_b_trace,
    )
    final_state, (spikes_a, spikes_b) = jax.lax.scan(
        scan_step, initial_state, drive
    )
    return final_state, spikes_a, spikes_b


def make_pool_for_partner(
    n_neurons: int,
    slots_per_post: int,
    rng: jax.Array,
    weight_scale: float = 0.05,
) -> SlotPool:
    """Build a single agent's slot pool for the paired pre-space.

    Helper for experiments that want to construct one agent's pool
    with the combined pre-space of size 2*n_neurons without building a
    full PairedState.
    """
    return init_random(
        n_post=n_neurons,
        n_pre=2 * n_neurons,
        slots_per_post=slots_per_post,
        rng=rng,
        weight_scale=weight_scale,
    )
