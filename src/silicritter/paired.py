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

from typing import Literal, NamedTuple

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

# Default `adrenaline_a` for `step_paired`: scalar 1.0 means "no gain
# modulation" — agent A runs open-loop. Most experiments keep A
# open-loop and only modulate B; this sentinel makes the open-loop
# case the silent default.
ADRENALINE_OPEN_LOOP: jax.Array = jnp.float32(1.0)


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


def cross_e_partner_mask(
    pool: SlotPool,
    n_own: int,
    partner_is_inh: jax.Array,
) -> jax.Array:
    """Boolean mask: which slots in `pool` point to E neurons in the partner.

    A slot is "cross-E" iff (a) its pre_id indexes into the partner
    substrate (`pre_id >= n_own`, where pre_ids occupy the combined
    `[own, partner]` raster of length `2 * n_own`), AND (b) the partner
    neuron at that index is excitatory (`~partner_is_inh[partner_idx]`).

    Used by both step16's pool-summary diagnostic and step17's per-run
    metrics. Centralized here so the classification logic stays
    consistent across experiments — a future refactor of the rule (e.g.
    if slot pre_ids ever encode something other than indices) only
    needs to land here.

    Args:
        pool: SlotPool whose pre_ids span `[0, 2 * n_own)`.
        n_own: own-agent population size; pre_ids in `[0, n_own)` are
            self-recurrent, in `[n_own, 2 * n_own)` are cross to the
            partner.
        partner_is_inh: boolean mask of length `n_own` marking which
            partner neurons are inhibitory (True) vs excitatory (False).

    Returns:
        Boolean array shaped like `pool.pre_ids`, True where the slot
        is cross-substrate AND points to a partner E neuron.
    """
    cross_mask = pool.pre_ids >= n_own
    # Safe lookup: where !cross, partner_idx is 0 (lookup result is
    # masked out by cross_mask anyway).
    partner_idx = jnp.where(cross_mask, pool.pre_ids - n_own, 0)
    return cross_mask & ~partner_is_inh[partner_idx]


def _validate_paired_ei(
    a_is_inh: jax.Array | None,
    b_is_inh: jax.Array | None,
) -> None:
    """Reject partial-E/I (one side set, the other None).

    Either both masks are supplied (paired E/I substrate active) or
    neither (no E/I substrate); mixing has no defined semantics.
    Partial E/I is almost certainly a programmer mistake — silent
    degradation hides it; loud failure surfaces it.

    Centralized here so a single canonical error fires regardless of
    which side is missing — the prior implementation called the
    combine helper twice with swapped labels, and only the first
    call's error could ever surface (depending on call order, not
    on which side was actually missing).
    """
    a_set = a_is_inh is not None
    b_set = b_is_inh is not None
    if a_set == b_set:
        return
    supplied = "a_is_inhibitory" if a_set else "b_is_inhibitory"
    missing = "b_is_inhibitory" if a_set else "a_is_inhibitory"
    raise ValueError(
        f"partial E/I substrate: {supplied} was supplied but "
        f"{missing} is None. Pass both masks or neither — mixing "
        f"has no defined semantics for paired-agent E/I."
    )


def _combine_ei_if_set(
    own_ei: jax.Array | None,
    partner_ei: jax.Array | None,
) -> jax.Array | None:
    """Return combined [own, partner] E/I array, or None if both missing.

    Precondition: callers must have validated via `_validate_paired_ei`
    that the two arrays are either both set or both None. This helper
    no longer raises — that responsibility moved up to `step_paired`
    where it can fire ONCE with both labels available.
    """
    if own_ei is not None and partner_ei is not None:
        return jnp.concatenate([own_ei, partner_ei])
    return None


def step_paired(
    state: PairedState,
    i_ext_a: jax.Array,
    i_ext_b: jax.Array,
    valence_a: jax.Array,
    valence_b: jax.Array,
    *,
    adrenaline_b: jax.Array,
    stdp_params: STDPParams,
    adrenaline_a: jax.Array = ADRENALINE_OPEN_LOOP,
    gain_mode: GainMode = "multiplicative",
    structural_params: StructuralParams | None = None,
    a_is_inhibitory: jax.Array | None = None,
    b_is_inhibitory: jax.Array | None = None,
    i_weight_multiplier: float = 8.0,
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
    scaled by `i_weight_multiplier` (default 8.0 per D008). When either E/I
    array is None, every pre is treated as excitatory and behavior
    is byte-exact to the step 2-8 code path.
    """
    # Validate E/I argument set ONCE up front (before either combine
    # call) so partial-E/I produces a single canonical error regardless
    # of which side is missing.
    _validate_paired_ei(a_is_inhibitory, b_is_inhibitory)
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
    i_weight_multiplier: float = 8.0,
    output_mode: Literal["raster", "rate"] = "raster",
) -> tuple[PairedState, jax.Array, jax.Array]:
    """Simulate a paired-agent sim over T steps.

    Returns the final state plus per-step output for both agents.
    ``output_mode`` selects what each per-step output is:

    - ``"raster"`` (default): full ``(T, n_neurons)`` spike raster per
      agent. Lets callers compute per-neuron metrics, raster plots,
      etc.
    - ``"rate"``: per-step population-mean firing rate ``(T,)`` per
      agent. ``rate_x[t] == raster_x[t].mean()`` within float32
      precision. Memory-friendly for long-T sweeps - at T=10M, N=256,
      ~40 MB per scalar trace vs ~2.56 GB per raster.

    E/I substrate is opt-in via the ``*_is_inhibitory`` arguments;
    see step_paired docstring.
    """
    if output_mode not in ("raster", "rate"):
        raise ValueError(
            f"output_mode must be 'raster' or 'rate', got {output_mode!r}"
        )

    # output_mode is a Python-level constant captured by the scan_step
    # closure - the `if output_mode == "raster"` branch is resolved at
    # trace time, not under jax.lax.cond. Do NOT promote output_mode to
    # a traced (jax.Array) argument; that would convert the branch into
    # a tracer-incompatible Python equality check on a Tracer.
    def scan_step(
        carry: PairedState, drive: _PairedDrive,
    ) -> tuple[PairedState, tuple[jax.Array, jax.Array]]:
        next_state = step_paired(
            carry,
            drive.i_ext_a, drive.i_ext_b,
            drive.valence_a, drive.valence_b,
            adrenaline_a=drive.adrenaline_a,
            adrenaline_b=drive.adrenaline_b,
            stdp_params=stdp_params,
            gain_mode=gain_mode,
            structural_params=structural_params,
            a_is_inhibitory=a_is_inhibitory,
            b_is_inhibitory=b_is_inhibitory,
            i_weight_multiplier=i_weight_multiplier,
        )
        if output_mode == "raster":
            return next_state, (
                next_state.a.lif.spikes, next_state.b.lif.spikes,
            )
        return next_state, (
            next_state.a.lif.spikes.astype(jnp.float32).mean(),
            next_state.b.lif.spikes.astype(jnp.float32).mean(),
        )

    drive = _PairedDrive(
        i_ext_a=i_ext_a_trace,
        i_ext_b=i_ext_b_trace,
        valence_a=valence_a_trace,
        valence_b=valence_b_trace,
        adrenaline_a=adrenaline_a_trace,
        adrenaline_b=adrenaline_b_trace,
    )
    final_state, (out_a, out_b) = jax.lax.scan(
        scan_step, initial_state, drive,
    )
    return final_state, out_a, out_b


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
