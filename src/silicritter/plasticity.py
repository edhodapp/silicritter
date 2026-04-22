"""Three-factor STDP + valence broadcast on slot-pool synapses.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 4 of the silicritter implementation ladder: add weight plasticity
on top of the static slot-pool representation from step 3. This is
weight plasticity only - slot acquisition, release, and eviction
(structural plasticity proper) land in step 4.5 / step 5.

The plasticity rule is online spike-timing-dependent (Song, Miller,
Abbott 2000) with pre-decayed eligibility traces, modulated by a
scalar valence signal (standard three-factor formulation):

    dv_slot = valence * plasticity_rate_slot * (
        A_plus  * post_spike_t * pre_trace_{t-}
      - A_minus * pre_spike_t  * post_trace_{t-}
    )

where `*_trace_{t-}` denotes the trace value just *before* this step's
spike contribution is added -- this is the standard Song/Miller/Abbott
convention that prevents a coincident pre+post spike from getting an
extra `A_plus` of LTP by looking at a trace that already folds in the
current-step pre spike.

Traces decay with per-population time constants, then incorporate the
current step's spike after the weight update:

    pre_trace(t)  = pre_trace(t-1)  * exp(-dt/tau_pre)  + pre_spike_t
    post_trace(t) = post_trace(t-1) * exp(-dt/tau_post) + post_spike_t

Weight v is clipped to [v_min, v_max] each step. Slots with
plasticity_rate = 0 are innate / hardwired and do not update. Slots
with v at v_min are functionally silent (contribute zero synaptic
current) but still occupy the slot structurally; actual slot release
lives in the structural-plasticity pass still to come.

Population assumption: this module targets a single-population
recurrent network, where the pre and post populations are the same
set of neurons (N_pre == N_post). The pre-trace and post-trace are
therefore both over the recurrent raster, and both are fed by
`spike_post` from the LIF step. `step_plastic` asserts dimensional
consistency; a separate-pre-population variant will need a broader
API that accepts distinct spike rasters.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from silicritter.lif import DT_MS, TAU_M_MS, LIFState, integrate_and_spike
from silicritter.slotpool import SlotPool, synaptic_current


class STDPParams(NamedTuple):
    """Hyperparameters for the three-factor STDP rule."""

    tau_pre_ms: float
    tau_post_ms: float
    a_plus: float
    a_minus: float
    v_min: float
    v_max: float


class Traces(NamedTuple):
    """Synaptic eligibility traces for STDP.

    Attributes:
        pre: per-presynaptic-neuron trace, shape (N_pre,), float32.
        post: per-postsynaptic-neuron trace, shape (N_post,), float32.
    """

    pre: jax.Array
    post: jax.Array


class PlasticNetState(NamedTuple):
    """Combined per-step state of a plastic slot-pool network."""

    lif: LIFState
    pool: SlotPool
    traces: Traces


def default_params() -> STDPParams:
    """Return a set of biologically-plausible default STDP parameters."""
    return STDPParams(
        tau_pre_ms=20.0,
        tau_post_ms=20.0,
        a_plus=0.01,
        a_minus=0.012,
        v_min=0.0,
        v_max=0.5,
    )


def init_traces(n_pre: int, n_post: int) -> Traces:
    """Zero traces for a fresh simulation."""
    return Traces(
        pre=jnp.zeros((n_pre,), dtype=jnp.float32),
        post=jnp.zeros((n_post,), dtype=jnp.float32),
    )


def step_plastic(
    state: PlasticNetState,
    i_ext: jax.Array,
    valence: jax.Array,
    params: STDPParams,
    dt_ms: float = DT_MS,
    tau_m_ms: float = TAU_M_MS,
) -> PlasticNetState:
    """Advance one timestep of a plastic slot-pool network.

    Args:
        state: current PlasticNetState.
        i_ext: external current drive this step, shape (N_post,).
        valence: scalar three-factor modulator for this step. Positive
            values reinforce the STDP sign convention (LTP on pre-then-
            post correlations, LTD on post-then-pre). Negative values
            scale *and* sign-flip the update, so the asymmetry between
            a_plus and a_minus carries through with reversed sign.
            Zero gates all plasticity off.
        params: STDP hyperparameters.
        dt_ms: integration timestep in ms.
        tau_m_ms: LIF membrane time constant in ms.

    Returns:
        Next PlasticNetState (new LIF, new pool.v, new traces).
    """
    # Single-population recurrent assumption: pre_ids index into the
    # recurrent raster, which has the same length as the trace vectors.
    # This assertion runs at JIT-trace time (shape check) and is free
    # at runtime.
    assert state.traces.pre.shape[0] == state.traces.post.shape[0], (
        "plasticity.step_plastic assumes a single-population recurrent "
        "network; pre and post trace vectors must have the same length."
    )

    prev_spikes = state.lif.spikes
    i_syn = synaptic_current(state.pool, prev_spikes)
    i_total = i_ext + i_syn
    v_next, spike_post = integrate_and_spike(
        state.lif.v, i_total, dt_ms, tau_m_ms
    )
    new_lif = LIFState(v=v_next, spikes=spike_post)

    spike_post_f = spike_post.astype(jnp.float32)
    decay_pre = jnp.exp(-dt_ms / params.tau_pre_ms)
    decay_post = jnp.exp(-dt_ms / params.tau_post_ms)
    # Pre-decayed traces: the trace value at the moment of this step's
    # spike, before the step's own spike is folded in. Using these in
    # the STDP update follows Song/Miller/Abbott and avoids the
    # coincident-spike inflation a post-increment trace would produce.
    pre_decayed = state.traces.pre * decay_pre
    post_decayed = state.traces.post * decay_post

    pre_trace_slot = pre_decayed[state.pool.pre_ids]
    post_trace_slot = post_decayed[:, None]
    pre_spike_slot = spike_post_f[state.pool.pre_ids]
    post_spike_slot = spike_post_f[:, None]

    ltp = params.a_plus * post_spike_slot * pre_trace_slot
    ltd = params.a_minus * pre_spike_slot * post_trace_slot
    dv = (ltp - ltd) * valence * state.pool.plasticity_rate
    dv = jnp.where(state.pool.active, dv, jnp.float32(0.0))

    new_v = jnp.clip(state.pool.v + dv, params.v_min, params.v_max)
    new_pool = state.pool._replace(v=new_v)

    # Finally fold this step's spike into the traces for the next step.
    new_traces = Traces(
        pre=pre_decayed + spike_post_f,
        post=post_decayed + spike_post_f,
    )

    return PlasticNetState(lif=new_lif, pool=new_pool, traces=new_traces)


def simulate_plastic(
    initial_state: PlasticNetState,
    i_ext_trace: jax.Array,
    valence_trace: jax.Array,
    params: STDPParams,
) -> tuple[PlasticNetState, jax.Array]:
    """Simulate a plastic slot-pool network over a drive + valence trace.

    Args:
        initial_state: starting PlasticNetState.
        i_ext_trace: external current per timestep, shape (T, N_post).
        valence_trace: scalar valence per timestep, shape (T,).
        params: STDP hyperparameters.

    Returns:
        final_state: PlasticNetState after T steps.
        spike_trace: boolean spikes per timestep, shape (T, N_post).
    """

    def scan_step(
        state: PlasticNetState,
        drive: tuple[jax.Array, jax.Array],
    ) -> tuple[PlasticNetState, jax.Array]:
        i_ext_t, valence_t = drive
        next_state = step_plastic(state, i_ext_t, valence_t, params)
        return next_state, next_state.lif.spikes

    final_state, spike_trace = jax.lax.scan(
        scan_step, initial_state, (i_ext_trace, valence_trace)
    )
    return final_state, spike_trace
