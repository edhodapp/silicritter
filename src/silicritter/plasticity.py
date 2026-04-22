"""Three-factor STDP + valence broadcast on slot-pool synapses.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 4 of the silicritter implementation ladder: add weight plasticity
on top of the static slot-pool representation from step 3. Weight
plasticity lives in this module; **slot release** (structural
plasticity's pruning half) is provided by
`silicritter.structural.apply_release` and landed in step 6 — it is
invoked from `step_plastic` when a `structural_params` argument is
passed. Slot **acquisition** (the exuberance / formation half) is
still to come; it requires PRNG threading through the scan carry and
lands in its own step when we need it.

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

Chemical-signal analogs: step 4 introduced one broadcast signal
(valence) as a scalar modulator gating STDP updates. Step 5 adds a
second scalar signal (adrenaline) acting as a neural-gain modulator
at the LIF integration step. The design choice is to add chemical
signals one at a time as specific experiments require them, rather
than pre-commit to a palette of modulators. Adrenaline is
biologically motivated by Aston-Jones & Cohen 2005 (noradrenergic
system raises neural gain; tonic and phasic components); the silicon
interpretation is a dedicated analog broadcast line whose value
scales effective input current to every post-neuron. A third, fourth,
etc. modulator will trigger a refactor into a Modulators struct, but
not before.
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp

from silicritter.lif import DT_MS, TAU_M_MS, LIFState, integrate_and_spike
from silicritter.slotpool import SlotPool, synaptic_current
from silicritter.structural import StructuralParams, apply_release


GainMode = Literal[
    "multiplicative",
    "multiplicative_mild",
    "additive",
    "tau_m_scale",
    "threshold_shift",
]

# Tuning constants used by the non-baseline gain mechanisms. Chosen so
# that an adrenaline range of [0.8, 1.5] (the step-5 task profile)
# produces meaningful but not catastrophic modulation.
_MILD_SENSITIVITY: float = 0.3
_ADDITIVE_OFFSET_MV: float = 5.0
_THRESHOLD_SHIFT_MV: float = 3.0


def _modulate_multiplicative(
    base_i_total: jax.Array,
    adrenaline: jax.Array,
    tau_m_ms: float,
    dt_ms: float,
) -> tuple[jax.Array, jax.Array]:
    """Step-4 baseline: i_total *= adrenaline. tau_m unchanged."""
    del dt_ms
    return base_i_total * adrenaline, jnp.asarray(tau_m_ms)


def _modulate_multiplicative_mild(
    base_i_total: jax.Array,
    adrenaline: jax.Array,
    tau_m_ms: float,
    dt_ms: float,
) -> tuple[jax.Array, jax.Array]:
    """Scaled multiplicative: i_total *= 1 + (adr - 1) * sensitivity."""
    del dt_ms
    factor = 1.0 + (adrenaline - 1.0) * _MILD_SENSITIVITY
    return base_i_total * factor, jnp.asarray(tau_m_ms)


def _modulate_additive(
    base_i_total: jax.Array,
    adrenaline: jax.Array,
    tau_m_ms: float,
    dt_ms: float,
) -> tuple[jax.Array, jax.Array]:
    """Additive bias offset: i_total += (adr - 1) * offset_mv."""
    del dt_ms
    offset = (adrenaline - 1.0) * _ADDITIVE_OFFSET_MV
    return base_i_total + offset, jnp.asarray(tau_m_ms)


def _modulate_tau_m_scale(
    base_i_total: jax.Array,
    adrenaline: jax.Array,
    tau_m_ms: float,
    dt_ms: float,
) -> tuple[jax.Array, jax.Array]:
    """Scale tau_m: higher adrenaline -> faster membrane integration.

    Precondition: adrenaline > 0. Division by a non-positive adrenaline
    produces inf or a negative effective tau, which flips the leak
    term's sign in `integrate_and_spike` and silently corrupts the
    dynamics (no NaN, no exception). Callers are responsible for
    keeping adrenaline > 0 under this gain_mode; biological plausibility
    aligns with this precondition (adrenaline levels are non-negative).
    """
    del dt_ms
    return base_i_total, jnp.asarray(tau_m_ms) / adrenaline


def _modulate_threshold_shift(
    base_i_total: jax.Array,
    adrenaline: jax.Array,
    tau_m_ms: float,
    dt_ms: float,
) -> tuple[jax.Array, jax.Array]:
    """Equivalent to lowering V_THRESH by (adr - 1) * shift_mv.

    Lowering V_THRESH by delta is equivalent, after one Euler step, to
    adding delta * (tau_m / dt) to i_total (so that the integrated dv
    picks up exactly delta of headroom). We implement the equivalence
    rather than plumbing a per-step V_THRESH through integrate_and_spike.
    """
    delta = (adrenaline - 1.0) * _THRESHOLD_SHIFT_MV
    shifted = base_i_total + delta * (tau_m_ms / dt_ms)
    return shifted, jnp.asarray(tau_m_ms)


GAIN_MODULATORS: dict[
    GainMode,
    Callable[
        [jax.Array, jax.Array, float, float],
        tuple[jax.Array, jax.Array],
    ],
] = {
    "multiplicative": _modulate_multiplicative,
    "multiplicative_mild": _modulate_multiplicative_mild,
    "additive": _modulate_additive,
    "tau_m_scale": _modulate_tau_m_scale,
    "threshold_shift": _modulate_threshold_shift,
}


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


def stdp_update(
    pool: SlotPool,
    traces: Traces,
    pre_spike_source: jax.Array,
    post_spikes: jax.Array,
    valence: jax.Array,
    params: STDPParams,
    dt_ms: float = DT_MS,
) -> tuple[SlotPool, Traces]:
    """Run one step of STDP weight update + trace decay + spike increment.

    Pure function split out of `step_plastic` so paired-agent sims can
    feed distinct pre and post rasters (the pre vector may span the
    agent's own population + partners it receives spikes from, while
    post stays over the agent's own neurons).

    Args:
        pool: current SlotPool.
        traces: current Traces. `traces.pre.shape[0]` must match the
            pre-raster length `pre_spike_source.shape[0]`; `traces.post.
            shape[0]` must match `post_spikes.shape[0]`.
        pre_spike_source: spike vector to fold into the pre-trace for
            the *next* step (shape (N_pre,), any bool or float dtype).
            Conventionally the *current* step's output of whichever
            population is upstream of this pool; in single-population
            recurrent use it is the same vector as `post_spikes`.
        post_spikes: this step's postsynaptic spike output, shape
            (N_post,).
        valence: scalar three-factor modulator.
        params: STDP hyperparameters.
        dt_ms: integration timestep in ms (for trace decay factor).

    Returns:
        (new_pool, new_traces) with updated pool.v, pool unchanged in
        other fields, and traces decayed + incremented by this step's
        spike contributions.
    """
    spike_post_f = post_spikes.astype(jnp.float32)
    pre_spike_f = pre_spike_source.astype(jnp.float32)

    decay_pre = jnp.exp(-dt_ms / params.tau_pre_ms)
    decay_post = jnp.exp(-dt_ms / params.tau_post_ms)
    # Pre-decayed traces: the trace value at the moment of this step's
    # spike, before the step's own spike is folded in (Song/Miller/
    # Abbott convention; avoids coincident-spike inflation).
    pre_decayed = traces.pre * decay_pre
    post_decayed = traces.post * decay_post

    pre_trace_slot = pre_decayed[pool.pre_ids]
    post_trace_slot = post_decayed[:, None]
    pre_spike_slot = pre_spike_f[pool.pre_ids]
    post_spike_slot = spike_post_f[:, None]

    ltp = params.a_plus * post_spike_slot * pre_trace_slot
    ltd = params.a_minus * pre_spike_slot * post_trace_slot
    dv = (ltp - ltd) * valence * pool.plasticity_rate
    dv = jnp.where(pool.active, dv, jnp.float32(0.0))

    new_v = jnp.clip(pool.v + dv, params.v_min, params.v_max)
    new_pool = pool._replace(v=new_v)

    new_traces = Traces(
        pre=pre_decayed + pre_spike_f,
        post=post_decayed + spike_post_f,
    )
    return new_pool, new_traces


def step_plastic(
    state: PlasticNetState,
    i_ext: jax.Array,
    valence: jax.Array,
    adrenaline: jax.Array,
    params: STDPParams,
    gain_mode: GainMode = "multiplicative",
    structural_params: StructuralParams | None = None,
    dt_ms: float = DT_MS,
    tau_m_ms: float = TAU_M_MS,
) -> PlasticNetState:
    """Advance one timestep of a plastic single-population recurrent net.

    Thin wrapper over `stdp_update`: computes the LIF forward pass using
    the agent's own previous-step spikes as synaptic input, then applies
    the single-population STDP convention where this step's output also
    serves as the pre-spike source for trace update. Paired-agent sims
    should call `stdp_update` directly with distinct pre/post rasters.

    Args:
        state: current PlasticNetState.
        i_ext: external current drive this step, shape (N_post,).
        valence: scalar three-factor modulator for this step.
        adrenaline: scalar neural-gain modulator for this step.
        params: STDP hyperparameters.
        gain_mode: adrenaline gain mechanism; see the GainMode Literal.
        structural_params: when set, apply slot release after the STDP
            update.
        dt_ms: integration timestep in ms.
        tau_m_ms: LIF membrane time constant in ms.

    Returns:
        Next PlasticNetState.
    """
    assert state.traces.pre.shape[0] == state.traces.post.shape[0], (
        "plasticity.step_plastic assumes a single-population recurrent "
        "network; pre and post trace vectors must have the same length. "
        "For paired-agent sims use paired.step_paired instead."
    )

    prev_spikes = state.lif.spikes
    i_syn = synaptic_current(state.pool, prev_spikes)
    base_i_total = i_ext + i_syn
    modulator = GAIN_MODULATORS[gain_mode]
    i_total, tau_eff = modulator(
        base_i_total, adrenaline, tau_m_ms, dt_ms
    )
    v_next, spike_post = integrate_and_spike(
        state.lif.v, i_total, dt_ms, tau_eff
    )
    new_lif = LIFState(v=v_next, spikes=spike_post)

    new_pool, new_traces = stdp_update(
        state.pool,
        state.traces,
        pre_spike_source=spike_post,
        post_spikes=spike_post,
        valence=valence,
        params=params,
        dt_ms=dt_ms,
    )

    if structural_params is not None:
        new_pool = apply_release(new_pool, structural_params)

    return PlasticNetState(lif=new_lif, pool=new_pool, traces=new_traces)


def simulate_plastic(
    initial_state: PlasticNetState,
    i_ext_trace: jax.Array,
    valence_trace: jax.Array,
    adrenaline_trace: jax.Array,
    params: STDPParams,
    gain_mode: GainMode = "multiplicative",
    structural_params: StructuralParams | None = None,
) -> tuple[PlasticNetState, jax.Array]:
    """Simulate a plastic slot-pool network over drive + modulator traces.

    Args:
        initial_state: starting PlasticNetState.
        i_ext_trace: external current per timestep, shape (T, N_post).
        valence_trace: scalar valence per timestep, shape (T,).
        adrenaline_trace: scalar adrenaline per timestep, shape (T,).
        params: STDP hyperparameters.
        gain_mode: adrenaline mechanism selector; see step_plastic.
        structural_params: when set, apply slot-release structural
            plasticity each step.

    Returns:
        final_state: PlasticNetState after T steps.
        spike_trace: boolean spikes per timestep, shape (T, N_post).
    """

    def scan_step(
        state: PlasticNetState,
        drive: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[PlasticNetState, jax.Array]:
        i_ext_t, valence_t, adrenaline_t = drive
        next_state = step_plastic(
            state,
            i_ext_t,
            valence_t,
            adrenaline_t,
            params,
            gain_mode=gain_mode,
            structural_params=structural_params,
        )
        return next_state, next_state.lif.spikes

    final_state, spike_trace = jax.lax.scan(
        scan_step,
        initial_state,
        (i_ext_trace, valence_trace, adrenaline_trace),
    )
    return final_state, spike_trace
