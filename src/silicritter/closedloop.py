"""Closed-loop adrenaline controller for paired-agent sims.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

A leaky-integrator (EMA) estimate of each agent's firing rate, driving
an error signal that modulates B's adrenaline via tau_m_scale. The
shape is the Aston-Jones & Cohen 2005 adaptive-gain story executed
as a control loop rather than an open-loop feed-forward.

Per-step update:
    rate_a_ema <- decay * rate_a_ema + (1 - decay) * mean(spikes_a)
    rate_b_ema <- decay * rate_b_ema + (1 - decay) * mean(spikes_b)
    error      = rate_a_ema - rate_b_ema
    adr_b      = clip(baseline + gain * error, adr_min, adr_max)

The controller reads only population-mean firing rates (not individual
neuron state), which matches the biological picture of LC-broadcast
NE acting as a global gain signal.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from silicritter.lif import DT_MS, TAU_M_MS
from silicritter.paired import PairedState, step_paired
from silicritter.plasticity import GainMode, STDPParams
from silicritter.structural import StructuralParams


class ControllerState(NamedTuple):
    """Leaky-integrator EMAs plus the current adrenaline output."""

    rate_a_ema: jax.Array
    rate_b_ema: jax.Array
    adrenaline_b: jax.Array


class ControllerParams(NamedTuple):
    """Static configuration of the closed-loop controller."""

    decay: float
    baseline: float
    gain: float
    adr_min: float
    adr_max: float


def init_controller(baseline: float = 1.0) -> ControllerState:
    """Initial controller state: zero rate EMAs, baseline adrenaline."""
    return ControllerState(
        rate_a_ema=jnp.float32(0.0),
        rate_b_ema=jnp.float32(0.0),
        adrenaline_b=jnp.float32(baseline),
    )


def step_closedloop(
    state: PairedState,
    ctrl: ControllerState,
    ctrl_params: ControllerParams,
    i_ext_a: jax.Array,
    i_ext_b: jax.Array,
    valence_a: jax.Array,
    valence_b: jax.Array,
    adrenaline_a: jax.Array,
    stdp_params: STDPParams,
    gain_mode: GainMode = "tau_m_scale",
    structural_params: StructuralParams | None = None,
    a_is_inhibitory: jax.Array | None = None,
    b_is_inhibitory: jax.Array | None = None,
    i_weight_multiplier: float = 4.0,
    dt_ms: float = DT_MS,
    tau_m_ms: float = TAU_M_MS,
) -> tuple[PairedState, ControllerState]:
    """Advance one paired step under closed-loop adrenaline control on B."""
    next_paired = step_paired(
        state,
        i_ext_a=i_ext_a, i_ext_b=i_ext_b,
        valence_a=valence_a, valence_b=valence_b,
        adrenaline_a=adrenaline_a, adrenaline_b=ctrl.adrenaline_b,
        stdp_params=stdp_params,
        gain_mode=gain_mode,
        structural_params=structural_params,
        a_is_inhibitory=a_is_inhibitory,
        b_is_inhibitory=b_is_inhibitory,
        i_weight_multiplier=i_weight_multiplier,
        dt_ms=dt_ms,
        tau_m_ms=tau_m_ms,
    )
    rate_a_now = next_paired.a.lif.spikes.astype(jnp.float32).mean()
    rate_b_now = next_paired.b.lif.spikes.astype(jnp.float32).mean()
    new_rate_a = (
        ctrl.rate_a_ema * ctrl_params.decay
        + rate_a_now * (1.0 - ctrl_params.decay)
    )
    new_rate_b = (
        ctrl.rate_b_ema * ctrl_params.decay
        + rate_b_now * (1.0 - ctrl_params.decay)
    )
    error = new_rate_a - new_rate_b
    raw_adr = (
        jnp.float32(ctrl_params.baseline)
        + jnp.float32(ctrl_params.gain) * error
    )
    new_adr = jnp.clip(
        raw_adr,
        jnp.float32(ctrl_params.adr_min),
        jnp.float32(ctrl_params.adr_max),
    )
    return next_paired, ControllerState(new_rate_a, new_rate_b, new_adr)


def simulate_closedloop(
    initial_state: PairedState,
    ctrl_params: ControllerParams,
    i_ext_a_trace: jax.Array,
    i_ext_b_trace: jax.Array,
    valence_a_trace: jax.Array,
    valence_b_trace: jax.Array,
    adrenaline_a_trace: jax.Array,
    stdp_params: STDPParams,
    gain_mode: GainMode = "tau_m_scale",
    structural_params: StructuralParams | None = None,
    a_is_inhibitory: jax.Array | None = None,
    b_is_inhibitory: jax.Array | None = None,
    i_weight_multiplier: float = 4.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run a closed-loop paired sim over T steps.

    Returns (spikes_a, spikes_b, adrenaline_b_trace), each of shape
    (T, ...) — the first two are (T, n_neurons); adrenaline_b is (T,).
    """
    def scan_step(
        carry: tuple[PairedState, ControllerState],
        drive: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[PairedState, ControllerState],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        paired_state, ctrl = carry
        i_ext_a_t, i_ext_b_t, val_a_t, val_b_t, adr_a_t = drive
        next_paired, next_ctrl = step_closedloop(
            paired_state, ctrl, ctrl_params,
            i_ext_a=i_ext_a_t, i_ext_b=i_ext_b_t,
            valence_a=val_a_t, valence_b=val_b_t,
            adrenaline_a=adr_a_t,
            stdp_params=stdp_params,
            gain_mode=gain_mode,
            structural_params=structural_params,
            a_is_inhibitory=a_is_inhibitory,
            b_is_inhibitory=b_is_inhibitory,
            i_weight_multiplier=i_weight_multiplier,
        )
        return (next_paired, next_ctrl), (
            next_paired.a.lif.spikes,
            next_paired.b.lif.spikes,
            next_ctrl.adrenaline_b,
        )

    initial_ctrl = init_controller(ctrl_params.baseline)
    drives = (
        i_ext_a_trace, i_ext_b_trace,
        valence_a_trace, valence_b_trace,
        adrenaline_a_trace,
    )
    _, (spikes_a, spikes_b, adr_trace) = jax.lax.scan(
        scan_step, (initial_state, initial_ctrl), drives
    )
    return spikes_a, spikes_b, adr_trace
