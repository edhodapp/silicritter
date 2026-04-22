"""Leaky Integrate-and-Fire (LIF) forward simulation in JAX.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 2 of the silicritter implementation ladder: a minimal, no-plasticity
SNN forward sim to verify the GPU path and measure throughput. A single
population of N neurons with fixed recurrent connectivity, driven by
external current.

The state variable V (membrane potential) evolves under leaky integration
with threshold-and-reset spiking:

    tau_m * dV/dt = -(V - V_rest) + I(t)

with I(t) composed of external drive and recurrent synaptic input from
the previous timestep's spikes. Current is expressed directly in mV units
(R folded into the input scale) since this is a forward-sim baseline, not
a biophysically-calibrated model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# Biologically plausible cortical-regime LIF parameters.
TAU_M_MS: float = 20.0
V_REST_MV: float = -65.0
V_THRESH_MV: float = -50.0
V_RESET_MV: float = -65.0
DT_MS: float = 1.0


class LIFState(NamedTuple):
    """Transient state of an LIF population across timesteps.

    Attributes:
        v: membrane potentials in mV, shape (N,), dtype float32.
        spikes: boolean spike output of the most recent step, shape (N,).
    """

    v: jax.Array
    spikes: jax.Array


def init_state(n_neurons: int) -> LIFState:
    """Build an initial LIFState with V at rest and no prior spikes."""
    v = jnp.full((n_neurons,), V_REST_MV, dtype=jnp.float32)
    spikes = jnp.zeros((n_neurons,), dtype=jnp.bool_)
    return LIFState(v=v, spikes=spikes)


def step(
    state: LIFState,
    weights: jax.Array,
    i_ext: jax.Array,
    dt_ms: float = DT_MS,
    tau_m_ms: float = TAU_M_MS,
) -> LIFState:
    """Advance the LIF population by one timestep (forward Euler).

    Args:
        state: current LIFState (V at time t, spikes at t-1).
        weights: recurrent weight matrix, shape (N, N).
        i_ext: external current drive for this step, shape (N,).
        dt_ms: integration timestep in ms.
        tau_m_ms: membrane time constant in ms.

    Returns:
        Next LIFState (V at t+1, spikes at t+1).
    """
    i_syn = weights @ state.spikes.astype(jnp.float32)
    i_total = i_ext + i_syn
    dv = (-(state.v - V_REST_MV) + i_total) * (dt_ms / tau_m_ms)
    v_integrated = state.v + dv

    spike = v_integrated >= V_THRESH_MV
    v_next = jnp.where(spike, V_RESET_MV, v_integrated)

    return LIFState(v=v_next, spikes=spike)


def simulate(
    initial_state: LIFState,
    weights: jax.Array,
    i_ext_trace: jax.Array,
) -> tuple[LIFState, jax.Array]:
    """Simulate the population across a trace of external input currents.

    Args:
        initial_state: starting LIFState.
        weights: recurrent weight matrix, shape (N, N).
        i_ext_trace: external current per timestep, shape (T, N).

    Returns:
        final_state: LIFState after T steps.
        spike_trace: boolean spikes per timestep, shape (T, N).
    """

    def scan_step(
        state: LIFState, i_ext_t: jax.Array
    ) -> tuple[LIFState, jax.Array]:
        next_state = step(state, weights, i_ext_t)
        return next_state, next_state.spikes

    final_state, spike_trace = jax.lax.scan(
        scan_step, initial_state, i_ext_trace
    )
    return final_state, spike_trace
