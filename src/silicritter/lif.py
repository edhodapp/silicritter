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

# Invariant guard: a reset potential at or above threshold would cause
# every post-reset step to spike immediately, locking the network into
# pathological activity. Assert at module import so any future edit of
# the constants is caught early.
assert V_RESET_MV < V_THRESH_MV, (
    "V_RESET_MV must be below V_THRESH_MV; otherwise neurons spike "
    "continuously after reset."
)
assert TAU_M_MS > 0.0, "TAU_M_MS must be positive."
assert DT_MS > 0.0, "DT_MS must be positive."


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


def integrate_and_spike(
    v: jax.Array,
    i_total: jax.Array,
    dt_ms: float = DT_MS,
    tau_m_ms: float | jax.Array = TAU_M_MS,
) -> tuple[jax.Array, jax.Array]:
    """Forward-Euler LIF integration with threshold-and-reset.

    Pure membrane-dynamics primitive: given the current membrane potential
    and the total input current, advance V by one timestep and emit
    threshold-crossing spikes with V reset to V_RESET_MV.

    Args:
        v: current membrane potentials, shape (N,), mV.
        i_total: total input current this step, shape (N,).
        dt_ms: integration timestep in ms.
        tau_m_ms: membrane time constant in ms. Accepts float or JAX
            scalar Array so modulators (e.g. adrenaline tau_m_scale)
            can pass a per-step effective time constant.

    Returns:
        v_next: post-step membrane potentials, with V_RESET_MV where a
            spike occurred.
        spike: boolean spike flags, shape (N,).
    """
    dv = (-(v - V_REST_MV) + i_total) * (dt_ms / tau_m_ms)
    v_integrated = v + dv
    spike = v_integrated >= V_THRESH_MV
    v_next = jnp.where(spike, V_RESET_MV, v_integrated)
    return v_next, spike


def step(
    state: LIFState,
    weights: jax.Array,
    i_ext: jax.Array,
    dt_ms: float = DT_MS,
    tau_m_ms: float = TAU_M_MS,
) -> LIFState:
    """Advance the LIF population by one timestep using dense weights.

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
    v_next, spike = integrate_and_spike(state.v, i_total, dt_ms, tau_m_ms)
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
