"""Slot-pool synapses: per-postsynaptic-neuron pools of synaptic slots.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 3 of the silicritter implementation ladder: introduce the structural
representation of synapses as per-postsynaptic-neuron pools of discrete
slots, with no plasticity (static pool) as the baseline.

Each postsynaptic neuron has a fixed-capacity pool of K slots. Each slot
is bound to a presynaptic neuron (pre_ids) and carries:

  - v: analog weight state (the "v" in hybrid w = N * v)
  - plasticity_rate: continuous [0, 1] parameter (0 = innate/hardwired,
    1 = fully plastic). Unused here; wired in for step 4+ plasticity.
  - active: whether the slot is currently allocated.

The effective weight from presynaptic neuron i to postsynaptic neuron j
is the sum over slots in j's pool bound to i of that slot's v. Multiple
slots binding the same (i, j) pair is allowed and accumulates -- the N
in hybrid w = N * v emerges naturally as the count of active slots
bound to that connection.

Structural plasticity (slot acquisition, release, eviction) is deferred
to step 4+. Step 3 presents a static slot pool to verify representation
and forward-sim correctness.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from silicritter.lif import DT_MS, TAU_M_MS, LIFState, integrate_and_spike


class SlotPool(NamedTuple):
    """Per-postsynaptic-neuron pool of synaptic slots.

    All arrays have shape (N_post, K) where K is the pool capacity.
    pre_ids is int32, so n_pre is capped near 2**31; far above any
    foreseeable silicritter network, but worth knowing.

    Attributes:
        pre_ids: int32, presynaptic neuron ID bound to each slot.
        v: float32, per-slot analog weight.
        plasticity_rate: float32 in [0, 1]; 0 = innate / hardwired,
            1 = fully plastic. Reserved for step 4+ plasticity rules.
        active: bool, whether the slot is allocated (True) or free.
        release_counter: int32, count of consecutive steps this slot's
            v has sat below the structural-release threshold. Reset to
            0 whenever v rises above the threshold or when the slot is
            released. Reaches `release_dwell_steps` (see
            `structural.StructuralParams`) triggers release. Ignored
            when structural plasticity is not enabled.
    """

    pre_ids: jax.Array
    v: jax.Array
    plasticity_rate: jax.Array
    active: jax.Array
    release_counter: jax.Array


def init_random(
    n_post: int,
    n_pre: int,
    slots_per_post: int,
    rng: jax.Array,
    weight_scale: float = 0.05,
    default_plasticity_rate: float = 1.0,
) -> SlotPool:
    """Build a random sparse slot pool for baseline forward simulations.

    Each post-neuron receives slots_per_post slots, each bound to a
    uniformly random presynaptic ID in [0, n_pre) with a random v drawn
    from a normal distribution scaled by weight_scale. All slots are
    active and carry the same default plasticity rate.

    Args:
        n_post: number of postsynaptic neurons.
        n_pre: number of presynaptic neurons.
        slots_per_post: K, pool capacity per postsynaptic neuron.
        rng: JAX PRNGKey.
        weight_scale: scale of the underlying N(0, 1) before absolute
            value; the realized per-slot v is drawn from the half-
            normal |N(0, 1)| * weight_scale (mean approx
            0.798 * weight_scale, std approx 0.603 * weight_scale).
        default_plasticity_rate: initial plasticity rate for all slots.

    Returns:
        SlotPool with arrays shaped (n_post, slots_per_post).
    """
    k_ids, k_v = jax.random.split(rng)
    pre_ids = jax.random.randint(
        k_ids,
        (n_post, slots_per_post),
        minval=0,
        maxval=n_pre,
        dtype=jnp.int32,
    )
    # Half-normal sample: |N(0, 1)| * scale. Keeps v non-negative so the
    # default v_min = 0 clip in the plasticity module doesn't silently
    # alter the initial pool on the first update. Excitatory-only is the
    # natural convention at this step; a separate inhibitory pool or a
    # signed variant can land when we need it.
    v = (
        jnp.abs(
            jax.random.normal(
                k_v, (n_post, slots_per_post), dtype=jnp.float32
            )
        )
        * weight_scale
    )
    plasticity_rate = jnp.full(
        (n_post, slots_per_post),
        default_plasticity_rate,
        dtype=jnp.float32,
    )
    active = jnp.ones((n_post, slots_per_post), dtype=jnp.bool_)
    release_counter = jnp.zeros(
        (n_post, slots_per_post), dtype=jnp.int32
    )
    return SlotPool(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=plasticity_rate,
        active=active,
        release_counter=release_counter,
    )


def synaptic_current(pool: SlotPool, spikes: jax.Array) -> jax.Array:
    """Compute per-postsynaptic synaptic input from presynaptic spikes.

    For each post-neuron, each active slot bound to a pre-neuron that
    fired this step contributes its v to the synaptic current.

    Contract: `spikes.shape[0]` must exceed every index in `pool.pre_ids`.
    JAX out-of-bounds gather-indexing clamps silently rather than raising,
    so a mismatch between the pool's pre-index range and `spikes.shape[0]`
    produces wrong results rather than an error. Callers are responsible
    for keeping these aligned; `init_random(n_post, n_pre, ...)` guarantees
    it when the spike vector is sized to `n_pre`.

    Args:
        pool: SlotPool with arrays of shape (N_post, K).
        spikes: presynaptic spikes, shape (N_pre,), boolean.

    Returns:
        Per-post synaptic current, shape (N_post,), float32.
    """
    spike_gathered = spikes[pool.pre_ids]
    contrib = pool.v * spike_gathered.astype(jnp.float32)
    contrib = jnp.where(pool.active, contrib, jnp.float32(0.0))
    return contrib.sum(axis=1)


def effective_weights(pool: SlotPool, n_pre: int) -> jax.Array:
    """Reconstruct the dense (N_post, N_pre) weight matrix from the pool.

    Useful for tests and equivalence checks against the dense-weight step
    in lif.py. Not efficient for large N_pre; intended for validation.

    Args:
        pool: SlotPool.
        n_pre: number of presynaptic neurons.

    Returns:
        Dense weight matrix, shape (N_post, n_pre), float32.
    """
    active_v = jnp.where(pool.active, pool.v, jnp.float32(0.0))
    onehot = jax.nn.one_hot(pool.pre_ids, n_pre, dtype=jnp.float32)
    return jnp.einsum("jk,jki->ji", active_v, onehot)


def step(
    state: LIFState,
    pool: SlotPool,
    i_ext: jax.Array,
    dt_ms: float = DT_MS,
    tau_m_ms: float = TAU_M_MS,
) -> LIFState:
    """Advance the LIF population by one timestep using slot-pool synapses.

    Args:
        state: current LIFState.
        pool: SlotPool (static at this step; plasticity lands at step 4+).
        i_ext: external current drive this step, shape (N_post,).
        dt_ms: integration timestep in ms.
        tau_m_ms: membrane time constant in ms.

    Returns:
        Next LIFState.
    """
    i_syn = synaptic_current(pool, state.spikes)
    i_total = i_ext + i_syn
    v_next, spike = integrate_and_spike(state.v, i_total, dt_ms, tau_m_ms)
    return LIFState(v=v_next, spikes=spike)


def simulate(
    initial_state: LIFState,
    pool: SlotPool,
    i_ext_trace: jax.Array,
) -> tuple[LIFState, jax.Array]:
    """Simulate the slot-pool network across a trace of input currents.

    Args:
        initial_state: starting LIFState.
        pool: SlotPool (static throughout the simulation at step 3).
        i_ext_trace: external current per timestep, shape (T, N_post).

    Returns:
        final_state: LIFState after T steps.
        spike_trace: boolean spikes per timestep, shape (T, N_post).
    """

    def scan_step(
        state: LIFState, i_ext_t: jax.Array
    ) -> tuple[LIFState, jax.Array]:
        next_state = step(state, pool, i_ext_t)
        return next_state, next_state.spikes

    final_state, spike_trace = jax.lax.scan(
        scan_step, initial_state, i_ext_trace
    )
    return final_state, spike_trace
