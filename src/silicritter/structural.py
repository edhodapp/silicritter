"""Structural plasticity: slot release and acquisition.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Slots in the slot-pool substrate undergo two structural-plasticity
dynamics:

Release (pruning):
    if slot.active and slot.v < v_release_threshold:
        slot.release_counter += 1
    else:
        slot.release_counter = 0
    if slot.release_counter >= release_dwell_steps:
        slot.active = False
        slot.v = 0
        slot.release_counter = 0

Acquisition (formation):
    for each inactive slot, with probability acquisition_prob per step:
        slot.pre_id = uniform / Hebbian-biased draw from [0, n_pre)
        slot.v = acquisition_initial_v
        slot.plasticity_rate = acquisition_plasticity_rate
        slot.active = True
        slot.release_counter = 0

Release is the "sculpting" dynamic (LTD weakens a slot to zero, the
release timer eventually retires it). Acquisition is the "exuberance"
dynamic (freed slots re-bind to new presynaptic partners so the pool
can explore alternatives). Together they implement the developmental
arc this project targets -- not just adult steady-state plasticity.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from silicritter.slotpool import SlotPool


class StructuralParams(NamedTuple):
    """Parameters controlling slot release and acquisition.

    Release:
        v_release_threshold: slots with v below this value accumulate
            toward release. Default 0.01 (just above v_min=0).
        release_dwell_steps: number of consecutive sub-threshold steps
            required to release the slot. Default 100 (= 100 ms at
            dt=1ms).

    Acquisition:
        acquisition_prob: per inactive slot per step, probability of
            rebinding. Default 0.0 preserves release-only behavior
            (backward compatible).
        acquisition_initial_v: v assigned to a newly-acquired slot.
            Default 0.2 (v_max/10 for V_MAX=2.0 -- starts weak, has
            to prove itself via STDP).
        acquisition_plasticity_rate: plasticity_rate assigned to a
            newly-acquired slot. Default 1.0 (fully plastic so STDP
            can evaluate the new binding).
    """

    v_release_threshold: float
    release_dwell_steps: int
    acquisition_prob: float = 0.0
    acquisition_initial_v: float = 0.2
    acquisition_plasticity_rate: float = 1.0


def default_structural_params() -> StructuralParams:
    """Release-only defaults (no acquisition; backward-compatible)."""
    return StructuralParams(
        v_release_threshold=0.01,
        release_dwell_steps=100,
    )


def apply_release(
    pool: SlotPool,
    params: StructuralParams,
) -> SlotPool:
    """Update release_counter; release slots whose dwell exceeds limit.

    Pure function: returns a new SlotPool with updated active, v, and
    release_counter arrays. pre_ids and plasticity_rate are unchanged
    (they persist across release; a subsequent acquisition will
    overwrite pre_ids and reset plasticity_rate).

    Args:
        pool: current SlotPool.
        params: StructuralParams controlling the dwell threshold.

    Returns:
        Updated SlotPool with released slots deactivated.
    """
    below = pool.v < params.v_release_threshold
    active_and_below = pool.active & below

    # Increment counter where active-and-below; reset to 0 otherwise.
    incremented = pool.release_counter + 1
    new_counter = jnp.where(active_and_below, incremented, 0)

    should_release = new_counter >= params.release_dwell_steps

    new_active = pool.active & ~should_release
    new_v = jnp.where(should_release, jnp.float32(0.0), pool.v)
    new_counter = jnp.where(should_release, jnp.int32(0), new_counter)

    return pool._replace(
        active=new_active,
        v=new_v,
        release_counter=new_counter,
    )


def apply_acquisition(
    pool: SlotPool,
    params: StructuralParams,
    rng: jax.Array,
    n_pre: int,
    pre_activity: jax.Array | None = None,
) -> SlotPool:
    """Rebind inactive slots to new pre-neurons with per-step probability.

    Inactive slots (active=False) are candidates. Each independently
    flips a Bernoulli(acquisition_prob); winners get a fresh pre_id,
    initial v, fully enabled plasticity_rate, and active=True.

    Args:
        pool: current SlotPool.
        params: StructuralParams (acquisition_prob, initial_v,
            plasticity_rate).
        rng: JAX PRNGKey. Split internally for the Bernoulli mask
            and the pre_id draws.
        n_pre: size of the presynaptic space (for uniform draws and
            Hebbian-biased draws alike).
        pre_activity: optional (n_pre,) float array. None gives
            uniform random pre_id draws (null hypothesis --
            exploration with no prior). A non-None array biases the
            draw proportional to pre_activity[j] + eps (Hebbian
            acquisition -- slots preferentially bind to
            recently-firing pre-neurons).

    Returns:
        Updated SlotPool with acquired slots rebound.
    """
    k_bernoulli, k_id = jax.random.split(rng)
    bernoulli = jax.random.uniform(
        k_bernoulli, pool.v.shape,
    ) < params.acquisition_prob
    acquire_mask = (~pool.active) & bernoulli
    if pre_activity is None:
        new_pre_ids = jax.random.randint(
            k_id, pool.pre_ids.shape,
            minval=0, maxval=n_pre, dtype=jnp.int32,
        )
    else:
        probs = pre_activity + jnp.float32(1e-6)
        probs = probs / probs.sum()
        new_pre_ids = jax.random.choice(
            k_id, n_pre, shape=pool.pre_ids.shape, p=probs, replace=True,
        ).astype(jnp.int32)
    return pool._replace(
        pre_ids=jnp.where(acquire_mask, new_pre_ids, pool.pre_ids),
        v=jnp.where(
            acquire_mask,
            jnp.float32(params.acquisition_initial_v),
            pool.v,
        ),
        plasticity_rate=jnp.where(
            acquire_mask,
            jnp.float32(params.acquisition_plasticity_rate),
            pool.plasticity_rate,
        ),
        active=pool.active | acquire_mask,
        release_counter=jnp.where(
            acquire_mask, jnp.int32(0), pool.release_counter,
        ),
    )
