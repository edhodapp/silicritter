"""Structural plasticity: slot release (and later, acquisition).

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 6 of the silicritter implementation ladder: add slot release to
the slot-pool substrate. Before this step, slots remained structurally
bound for the full lifetime of the simulation even if their weight v
was driven to v_min by LTD -- functionally silent but occupying
hardware. Release converts that "functionally silent" state into a
genuinely free slot that returns to the pool.

Release rule (dwell-based):

    if slot.active and slot.v < v_release_threshold:
        slot.release_counter += 1
    else:
        slot.release_counter = 0

    if slot.release_counter >= release_dwell_steps:
        slot.active = False
        slot.v = 0
        slot.release_counter = 0

This is the "pruning" half of the developmental exuberance-and-pruning
story. **Acquisition** -- opportunistic rebinding of freed slots to
new presynaptic partners based on correlated activity -- requires
PRNG threading through the inner-loop scan and lands in a later step
(6.5 or 7). Step 6 delivers release-only; without acquisition, the
pool only contracts over time, matching lifelong pruning but not
ongoing formation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from silicritter.slotpool import SlotPool


class StructuralParams(NamedTuple):
    """Parameters controlling slot release.

    Attributes:
        v_release_threshold: slots with v below this value accumulate
            toward release. Default 0.01 (just above v_min=0, so a slot
            clipped to v_min is always counting down).
        release_dwell_steps: number of consecutive sub-threshold steps
            required to release the slot. Default 100 (= 100 ms
            simulated at dt=1ms).
    """

    v_release_threshold: float
    release_dwell_steps: int


def default_structural_params() -> StructuralParams:
    """Return a modest default StructuralParams appropriate for step 6."""
    return StructuralParams(
        v_release_threshold=0.01,
        release_dwell_steps=100,
    )


def apply_release(
    pool: SlotPool,
    params: StructuralParams,
) -> SlotPool:
    """Update release_counter and release slots whose dwell exceeds dwell_steps.

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
