"""Functional contracts for step16 `_describe_pool` cross-E classification.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

These tests pin the behavioral contract that `_describe_pool` uses
the partner's `a_is_inh` mask to classify cross-substrate slots,
NOT the contiguous-layout assumption baked into `assign_ei_identity`.

Pre-task-#4 (2026-04-27 cleanup): cross_e_mask was computed as
    (pre_ids >= N_NEURONS) & (pre_ids < N_NEURONS + n_ex)
which re-derived `assign_ei_identity`'s "first n_ex are E, last
n_inh are I" invariant in the diagnostic itself. Correct under the
documented contract, but coupled the diagnostic to a specific
layout.

Post-task-#4: cross_e_mask = `cross & ~a_is_inh[partner_idx]`. The
diagnostic now consults the actual mask, decoupled from any layout
assumption.

The test below uses an *inverted* a_is_inh layout (I at low indices,
E at high indices) — opposite of the documented contiguous one — and
constructs a pool whose pre_ids all point to a known I-position. The
correct (post-task-#4) cross_e_frac is 0.0; the pre-task-#4 code
returns a non-zero value because it doesn't consult a_is_inh.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

# pylint: disable=wrong-import-position,import-error
import step16_stdp_learning as s16  # noqa: E402
from silicritter.slotpool import SlotPool, assign_ei_identity  # noqa: E402

# pylint: disable=protected-access


def _build_test_pool(pre_ids: jnp.ndarray) -> SlotPool:
    """Build a minimal SlotPool from supplied pre_ids (rest defaulted)."""
    shape = pre_ids.shape
    return SlotPool(
        pre_ids=pre_ids,
        v=jnp.full(shape, 0.5, dtype=jnp.float32),
        plasticity_rate=jnp.ones(shape, dtype=jnp.float32),
        active=jnp.ones(shape, dtype=jnp.bool_),
        release_counter=jnp.zeros(shape, dtype=jnp.int32),
    )


def test_describe_pool_cross_e_mask_consults_a_is_inh_under_inverted_layout(
) -> None:
    """cross_e_frac uses a_is_inh, not the contiguous-layout assumption.

    Setup: all 8 slots point to partner_idx 0; a_is_inh is inverted
    (I at index 0, E everywhere else). Partner's neuron 0 is I, so
    no slot points to a partner E neuron → cross_e_frac = 0.0.

    Pre-task-#4 code computes cross_e_mask = (pre_ids in [N, N+n_ex))
    which is True for all eight slots (n_ex = 256-51 = 205 here, and
    256 ∈ [256, 461)). It would return cross_e_frac = 1.0, ignoring
    a_is_inh entirely. This test fails under that code.
    """
    n_slots = 8
    # All slots point to the partner's neuron 0.
    pre_ids = jnp.full((1, n_slots), s16.N_NEURONS, dtype=jnp.int32)
    pool = _build_test_pool(pre_ids)

    # Inverted layout: a_is_inh[0] = True (I), all others False (E).
    a_is_inh = jnp.zeros(s16.N_NEURONS, dtype=jnp.bool_).at[0].set(True)

    stats = s16._describe_pool(pool, a_is_inh)

    # Under the correct (post-task-#4) code, all slots point to an I
    # neuron of the partner, so no slot is "cross-E".
    cross_e_frac = stats["cross_e_frac"]
    cross_frac = stats["cross_frac"]
    assert cross_e_frac == 0.0, (
        f"cross_e_frac should be 0.0 under inverted a_is_inh "
        f"(all pre_ids point to a_is_inh[0]=I); got {cross_e_frac}"
    )
    # Sanity: every slot IS in the partner space, so cross_frac is 1.0.
    assert cross_frac == 1.0, (
        f"cross_frac should be 1.0 (all pre_ids in partner space); "
        f"got {cross_frac}"
    )


def test_describe_pool_cross_e_mask_correct_under_canonical_layout() -> None:
    """Sanity: with the canonical assign_ei_identity layout, the
    mask-based check produces the correct cross-E count.

    Layout: E at low indices [0, n_ex), I at high indices [n_ex, N).
    Pool: 4 slots pointing to canonical-E partner neurons (low
    partner indices), 4 slots pointing to canonical-I partner neurons
    (high partner indices). Expected cross_e_frac = 0.5.

    This is an independent positive case (the inverted-layout test
    proves the new code respects a_is_inh; this test proves it
    produces the right count under the layout that all the existing
    experiments use).
    """
    a_is_inh = assign_ei_identity(s16.N_NEURONS, s16.INHIBITORY_FRACTION)
    n_ex = int((~a_is_inh).sum())

    # 4 slots → partner E neurons (indices 0..3), 4 slots → partner I
    # neurons (indices n_ex..n_ex+3, which lie in the I region).
    e_targets = [s16.N_NEURONS + i for i in range(4)]
    i_targets = [s16.N_NEURONS + n_ex + i for i in range(4)]
    pre_ids = jnp.array([e_targets + i_targets], dtype=jnp.int32)
    pool = _build_test_pool(pre_ids)

    stats = s16._describe_pool(pool, a_is_inh)

    # 4 of 8 slots point to partner E neurons → cross_e_frac = 0.5.
    cross_e_frac = stats["cross_e_frac"]
    cross_frac = stats["cross_frac"]
    assert abs(cross_e_frac - 0.5) < 1e-7, (
        f"cross_e_frac should be 0.5 (4 of 8 slots point to partner E); "
        f"got {cross_e_frac}"
    )
    assert cross_frac == 1.0, (
        f"cross_frac should be 1.0 (all pre_ids in partner space); "
        f"got {cross_frac}"
    )


def test_describe_pool_self_recurrent_slots_are_not_cross_e() -> None:
    """Self-recurrent slots (pre_ids < N_NEURONS) are not counted as cross-E.

    Pre-task-#4 cross_e_mask includes the `pool.pre_ids >= N_NEURONS`
    guard; post-task-#4 mask retains the same guard via `cross_mask`.
    This test pins that guard: B-self-recurrent slots are excluded
    from the cross-E count regardless of a_is_inh.
    """
    n_slots = 8
    # All slots point INTO B's own space (self-recurrent).
    pre_ids = jnp.array([list(range(n_slots))], dtype=jnp.int32)
    pool = _build_test_pool(pre_ids)
    a_is_inh = assign_ei_identity(s16.N_NEURONS, s16.INHIBITORY_FRACTION)

    stats = s16._describe_pool(pool, a_is_inh)

    cross_e_frac = stats["cross_e_frac"]
    cross_frac = stats["cross_frac"]
    assert cross_e_frac == 0.0, (
        f"cross_e_frac should be 0.0 for self-recurrent slots; "
        f"got {cross_e_frac}"
    )
    assert cross_frac == 0.0, (
        f"cross_frac should be 0.0 for self-recurrent slots; "
        f"got {cross_frac}"
    )
