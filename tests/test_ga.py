"""Behavioral tests for the GA primitives.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.ga import (
    decode_to_pool,
    mutate,
    random_genome,
    random_population,
    tournament_select,
    uniform_crossover,
)
from silicritter.slotpool import SlotPool


def test_random_genome_shapes_and_bounds() -> None:
    """random_genome produces correctly-shaped, in-bounds arrays."""
    n_post, n_pre, k = 6, 8, 4
    g = random_genome(n_post, n_pre, k, jax.random.PRNGKey(0))
    assert g.pre_ids.shape == (n_post, k)
    assert g.v.shape == (n_post, k)
    assert g.plasticity_rate.shape == (n_post, k)
    assert g.pre_ids.dtype == jnp.int32
    assert g.v.dtype == jnp.float32
    assert g.plasticity_rate.dtype == jnp.float32
    assert bool(jnp.all(g.pre_ids >= 0))
    assert bool(jnp.all(g.pre_ids < n_pre))
    assert bool(jnp.all(g.v >= 0.0))
    assert bool(jnp.all(g.plasticity_rate >= 0.0))
    assert bool(jnp.all(g.plasticity_rate <= 1.0))


def test_random_population_batches_correctly() -> None:
    """random_population stacks pop_size independent genomes."""
    pop_size, n_post, n_pre, k = 5, 4, 6, 3
    pop = random_population(
        pop_size, n_post, n_pre, k, jax.random.PRNGKey(1)
    )
    assert pop.pre_ids.shape == (pop_size, n_post, k)
    assert pop.v.shape == (pop_size, n_post, k)
    assert pop.plasticity_rate.shape == (pop_size, n_post, k)
    # Different members should not all be identical (vmap over keys).
    v0 = pop.v[0]
    v1 = pop.v[1]
    assert not bool(jnp.allclose(v0, v1))


def test_decode_to_pool_all_active() -> None:
    """decode_to_pool returns a SlotPool with every slot active."""
    g = random_genome(5, 7, 2, jax.random.PRNGKey(2))
    pool = decode_to_pool(g)
    assert isinstance(pool, SlotPool)
    assert bool(jnp.all(pool.active))
    assert bool(jnp.all(pool.pre_ids == g.pre_ids))
    assert bool(jnp.all(pool.v == g.v))
    assert bool(jnp.all(pool.plasticity_rate == g.plasticity_rate))


def test_tournament_select_picks_highest_fitness() -> None:
    """Each tournament's winner has max fitness among that tournament's draws.

    Sampling is with replacement, so a tournament of size == pop_size does
    not necessarily cover every index; we verify the selection logic by
    replicating the random draw with the same key and checking that the
    returned winner is the argmax over that specific draw.
    """
    pop_size, n_winners, tournament_size = 6, 4, 6
    pop = random_population(pop_size, 3, 4, 2, jax.random.PRNGKey(3))
    fitness = jnp.array([0.1, 0.9, 0.3, 0.5, 0.7, 0.2], dtype=jnp.float32)
    rng = jax.random.PRNGKey(4)

    winners = tournament_select(
        pop, fitness, n_winners, tournament_size, rng
    )

    # Replicate the draw to verify the selection produces the argmax per
    # tournament. Deterministic under the same key.
    contestant_idx = jax.random.randint(
        rng,
        (n_winners, tournament_size),
        minval=0,
        maxval=pop_size,
        dtype=jnp.int32,
    )
    for i in range(n_winners):
        idx_row = contestant_idx[i]
        expected = idx_row[int(jnp.argmax(fitness[idx_row]))]
        assert bool(
            jnp.all(winners.pre_ids[i] == pop.pre_ids[int(expected)])
        )
        assert bool(jnp.all(winners.v[i] == pop.v[int(expected)]))
        assert bool(
            jnp.all(
                winners.plasticity_rate[i]
                == pop.plasticity_rate[int(expected)]
            )
        )


def test_uniform_crossover_child_inherits_from_parents() -> None:
    """Every child slot matches one parent, and both parents contribute."""
    n_post, n_pre, k = 4, 5, 3
    parent_a = random_genome(n_post, n_pre, k, jax.random.PRNGKey(5))
    parent_b = random_genome(n_post, n_pre, k, jax.random.PRNGKey(6))
    child = uniform_crossover(parent_a, parent_b, jax.random.PRNGKey(7))
    from_a = (
        (child.pre_ids == parent_a.pre_ids)
        & (child.v == parent_a.v)
        & (child.plasticity_rate == parent_a.plasticity_rate)
    )
    from_b = (
        (child.pre_ids == parent_b.pre_ids)
        & (child.v == parent_b.v)
        & (child.plasticity_rate == parent_b.plasticity_rate)
    )
    # Every slot comes from one parent (structural correctness).
    assert bool(jnp.all(from_a | from_b))
    # With 12 slots under a fair coin, probability of all-from-one-parent
    # is 2 / 2^12 ~ 5e-4; the fixed seed must produce a mixed child to
    # actually exercise the mask. Verify both parents contribute at least
    # one slot so a buggy "always return parent_a" would fail this test.
    # Exclude slots where both parents happen to be identical (possible
    # but vanishingly rare with random draws).
    only_from_a = from_a & ~from_b
    only_from_b = from_b & ~from_a
    assert bool(jnp.any(only_from_a))
    assert bool(jnp.any(only_from_b))


def test_mutate_preserves_bounds() -> None:
    """Mutation respects v >= 0, rate in [0, 1], pre_ids in [0, n_pre)."""
    n_pre = 10
    g = random_genome(4, n_pre, 3, jax.random.PRNGKey(8))
    mutated = mutate(
        g,
        jax.random.PRNGKey(9),
        n_pre=n_pre,
        v_sigma=0.5,
        rate_sigma=0.5,
        pre_resample_prob=0.5,
    )
    assert bool(jnp.all(mutated.v >= 0.0))
    assert bool(jnp.all(mutated.plasticity_rate >= 0.0))
    assert bool(jnp.all(mutated.plasticity_rate <= 1.0))
    assert bool(jnp.all(mutated.pre_ids >= 0))
    assert bool(jnp.all(mutated.pre_ids < n_pre))


def test_mutate_with_zero_noise_is_identity() -> None:
    """All three fields remain unchanged when noise params are zero."""
    n_pre = 7
    g = random_genome(3, n_pre, 2, jax.random.PRNGKey(10))
    mutated = mutate(
        g,
        jax.random.PRNGKey(11),
        n_pre=n_pre,
        v_sigma=0.0,
        rate_sigma=0.0,
        pre_resample_prob=0.0,
    )
    # With zero resample probability, pre_ids also unchanged.
    assert bool(jnp.all(mutated.v == g.v))
    assert bool(jnp.all(mutated.plasticity_rate == g.plasticity_rate))
    assert bool(jnp.all(mutated.pre_ids == g.pre_ids))
