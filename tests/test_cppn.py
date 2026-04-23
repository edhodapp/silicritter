"""Behavioral tests for the CPPN indirect-encoding module.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.cppn import (
    INPUT_DIM,
    OUTPUT_DIM,
    CPPNGenome,
    decode_cppn_to_pool,
    mutate_cppn,
    random_cppn_genome,
    random_cppn_population,
    tournament_select_cppn,
    uniform_crossover_cppn,
)
from silicritter.slotpool import SlotPool


def test_random_cppn_genome_shapes() -> None:
    """random_cppn_genome returns correctly-shaped weight matrices."""
    hidden_dim = 8
    g = random_cppn_genome(hidden_dim, jax.random.PRNGKey(0))
    assert g.w1.shape == (INPUT_DIM, hidden_dim)
    assert g.w2.shape == (hidden_dim + 1, OUTPUT_DIM)
    assert g.w1.dtype == jnp.float32
    assert g.w2.dtype == jnp.float32


def test_random_cppn_population_batches() -> None:
    """random_cppn_population stacks pop_size independent genomes."""
    pop_size, hidden_dim = 4, 6
    pop = random_cppn_population(
        pop_size, hidden_dim, jax.random.PRNGKey(1)
    )
    assert pop.w1.shape == (pop_size, INPUT_DIM, hidden_dim)
    assert pop.w2.shape == (pop_size, hidden_dim + 1, OUTPUT_DIM)
    # Different members should differ (vmap produces distinct genomes).
    assert not bool(jnp.allclose(pop.w1[0], pop.w1[1]))


def test_decode_produces_valid_slot_pool() -> None:
    """decode_cppn_to_pool returns a SlotPool with correct shapes/bounds."""
    n_post, n_pre, k = 16, 32, 4
    v_max = 2.0
    g = random_cppn_genome(8, jax.random.PRNGKey(2))
    pool = decode_cppn_to_pool(g, n_post, n_pre, k, v_max)
    assert isinstance(pool, SlotPool)
    assert pool.pre_ids.shape == (n_post, k)
    assert pool.v.shape == (n_post, k)
    assert pool.plasticity_rate.shape == (n_post, k)
    assert pool.active.shape == (n_post, k)
    assert pool.release_counter.shape == (n_post, k)
    # Bounds enforced by decoder.
    assert bool(jnp.all(pool.pre_ids >= 0))
    assert bool(jnp.all(pool.pre_ids < n_pre))
    assert bool(jnp.all(pool.v >= 0.0))
    assert bool(jnp.all(pool.v <= v_max))
    assert bool(jnp.all(pool.plasticity_rate >= 0.0))
    assert bool(jnp.all(pool.plasticity_rate <= 1.0))
    assert bool(jnp.all(pool.active))
    assert bool(jnp.all(pool.release_counter == 0))


def test_decode_is_deterministic() -> None:
    """Same genome + same pool size produce bit-identical SlotPools."""
    g = random_cppn_genome(8, jax.random.PRNGKey(3))
    pool_a = decode_cppn_to_pool(g, 10, 20, 4, 2.0)
    pool_b = decode_cppn_to_pool(g, 10, 20, 4, 2.0)
    assert bool(jnp.all(pool_a.pre_ids == pool_b.pre_ids))
    assert bool(jnp.all(pool_a.v == pool_b.v))
    assert bool(jnp.all(pool_a.plasticity_rate == pool_b.plasticity_rate))


def test_decode_distinct_genomes_produce_distinct_pools() -> None:
    """Distinct genomes typically produce different SlotPools."""
    g1 = random_cppn_genome(8, jax.random.PRNGKey(4))
    g2 = random_cppn_genome(8, jax.random.PRNGKey(5))
    pool_1 = decode_cppn_to_pool(g1, 16, 32, 4, 2.0)
    pool_2 = decode_cppn_to_pool(g2, 16, 32, 4, 2.0)
    # Not EVERY slot needs to differ, but with random weights most should.
    assert not bool(jnp.all(pool_1.pre_ids == pool_2.pre_ids))
    assert not bool(jnp.allclose(pool_1.v, pool_2.v))


def test_mutate_cppn_zero_sigma_is_identity() -> None:
    """Mutation with sigma=0 returns the genome unchanged."""
    g = random_cppn_genome(8, jax.random.PRNGKey(6))
    mut = mutate_cppn(g, jax.random.PRNGKey(7), sigma=0.0)
    assert bool(jnp.all(mut.w1 == g.w1))
    assert bool(jnp.all(mut.w2 == g.w2))


def test_mutate_cppn_positive_sigma_changes_weights() -> None:
    """Mutation with positive sigma actually moves the weights."""
    g = random_cppn_genome(8, jax.random.PRNGKey(8))
    mut = mutate_cppn(g, jax.random.PRNGKey(9), sigma=0.1)
    # Every weight should differ at nonzero sigma.
    assert not bool(jnp.allclose(mut.w1, g.w1))
    assert not bool(jnp.allclose(mut.w2, g.w2))


def test_tournament_select_cppn_picks_argmax_per_tournament() -> None:
    """Tournament-selected winners match the argmax of each random draw."""
    pop_size, n_winners, tournament_size = 6, 4, 6
    pop = random_cppn_population(pop_size, 8, jax.random.PRNGKey(10))
    fitness = jnp.array(
        [0.1, 0.9, 0.3, 0.5, 0.7, 0.2], dtype=jnp.float32
    )
    rng = jax.random.PRNGKey(11)
    winners = tournament_select_cppn(
        pop, fitness, n_winners, tournament_size, rng
    )
    # Replicate the draw and verify argmax.
    contestant_idx = jax.random.randint(
        rng,
        (n_winners, tournament_size),
        minval=0,
        maxval=pop_size,
        dtype=jnp.int32,
    )
    for i in range(n_winners):
        idx_row = contestant_idx[i]
        expected = int(idx_row[int(jnp.argmax(fitness[idx_row]))])
        assert bool(jnp.all(winners.w1[i] == pop.w1[expected]))
        assert bool(jnp.all(winners.w2[i] == pop.w2[expected]))


def test_uniform_crossover_cppn_contributes_from_both_parents() -> None:
    """Uniform crossover produces a child where both parents contribute."""
    parent_a = random_cppn_genome(8, jax.random.PRNGKey(12))
    parent_b = random_cppn_genome(8, jax.random.PRNGKey(13))
    child = uniform_crossover_cppn(
        parent_a, parent_b, jax.random.PRNGKey(14)
    )
    # Every position should match one parent (structural property).
    from_a_w1 = child.w1 == parent_a.w1
    from_b_w1 = child.w1 == parent_b.w1
    assert bool(jnp.all(from_a_w1 | from_b_w1))
    # And both parents should contribute at least one weight.
    only_a_w1 = from_a_w1 & ~from_b_w1
    only_b_w1 = from_b_w1 & ~from_a_w1
    assert bool(jnp.any(only_a_w1))
    assert bool(jnp.any(only_b_w1))


def test_cppn_genome_can_produce_near_uniform_cross_pool() -> None:
    """Hand-crafted CPPN produces a near-uniform all-cross pool.

    This is a sanity check on expressivity: a generator with strongly
    positive pre_id bias and large-positive v bias should produce a
    pool where most slots point into the upper half of the pre index
    range (the A-side in paired scenarios) with v near v_max.
    """
    hidden_dim = 4
    # Construct a CPPN whose pre_id head biases strongly toward +inf,
    # whose v head also biases strongly toward +inf, and whose rate head
    # is unconstrained. Zero out the hidden weights so every hidden unit
    # outputs the same (tanh(bias)) and the output bias dominates.
    w1 = jnp.zeros((INPUT_DIM, hidden_dim), dtype=jnp.float32)
    # Put bias on the row reserved for the bias input (index 2).
    w1 = w1.at[2, :].set(0.0)
    # w2: strong output biases on pre_id and v, zero on rate.
    w2 = jnp.zeros((hidden_dim + 1, OUTPUT_DIM), dtype=jnp.float32)
    w2 = w2.at[-1, 0].set(5.0)  # pre_id bias: sigmoid(5) ~ 0.993 -> A side
    w2 = w2.at[-1, 1].set(5.0)  # v bias: sigmoid(5) * v_max
    genome = CPPNGenome(w1=w1, w2=w2)

    n_post, n_pre, k = 16, 64, 8
    v_max = 2.0
    pool = decode_cppn_to_pool(genome, n_post, n_pre, k, v_max)
    # pre_ids should all be in the upper half (cross side) of [0, n_pre).
    assert bool(jnp.all(pool.pre_ids >= n_pre // 2))
    # v should be near v_max (sigmoid(5) * 2.0 ~ 1.987).
    assert bool(jnp.all(pool.v > 0.9 * v_max))
