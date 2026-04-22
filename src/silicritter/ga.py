"""Genetic-algorithm primitives for silicritter outer-loop search.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 5 scaffolding: direct-encoding GA primitives that treat a full
slot-pool configuration (pre_ids, v, plasticity_rate) as a genome.
Population-level operations (random init, tournament selection,
uniform crossover, mutation) plus a decoder from a single genome to
a SlotPool. Fitness evaluation and the main GA loop live in the
experiment script since they depend on the task at hand.

Direct encoding is tractable at small N * K but scales poorly;
indirect encoding (CPPN / developmental rules) is the path for
larger networks, per the D004-era discussion. This module exists to
validate the two-loop structure on tiny problems, not to carry the
eventual GA workload.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from silicritter.slotpool import SlotPool


class Genome(NamedTuple):
    """Direct-encoded silicritter genome.

    Attributes:
        pre_ids: int32 presynaptic index for each slot, shape (N_post, K).
        v: float32 initial per-slot analog weight, shape (N_post, K).
        plasticity_rate: float32 per-slot plasticity rate in [0, 1],
            shape (N_post, K).
    """

    pre_ids: jax.Array
    v: jax.Array
    plasticity_rate: jax.Array


def decode_to_pool(genome: Genome) -> SlotPool:
    """Decode a Genome into a SlotPool with all slots active."""
    active = jnp.ones_like(genome.v, dtype=jnp.bool_)
    return SlotPool(
        pre_ids=genome.pre_ids,
        v=genome.v,
        plasticity_rate=genome.plasticity_rate,
        active=active,
    )


def random_genome(
    n_post: int,
    n_pre: int,
    slots_per_post: int,
    rng: jax.Array,
    v_scale: float = 0.05,
) -> Genome:
    """Draw a random genome for a single individual."""
    k_ids, k_v, k_plast = jax.random.split(rng, 3)
    pre_ids = jax.random.randint(
        k_ids,
        (n_post, slots_per_post),
        minval=0,
        maxval=n_pre,
        dtype=jnp.int32,
    )
    # Half-normal for non-negative v, matching slotpool.init_random.
    v = (
        jnp.abs(
            jax.random.normal(
                k_v, (n_post, slots_per_post), dtype=jnp.float32
            )
        )
        * v_scale
    )
    plasticity_rate = jax.random.uniform(
        k_plast,
        (n_post, slots_per_post),
        minval=0.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    return Genome(
        pre_ids=pre_ids, v=v, plasticity_rate=plasticity_rate
    )


def random_population(
    pop_size: int,
    n_post: int,
    n_pre: int,
    slots_per_post: int,
    rng: jax.Array,
    v_scale: float = 0.05,
) -> Genome:
    """Draw `pop_size` random genomes as stacked arrays."""
    keys = jax.random.split(rng, pop_size)
    stacked = jax.vmap(
        lambda k: random_genome(n_post, n_pre, slots_per_post, k, v_scale)
    )(keys)
    return stacked


def tournament_select(
    population: Genome,
    fitness: jax.Array,
    n_winners: int,
    tournament_size: int,
    rng: jax.Array,
) -> Genome:
    """Run `n_winners` independent tournaments, return the winning genomes.

    Each tournament draws `tournament_size` contestants from the
    population uniformly, and the contestant with the highest fitness
    is selected. Sampling is with replacement both across tournaments
    and within each tournament, so a tournament can include duplicate
    contestants. This weakly inflates selection pressure on already-
    frequent high-fitness individuals; acceptable at small-population
    scale, worth revisiting if selection pressure becomes a problem.
    """
    pop_size = fitness.shape[0]
    contestant_idx = jax.random.randint(
        rng,
        (n_winners, tournament_size),
        minval=0,
        maxval=pop_size,
        dtype=jnp.int32,
    )
    contestant_fitness = fitness[contestant_idx]
    local_winner = jnp.argmax(contestant_fitness, axis=1)
    winner_idx = contestant_idx[jnp.arange(n_winners), local_winner]
    return Genome(
        pre_ids=population.pre_ids[winner_idx],
        v=population.v[winner_idx],
        plasticity_rate=population.plasticity_rate[winner_idx],
    )


def uniform_crossover(
    parent_a: Genome,
    parent_b: Genome,
    rng: jax.Array,
) -> Genome:
    """Per-slot uniform crossover between two parents.

    For each slot, a fair coin decides whether the child inherits
    from parent_a or parent_b. The three fields (pre_ids, v,
    plasticity_rate) move together per slot -- structurally a slot
    is either parent_a's or parent_b's in its entirety.

    Baldwin-interference caveat: during the inner plastic simulation,
    v and plasticity_rate co-evolve via STDP. A child slot inherits
    the *initial* triplet (pre_ids, v, plasticity_rate) from one
    parent, but neither parent's *plasticized trajectory* transfers,
    so this is not true hereditary inheritance of learned weights.
    That's a deliberate choice (we evolve the innate scaffold, not
    the lifetime-shaped weights), but it means crossover produces
    children whose initial state may land in a region neither parent
    explored during its plastic lifetime.

    Assumes all three fields of both parents share the same (N_post, K)
    shape; the coin-flip mask is built against `parent_a.v.shape`.
    """
    mask = jax.random.bernoulli(rng, p=0.5, shape=parent_a.v.shape)
    return Genome(
        pre_ids=jnp.where(mask, parent_a.pre_ids, parent_b.pre_ids),
        v=jnp.where(mask, parent_a.v, parent_b.v),
        plasticity_rate=jnp.where(
            mask, parent_a.plasticity_rate, parent_b.plasticity_rate
        ),
    )


def mutate(
    genome: Genome,
    rng: jax.Array,
    n_pre: int,
    v_sigma: float = 0.02,
    rate_sigma: float = 0.1,
    pre_resample_prob: float = 0.05,
) -> Genome:
    """Mutate a genome in place-like fashion (returns new Genome).

    - Each slot's v gets zero-mean Gaussian noise (std v_sigma),
      clipped to [0, infinity) to keep v non-negative.
    - Each slot's plasticity_rate gets Gaussian noise (std rate_sigma),
      clipped to [0, 1].
    - Each slot's pre_id is resampled uniformly with probability
      pre_resample_prob.
    """
    k_v, k_rate, k_pre_mask, k_pre_sample = jax.random.split(rng, 4)

    v_noise = (
        jax.random.normal(k_v, genome.v.shape, dtype=jnp.float32) * v_sigma
    )
    new_v = jnp.maximum(genome.v + v_noise, jnp.float32(0.0))

    rate_noise = (
        jax.random.normal(
            k_rate, genome.plasticity_rate.shape, dtype=jnp.float32
        )
        * rate_sigma
    )
    new_rate = jnp.clip(genome.plasticity_rate + rate_noise, 0.0, 1.0)

    resample = jax.random.bernoulli(
        k_pre_mask, p=pre_resample_prob, shape=genome.pre_ids.shape
    )
    fresh_pre = jax.random.randint(
        k_pre_sample,
        genome.pre_ids.shape,
        minval=0,
        maxval=n_pre,
        dtype=jnp.int32,
    )
    new_pre = jnp.where(resample, fresh_pre, genome.pre_ids)

    return Genome(pre_ids=new_pre, v=new_v, plasticity_rate=new_rate)
