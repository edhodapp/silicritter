"""CPPN (Compositional Pattern Producing Network) indirect encoding.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Indirect-encoding genome for the silicritter GA. Direct encoding
(ga.Genome) uses one integer + two floats per slot, which means a
256 x 32 pool has 24,576 evolved parameters and cannot practically
be moved to structured configurations (e.g., "all slots bound to
A with v near v_max") in 30 generations.

A CPPN is a small neural network whose INPUTS are slot coordinates
`(post_idx, slot_idx)` (each normalized to [0, 1]) and whose OUTPUTS
are the raw numbers used to fill the slot: `(pre_id_raw, v_raw,
plasticity_rate_raw)`. The CPPN's WEIGHTS are the evolved genome.
One CPPN genome produces the entire (N, K) pool deterministically
by running the network at every (post, slot) coordinate.

Key advantage over direct encoding: large-scale structured patterns
("every slot cross-bound to A with high v") are expressible as a
single weight configuration of a small network -- ~50-100 weights
vs. ~24,000 direct parameters. The search space shrinks by roughly
two orders of magnitude and spatially-coherent configurations become
reachable by a GA within tens of generations rather than tens of
thousands.

Architecture here is the simplest useful one: a fixed-topology 2-layer
MLP with tanh hidden activations and sigmoid-scaled outputs. Inputs
(including a bias term) -> hidden layer with configurable width -> 3
output heads (pre_id, v, plasticity_rate). Total parameters for the
default hidden width of 8: 3*8 + 9*3 = 51. Evolving richer CPPNs
(mixed activations, variable topology per HyperNEAT / NEAT) is a
future step; this module validates the encoding class.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from silicritter.slotpool import SlotPool


# Input dimensionality: (post_coord, slot_coord, bias=1).
INPUT_DIM: int = 3
# Three output heads: pre_id, v, plasticity_rate.
OUTPUT_DIM: int = 3


class CPPNGenome(NamedTuple):
    """Weights of a 2-layer MLP CPPN.

    Attributes:
        w1: input-to-hidden weight matrix, shape (INPUT_DIM, hidden_dim).
            A single column of w1 contains the weights from (post_coord,
            slot_coord, bias) into one hidden unit.
        w2: hidden-to-output weight matrix, shape
            (hidden_dim + 1, OUTPUT_DIM). The extra row is the bias
            connection from the hidden layer to each output head.
    """

    w1: jax.Array
    w2: jax.Array


def random_cppn_genome(
    hidden_dim: int,
    rng: jax.Array,
    init_scale: float = 1.0,
) -> CPPNGenome:
    """Initialize a single CPPN genome with Gaussian-random weights."""
    k1, k2 = jax.random.split(rng, 2)
    w1 = (
        jax.random.normal(k1, (INPUT_DIM, hidden_dim), dtype=jnp.float32)
        * init_scale
    )
    w2 = (
        jax.random.normal(
            k2, (hidden_dim + 1, OUTPUT_DIM), dtype=jnp.float32
        )
        * init_scale
    )
    return CPPNGenome(w1=w1, w2=w2)


def random_cppn_population(
    pop_size: int,
    hidden_dim: int,
    rng: jax.Array,
    init_scale: float = 1.0,
) -> CPPNGenome:
    """Build `pop_size` random CPPN genomes stacked on the leading axis."""
    keys = jax.random.split(rng, pop_size)
    stacked = jax.vmap(
        lambda k: random_cppn_genome(hidden_dim, k, init_scale)
    )(keys)
    return stacked


def _cppn_forward(genome: CPPNGenome, inputs: jax.Array) -> jax.Array:
    """Forward pass: (N, INPUT_DIM) inputs -> (N, OUTPUT_DIM) outputs."""
    hidden = jnp.tanh(inputs @ genome.w1)
    hidden_with_bias = jnp.concatenate(
        [hidden, jnp.ones((hidden.shape[0], 1), dtype=hidden.dtype)],
        axis=-1,
    )
    return hidden_with_bias @ genome.w2


def decode_cppn_to_pool(
    genome: CPPNGenome,
    n_post: int,
    n_pre: int,
    slots_per_post: int,
    v_max: float,
) -> SlotPool:
    """Run the CPPN at every (post, slot) coordinate to fill a SlotPool.

    Inputs to the CPPN for each (post_idx, slot_idx) pair:

      (post_idx / (n_post - 1), slot_idx / (slots_per_post - 1), 1.0)

    Normalization means the network sees coordinates in [0, 1] regardless
    of pool scale. The bias column lets the network learn constant
    offsets without an explicit bias term.

    Outputs are passed through sigmoid activations and scaled into their
    respective ranges:

      pre_id         = floor(sigmoid(pre_id_raw) * n_pre), clipped to
                       [0, n_pre - 1]
      v              = sigmoid(v_raw) * v_max
      plasticity_rate= sigmoid(plasticity_rate_raw)

    Every slot is marked active; release_counter is zero-initialized.
    """
    post_coords = jnp.linspace(0.0, 1.0, n_post, dtype=jnp.float32)
    slot_coords = jnp.linspace(
        0.0, 1.0, slots_per_post, dtype=jnp.float32
    )
    post_grid, slot_grid = jnp.meshgrid(
        post_coords, slot_coords, indexing="ij"
    )
    ones = jnp.ones_like(post_grid)
    inputs = jnp.stack([post_grid, slot_grid, ones], axis=-1)
    flat_inputs = inputs.reshape(-1, INPUT_DIM)

    flat_outputs = _cppn_forward(genome, flat_inputs)
    outputs = flat_outputs.reshape(n_post, slots_per_post, OUTPUT_DIM)
    pre_id_raw = outputs[:, :, 0]
    v_raw = outputs[:, :, 1]
    rate_raw = outputs[:, :, 2]

    pre_id_norm = jax.nn.sigmoid(pre_id_raw)
    pre_ids = jnp.floor(pre_id_norm * n_pre).astype(jnp.int32)
    pre_ids = jnp.minimum(pre_ids, n_pre - 1)

    v = jax.nn.sigmoid(v_raw) * v_max
    plasticity_rate = jax.nn.sigmoid(rate_raw)

    active = jnp.ones_like(v, dtype=jnp.bool_)
    release_counter = jnp.zeros_like(pre_ids, dtype=jnp.int32)
    return SlotPool(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=plasticity_rate,
        active=active,
        release_counter=release_counter,
    )


def tournament_select_cppn(
    population: CPPNGenome,
    fitness: jax.Array,
    n_winners: int,
    tournament_size: int,
    rng: jax.Array,
) -> CPPNGenome:
    """Same tournament semantics as ga.tournament_select, over CPPN genomes."""
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
    return CPPNGenome(
        w1=population.w1[winner_idx],
        w2=population.w2[winner_idx],
    )


def uniform_crossover_cppn(
    parent_a: CPPNGenome,
    parent_b: CPPNGenome,
    rng: jax.Array,
) -> CPPNGenome:
    """Per-weight uniform crossover between two CPPN genomes."""
    k1, k2 = jax.random.split(rng, 2)
    mask_w1 = jax.random.bernoulli(k1, p=0.5, shape=parent_a.w1.shape)
    mask_w2 = jax.random.bernoulli(k2, p=0.5, shape=parent_a.w2.shape)
    return CPPNGenome(
        w1=jnp.where(mask_w1, parent_a.w1, parent_b.w1),
        w2=jnp.where(mask_w2, parent_a.w2, parent_b.w2),
    )


def mutate_cppn(
    genome: CPPNGenome,
    rng: jax.Array,
    sigma: float = 0.1,
) -> CPPNGenome:
    """Gaussian perturbation on every weight (additive noise)."""
    k1, k2 = jax.random.split(rng, 2)
    w1_noise = (
        jax.random.normal(k1, genome.w1.shape, dtype=jnp.float32) * sigma
    )
    w2_noise = (
        jax.random.normal(k2, genome.w2.shape, dtype=jnp.float32) * sigma
    )
    return CPPNGenome(
        w1=genome.w1 + w1_noise,
        w2=genome.w2 + w2_noise,
    )
