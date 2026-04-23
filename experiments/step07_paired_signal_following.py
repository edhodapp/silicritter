"""Step 7: paired-agent signal-following task.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Asymmetric paired-agent scenario. Agent A ("the signal source") is
externally driven by a piecewise i_ext pattern that puts it into four
distinct firing-rate regimes over the simulation. Its pool is a fixed
random scaffold, the same for every critter in the population.

Agent B ("the predictor") receives no external drive; its only input
is A's spikes through cross-bound slots plus its own recurrent slots.
B's scaffold is evolved by a direct-encoding GA. Fitness is negative
MSE between B's per-window firing rate at window t and A's per-window
firing rate at window t+1 -- B has to anticipate where A goes next,
using only what it can see of A so far.

This is the minimal "awareness of others + learning to predict their
behavior" primitive. At this scale (N = 32 per agent, K = 8 slots /
post, pop = 32, gens = 30) it exercises the two-loop paired
architecture; it is not a demonstration of social *cognition*.

Usage:
    .venv/bin/python experiments/step07_paired_signal_following.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp

from silicritter.ga import (
    Genome,
    decode_to_pool,
    mutate,
    random_population,
    tournament_select,
    uniform_crossover,
)
from silicritter.lif import init_state
from silicritter.paired import (
    PairedState,
    make_pool_for_partner,
    simulate_paired,
)
from silicritter.plasticity import (
    PlasticNetState,
    default_params,
    init_traces,
)
from silicritter.slotpool import SlotPool


# Scale
N_NEURONS: int = 32
K_SLOTS: int = 8
N_TIMESTEPS: int = 2_000
WINDOW_STEPS: int = 100

# GA
POP_SIZE: int = 32
N_GENERATIONS: int = 30
TOURNAMENT_SIZE: int = 3
ELITE_COUNT: int = 2
V_SIGMA: float = 0.01
RATE_SIGMA: float = 0.05
PRE_RESAMPLE_PROB: float = 0.03

# Agent A's piecewise drive: four segments at different mV levels,
# producing four distinct firing regimes B has to anticipate.
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)

# Baseline tonic drive for B. Pure zero-input B cannot fire at the
# small cross-weight scale used here (~0.04 mV per firing A neuron),
# so the population degenerates to all-silent and the GA sees a flat
# fitness landscape. Tonic drive near threshold lets A's cross-input
# modulate B's firing rate rather than gate B between silent and
# spiking. Biologically this is the standard "subthreshold depolarising
# bias" that cortical circuits receive from neuromodulatory arousal.
B_BASELINE_DRIVE_MV: float = 16.0


Scenario = tuple[
    SlotPool,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


def _build_scenario() -> Scenario:
    """Build fixed A scaffold + drive traces for the whole population."""
    seg_len = N_TIMESTEPS // len(A_DRIVE_PROFILE)
    i_ext_a_trace = jnp.concatenate(
        [
            jnp.full((seg_len, N_NEURONS), level, dtype=jnp.float32)
            for level in A_DRIVE_PROFILE
        ]
    )
    i_ext_b_trace = jnp.full(
        (N_TIMESTEPS, N_NEURONS),
        B_BASELINE_DRIVE_MV,
        dtype=jnp.float32,
    )
    valence_a_trace = jnp.zeros((N_TIMESTEPS,), dtype=jnp.float32)
    valence_b_trace = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    adrenaline_trace = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    # A's pool is the same fixed random scaffold for every critter in
    # the population; we pass it in as part of the scenario so the
    # evaluator closes over a single shared A rather than rebuilding.
    return (
        make_pool_for_partner(N_NEURONS, K_SLOTS, jax.random.PRNGKey(777)),
        i_ext_a_trace,
        i_ext_b_trace,
        valence_a_trace,
        valence_b_trace,
        adrenaline_trace,
        adrenaline_trace,
    )


def _prediction_fitness(
    spikes_a: jax.Array, spikes_b: jax.Array
) -> jax.Array:
    """Negative MSE between B's window-t rate and A's window-(t+1) rate."""
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    a_rate = spikes_a.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    b_rate = spikes_b.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    return -jnp.mean((b_rate[:-1] - a_rate[1:]) ** 2)


def _evaluate_single(
    genome_b: Genome,
    scenario: Scenario,
) -> jax.Array:
    """Run one paired critter (fixed A, genome-specified B); return fitness."""
    pool_a, i_ext_a, i_ext_b, val_a, val_b, adr_a, adr_b = scenario
    pool_b = decode_to_pool(genome_b)
    state = PairedState(
        a=PlasticNetState(
            lif=init_state(N_NEURONS),
            pool=pool_a,
            traces=init_traces(n_pre=2 * N_NEURONS, n_post=N_NEURONS),
        ),
        b=PlasticNetState(
            lif=init_state(N_NEURONS),
            pool=pool_b,
            traces=init_traces(n_pre=2 * N_NEURONS, n_post=N_NEURONS),
        ),
    )
    _, spikes_a, spikes_b = simulate_paired(
        state,
        i_ext_a, i_ext_b,
        val_a, val_b,
        adr_a, adr_b,
        default_params(),
    )
    return _prediction_fitness(spikes_a, spikes_b)


def _next_generation(
    population: Genome,
    fitness: jax.Array,
    rng: jax.Array,
    n_pre: int,
) -> Genome:
    """Produce the next-generation population."""
    k_sel, k_pair, k_cross, k_mut = jax.random.split(rng, 4)
    safe_fitness = jnp.where(
        jnp.isnan(fitness), jnp.float32(-jnp.inf), fitness
    )
    elite_idx = jnp.argsort(-safe_fitness)[:ELITE_COUNT]
    elites = Genome(
        pre_ids=population.pre_ids[elite_idx],
        v=population.v[elite_idx],
        plasticity_rate=population.plasticity_rate[elite_idx],
    )
    n_offspring = POP_SIZE - ELITE_COUNT
    parents_a = tournament_select(
        population, fitness, n_offspring, TOURNAMENT_SIZE, k_sel
    )
    parents_b = tournament_select(
        population, fitness, n_offspring, TOURNAMENT_SIZE, k_pair
    )
    cross_keys = jax.random.split(k_cross, n_offspring)
    children = jax.vmap(uniform_crossover)(
        parents_a, parents_b, cross_keys
    )
    mut_keys = jax.random.split(k_mut, n_offspring)
    mutated = jax.vmap(
        lambda g, k: mutate(
            g, k, n_pre, V_SIGMA, RATE_SIGMA, PRE_RESAMPLE_PROB
        )
    )(children, mut_keys)
    return Genome(
        pre_ids=jnp.concatenate(
            [elites.pre_ids, mutated.pre_ids], axis=0
        ),
        v=jnp.concatenate([elites.v, mutated.v], axis=0),
        plasticity_rate=jnp.concatenate(
            [elites.plasticity_rate, mutated.plasticity_rate], axis=0
        ),
    )


def _report_best(
    best_genome: Genome,
    scenario: Scenario,
) -> None:
    """Re-run the best genome and print per-window A vs. B rates."""
    pool_a, i_ext_a, i_ext_b, val_a, val_b, adr_a, adr_b = scenario
    pool_b = decode_to_pool(best_genome)
    state = PairedState(
        a=PlasticNetState(
            lif=init_state(N_NEURONS),
            pool=pool_a,
            traces=init_traces(n_pre=2 * N_NEURONS, n_post=N_NEURONS),
        ),
        b=PlasticNetState(
            lif=init_state(N_NEURONS),
            pool=pool_b,
            traces=init_traces(n_pre=2 * N_NEURONS, n_post=N_NEURONS),
        ),
    )
    _, spikes_a, spikes_b = simulate_paired(
        state, i_ext_a, i_ext_b, val_a, val_b, adr_a, adr_b,
        default_params(),
    )
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    a_rate_hz = (
        spikes_a.astype(jnp.float32)
        .reshape(n_windows, WINDOW_STEPS, N_NEURONS)
        .mean(axis=(1, 2))
    ) * 1000.0
    b_rate_hz = (
        spikes_b.astype(jnp.float32)
        .reshape(n_windows, WINDOW_STEPS, N_NEURONS)
        .mean(axis=(1, 2))
    ) * 1000.0
    print("--- best genome window-by-window ---")
    print(
        " window   A rate (Hz)   B rate (Hz)   "
        "|B(t) - A(t+1)|"
    )
    for w in range(n_windows):
        a_next = (
            float(a_rate_hz[w + 1]) if w + 1 < n_windows else float("nan")
        )
        err = (
            abs(float(b_rate_hz[w]) - a_next)
            if w + 1 < n_windows else float("nan")
        )
        a_str = f"{float(a_rate_hz[w]):8.1f}"
        b_str = f"{float(b_rate_hz[w]):8.1f}"
        err_str = "  n/a" if w + 1 >= n_windows else f"{err:8.2f}"
        print(f"  {w:3d}     {a_str}      {b_str}       {err_str}")


def _apply_scale_overrides(
    n_neurons: int | None,
    slots_per_post: int | None,
    n_timesteps: int | None,
) -> None:
    """Apply optional N / K / T overrides to the module-level constants."""
    # pylint: disable=global-statement
    global N_NEURONS, K_SLOTS, N_TIMESTEPS
    if n_neurons is not None:
        N_NEURONS = n_neurons
    if slots_per_post is not None:
        K_SLOTS = slots_per_post
    if n_timesteps is not None:
        N_TIMESTEPS = n_timesteps


def run(
    seed: int = 0,
    n_neurons: int | None = None,
    slots_per_post: int | None = None,
    n_timesteps: int | None = None,
) -> None:
    """Execute the paired-agent GA and report results."""
    _apply_scale_overrides(n_neurons, slots_per_post, n_timesteps)
    scenario = _build_scenario()
    n_pre_b = 2 * N_NEURONS

    evaluate_pop = jax.jit(
        jax.vmap(lambda g: _evaluate_single(g, scenario))
    )

    rng = jax.random.PRNGKey(seed + 100)
    rng, pop_key = jax.random.split(rng)
    population = random_population(
        POP_SIZE, N_NEURONS, n_pre_b, K_SLOTS, pop_key
    )

    # Warmup.
    _ = evaluate_pop(population).block_until_ready()

    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"pop={POP_SIZE}, gens={N_GENERATIONS}, "
        f"N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}, "
        f"window={WINDOW_STEPS}"
    )
    print(f"A drive profile (mV per segment): {list(A_DRIVE_PROFILE)}")
    print()

    gen_times: list[float] = []
    for gen in range(N_GENERATIONS):
        t0 = time.perf_counter()
        fitness = evaluate_pop(population)
        fitness.block_until_ready()
        gen_times.append(time.perf_counter() - t0)
        best = float(fitness.max())
        mean = float(fitness.mean())
        print(
            f"gen {gen:3d}: best_fit={best:8.3e}  "
            f"mean_fit={mean:8.3e}  "
            f"eval={gen_times[-1]*1000:6.1f} ms"
        )
        if gen == N_GENERATIONS - 1:
            break
        rng, gen_key = jax.random.split(rng)
        population = _next_generation(
            population, fitness, gen_key, n_pre=n_pre_b
        )

    final_fitness = evaluate_pop(population)
    best_idx = int(jnp.argmax(final_fitness))
    best_genome = Genome(
        pre_ids=population.pre_ids[best_idx],
        v=population.v[best_idx],
        plasticity_rate=population.plasticity_rate[best_idx],
    )
    print()
    print(
        f"final best fitness = {float(final_fitness.max()):.3e}  "
        f"(mean = {float(final_fitness.mean()):.3e})"
    )
    print(
        f"total eval time: {sum(gen_times):.1f}s  "
        f"median gen: {statistics.median(gen_times)*1000:.1f}ms"
    )
    print()
    _report_best(best_genome, scenario)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-neurons", type=int, default=None)
    parser.add_argument("--slots-per-post", type=int, default=None)
    parser.add_argument("--n-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        seed=args.seed,
        n_neurons=args.n_neurons,
        slots_per_post=args.slots_per_post,
        n_timesteps=args.n_timesteps,
    )
