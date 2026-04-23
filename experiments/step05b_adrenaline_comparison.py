"""Step 5.5: side-by-side comparison of five adrenaline gain mechanisms.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Runs the same step-5 GA (direct-encoding, tournament selection,
uniform crossover, bounded mutation, elitism) against the same target-
firing-rate task under each of the five gain_mode mechanisms exposed
by plasticity.py. Reports per-mode final best fitness and the
segment-by-segment tracking error so the mechanisms can be compared
on equal footing.

Exit criterion from the discussion preceding this experiment: declare
a winner if any mechanism produces all four adrenaline segments
tracking within 5 Hz of target. Otherwise the GA (direct encoding at
N=32, pop=48) is the limiter rather than the gain mechanism, and the
finding is "no multiplicative-family formulation solves this task at
this GA scale."

Usage:
    .venv/bin/python experiments/step05b_adrenaline_comparison.py
"""

from __future__ import annotations

import statistics
import time
from typing import NamedTuple

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
from silicritter.plasticity import (
    GainMode,
    PlasticNetState,
    default_params,
    init_traces,
    simulate_plastic,
)


# Scenario (matches step 5 exactly).
N_NEURONS: int = 32
K_SLOTS: int = 8
N_TIMESTEPS: int = 2_000
ADRENALINE_PROFILE: tuple[float, ...] = (1.0, 1.5, 0.8, 1.2)
BASELINE_RATE_HZ: float = 40.0
WINDOW_STEPS: int = 100

# GA hyperparameters (match step 5).
POP_SIZE: int = 48
N_GENERATIONS: int = 80
TOURNAMENT_SIZE: int = 3
ELITE_COUNT: int = 2
V_SIGMA: float = 0.01
RATE_SIGMA: float = 0.05
PRE_RESAMPLE_PROB: float = 0.03

GAIN_MODES: tuple[GainMode, ...] = (
    "multiplicative",
    "multiplicative_mild",
    "additive",
    "tau_m_scale",
    "threshold_shift",
)

# Exit criterion: all segments within this Hz of target.
SUCCESS_HZ_TOLERANCE: float = 5.0


class ModeResult(NamedTuple):
    """Summary of one mode's GA run."""

    mode: GainMode
    best_fitness: float
    mean_fitness: float
    seg_achieved_hz: tuple[float, ...]
    seg_error_hz: tuple[float, ...]
    total_time_s: float


def _build_scenario(
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build (i_ext, valence, adrenaline, target_rate_per_window)."""
    drive_key = jax.random.PRNGKey(seed)
    i_ext_trace = jax.random.uniform(
        drive_key,
        (N_TIMESTEPS, N_NEURONS),
        minval=17.0,
        maxval=22.0,
        dtype=jnp.float32,
    )
    valence_trace = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    seg_len = N_TIMESTEPS // len(ADRENALINE_PROFILE)
    adrenaline_trace = jnp.concatenate(
        [
            jnp.full((seg_len,), level, dtype=jnp.float32)
            for level in ADRENALINE_PROFILE
        ]
    )
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    window_adrenaline = adrenaline_trace.reshape(
        n_windows, WINDOW_STEPS
    ).mean(axis=1)
    target_rate = BASELINE_RATE_HZ * window_adrenaline
    return i_ext_trace, valence_trace, adrenaline_trace, target_rate


def _fitness_from_spikes(
    spikes: jax.Array, target_rate: jax.Array
) -> jax.Array:
    """Negative MSE between per-window firing rate (Hz) and target."""
    per_step_rate_hz = spikes.astype(jnp.float32).mean(axis=1) * 1000.0
    n_windows = target_rate.shape[0]
    window_rate = per_step_rate_hz.reshape(n_windows, -1).mean(axis=1)
    return -jnp.mean((window_rate - target_rate) ** 2)


def _evaluate_single(
    genome: Genome,
    i_ext_trace: jax.Array,
    valence_trace: jax.Array,
    adrenaline_trace: jax.Array,
    target_rate: jax.Array,
    mode: GainMode,
) -> jax.Array:
    """Run one critter lifetime under `mode`; return fitness."""
    pool = decode_to_pool(genome)
    state = PlasticNetState(
        lif=init_state(N_NEURONS),
        pool=pool,
        traces=init_traces(n_pre=N_NEURONS, n_post=N_NEURONS),
    )
    params = default_params()
    _, spikes = simulate_plastic(
        state,
        i_ext_trace,
        valence_trace,
        adrenaline_trace,
        params,
        gain_mode=mode,
    )
    return _fitness_from_spikes(spikes, target_rate)


def _next_generation(
    population: Genome,
    fitness: jax.Array,
    rng: jax.Array,
    n_pre: int,
) -> Genome:
    """Produce the next generation: elites + tournament + crossover + mutate."""
    k_select, k_pair, k_cross, k_mut = jax.random.split(rng, 4)
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
        population, fitness, n_offspring, TOURNAMENT_SIZE, k_select
    )
    parents_b = tournament_select(
        population, fitness, n_offspring, TOURNAMENT_SIZE, k_pair
    )
    cross_keys = jax.random.split(k_cross, n_offspring)
    children = jax.vmap(uniform_crossover)(parents_a, parents_b, cross_keys)
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


def _run_one_mode(
    mode: GainMode,
    scenario: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    seed: int,
) -> ModeResult:
    """Run the full GA for one gain_mode; return summary stats."""
    i_ext_trace, valence_trace, adrenaline_trace, target_rate = scenario

    evaluate_pop = jax.jit(
        jax.vmap(
            lambda g: _evaluate_single(
                g,
                i_ext_trace,
                valence_trace,
                adrenaline_trace,
                target_rate,
                mode,
            )
        )
    )

    # Seed is shared across modes on purpose: identical initial populations
    # and identical mutation / selection noise sequences make fitness
    # differences reflect the gain mechanism rather than GA noise. This
    # is a controlled-variance comparison, not a bias.
    rng = jax.random.PRNGKey(seed + 100)
    rng, pop_key = jax.random.split(rng)
    population = random_population(
        POP_SIZE, N_NEURONS, N_NEURONS, K_SLOTS, pop_key
    )
    _ = evaluate_pop(population).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(N_GENERATIONS):
        fitness = evaluate_pop(population)
        fitness.block_until_ready()
        rng, gen_key = jax.random.split(rng)
        population = _next_generation(
            population, fitness, gen_key, n_pre=N_NEURONS
        )
    fitness = evaluate_pop(population)
    fitness.block_until_ready()
    total_time = time.perf_counter() - t0

    best_idx = int(jnp.argmax(fitness))
    best_genome = Genome(
        pre_ids=population.pre_ids[best_idx],
        v=population.v[best_idx],
        plasticity_rate=population.plasticity_rate[best_idx],
    )
    pool = decode_to_pool(best_genome)
    state = PlasticNetState(
        lif=init_state(N_NEURONS),
        pool=pool,
        traces=init_traces(n_pre=N_NEURONS, n_post=N_NEURONS),
    )
    _, best_spikes = simulate_plastic(
        state,
        i_ext_trace,
        valence_trace,
        adrenaline_trace,
        default_params(),
        gain_mode=mode,
    )
    per_step_rate = best_spikes.astype(jnp.float32).mean(axis=1) * 1000.0
    n_segments = len(ADRENALINE_PROFILE)
    achieved_per_segment = per_step_rate.reshape(n_segments, -1).mean(axis=1)
    targets_per_segment = target_rate.reshape(n_segments, -1).mean(axis=1)

    seg_achieved = tuple(float(x) for x in achieved_per_segment)
    seg_error = tuple(
        float(abs(a - t))
        for a, t in zip(achieved_per_segment, targets_per_segment)
    )
    return ModeResult(
        mode=mode,
        best_fitness=float(fitness.max()),
        mean_fitness=float(fitness.mean()),
        seg_achieved_hz=seg_achieved,
        seg_error_hz=seg_error,
        total_time_s=total_time,
    )


def _print_summary(results: list[ModeResult]) -> None:
    """Print a comparison table and declare winner or GA-is-limiter."""
    print()
    print("=" * 78)
    print("ADRENALINE GAIN MECHANISM COMPARISON")
    print("=" * 78)
    print(
        f"target rates (Hz) per adrenaline segment: "
        f"{[BASELINE_RATE_HZ * a for a in ADRENALINE_PROFILE]}"
    )
    print()
    header_cols = [
        "mode".ljust(22),
        "best fit".rjust(9),
        "err 0".rjust(6),
        "err 1".rjust(6),
        "err 2".rjust(6),
        "err 3".rjust(6),
        "max err".rjust(7),
    ]
    header = " | ".join(header_cols)
    print(header)
    print("-" * len(header))
    for r in results:
        max_err = max(r.seg_error_hz)
        print(
            f"{r.mode:<22} | {r.best_fitness:9.2f} | "
            f"{r.seg_error_hz[0]:6.2f} | {r.seg_error_hz[1]:6.2f} | "
            f"{r.seg_error_hz[2]:6.2f} | {r.seg_error_hz[3]:6.2f} | "
            f"{max_err:7.2f}"
        )
    print()

    passers = [
        r for r in results if max(r.seg_error_hz) <= SUCCESS_HZ_TOLERANCE
    ]
    if passers:
        winner = min(passers, key=lambda r: max(r.seg_error_hz))
        print(
            f"WINNER: {winner.mode} "
            f"(max segment error {max(winner.seg_error_hz):.2f} Hz "
            f"<= {SUCCESS_HZ_TOLERANCE} Hz tolerance)"
        )
    else:
        print(
            f"NO WINNER: no mechanism kept all segments within "
            f"{SUCCESS_HZ_TOLERANCE} Hz of target. Conclusion: the GA "
            f"(direct encoding at N={N_NEURONS}, pop={POP_SIZE}) is the "
            "limiter, not the gain mechanism."
        )
    print()
    print(
        f"median per-mode GA time: "
        f"{statistics.median(r.total_time_s for r in results):.2f}s"
    )


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
    """Run the comparison for every gain_mode and print a summary."""
    _apply_scale_overrides(n_neurons, slots_per_post, n_timesteps)
    scenario = _build_scenario(seed)
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"pop={POP_SIZE}, gens={N_GENERATIONS}, "
        f"N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}, "
        f"modes={len(GAIN_MODES)}"
    )
    results: list[ModeResult] = []
    for mode in GAIN_MODES:
        print(f"--- running gain_mode={mode} ---")
        result = _run_one_mode(mode, scenario, seed)
        print(
            f"    best_fit={result.best_fitness:.2f}  "
            f"max_err={max(result.seg_error_hz):.2f} Hz  "
            f"time={result.total_time_s:.2f}s"
        )
        results.append(result)
    _print_summary(results)


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
