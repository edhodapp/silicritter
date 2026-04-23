"""Step 5: GA outer loop on target firing rate with adrenaline modulation.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Direct-encoding GA that evolves silicritter slot-pool genomes to track
a time-varying target firing rate driven by an adrenaline signal.
Validates the two-loop structure (outer GA + inner plastic sim) at
small scale; the encoding is known to be scaling-hostile and is not
meant to generalize past this step.

Population-level simulations are vmapped on the GPU; the GA main
loop runs in Python. Each generation evaluates all genomes in one
batched pass.

Usage:
    .venv/bin/python experiments/step05_ga_target_rate.py
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
from silicritter.plasticity import (
    PlasticNetState,
    default_params,
    init_traces,
    simulate_plastic,
)


# --- scale / scenario ---

N_NEURONS: int = 32
K_SLOTS: int = 8
N_TIMESTEPS: int = 2_000

# --- GA hyperparameters ---

POP_SIZE: int = 48
N_GENERATIONS: int = 80
TOURNAMENT_SIZE: int = 3
ELITE_COUNT: int = 2  # top-k survive unchanged each generation
V_SIGMA: float = 0.01
RATE_SIGMA: float = 0.05
PRE_RESAMPLE_PROB: float = 0.03

# --- task: adrenaline profile + firing-rate target ---

BASELINE_RATE_HZ: float = 40.0
WINDOW_STEPS: int = 100  # firing-rate is estimated over 100-step windows
# Piecewise adrenaline over four equal segments; target firing rate
# per segment = BASELINE_RATE_HZ * ADRENALINE_PROFILE[segment].
ADRENALINE_PROFILE: tuple[float, ...] = (1.0, 1.5, 0.8, 1.2)


def _build_scenario(
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the step 5 scenario traces and per-window target rates.

    Returns (i_ext_trace, valence_trace, adrenaline_trace, target_rate):

    - i_ext_trace: mild-ish constant drive with per-neuron jitter, shape
      (T, N_neurons).
    - valence_trace: constant +1, shape (T,).
    - adrenaline_trace: piecewise constant over four segments of T/4
      steps: [1.0, 1.5, 0.8, 1.2]. Shape (T,).
    - target_rate: desired mean firing rate (Hz) for each non-overlapping
      WINDOW_STEPS segment, shape (T / WINDOW_STEPS,).
    """
    drive_key = jax.random.PRNGKey(seed)
    # Drive chosen so baseline adrenaline=1.0 produces firing around the
    # BASELINE_RATE_HZ target. The low-adrenaline segment (0.8) can push
    # some cells below firing threshold at the lower end of the drive
    # range; this is a genuine architectural observation about the
    # multiplicative-gain mechanism, not a bug to smooth over.
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
    # Target rate per window = baseline * adrenaline averaged in window.
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    window_adrenaline = adrenaline_trace.reshape(n_windows, WINDOW_STEPS).mean(
        axis=1
    )
    target_rate_per_window = BASELINE_RATE_HZ * window_adrenaline
    return i_ext_trace, valence_trace, adrenaline_trace, target_rate_per_window


def _fitness_from_spikes(
    spikes: jax.Array, target_rate_per_window: jax.Array
) -> jax.Array:
    """Negative MSE between windowed firing rate and target (Hz).

    spikes: shape (T, N_neurons), boolean.
    target_rate_per_window: shape (n_windows,).
    Returns: scalar fitness (higher = better).
    """
    # Population mean rate at each step (Hz) given dt_ms = 1.
    per_step_rate_hz = spikes.astype(jnp.float32).mean(axis=1) * 1000.0
    n_windows = target_rate_per_window.shape[0]
    window_rate = per_step_rate_hz.reshape(n_windows, -1).mean(axis=1)
    mse = jnp.mean((window_rate - target_rate_per_window) ** 2)
    return -mse


def _evaluate_single(
    genome: Genome,
    i_ext_trace: jax.Array,
    valence_trace: jax.Array,
    adrenaline_trace: jax.Array,
    target_rate_per_window: jax.Array,
) -> jax.Array:
    """Run one critter's lifetime, return its fitness."""
    pool = decode_to_pool(genome)
    state = PlasticNetState(
        lif=init_state(N_NEURONS),
        pool=pool,
        traces=init_traces(n_pre=N_NEURONS, n_post=N_NEURONS),
    )
    params = default_params()
    _, spikes = simulate_plastic(
        state, i_ext_trace, valence_trace, adrenaline_trace, params
    )
    return _fitness_from_spikes(spikes, target_rate_per_window)


def _next_generation(
    population: Genome,
    fitness: jax.Array,
    rng: jax.Array,
    n_pre: int,
) -> Genome:
    """Produce the next-generation population from the current one."""
    k_select, k_pair, k_cross, k_mut = jax.random.split(rng, 4)

    # Elitism: top ELITE_COUNT genomes copy forward unchanged.
    # Guard against NaN fitness (degenerate genomes could produce NaN in
    # simulate_plastic); push NaN to -inf so it never wins elite slots.
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
            g,
            k,
            n_pre,
            V_SIGMA,
            RATE_SIGMA,
            PRE_RESAMPLE_PROB,
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
    """Execute the GA outer loop, printing progress each generation."""
    _apply_scale_overrides(n_neurons, slots_per_post, n_timesteps)
    scenario = _build_scenario(seed)
    i_ext_trace, valence_trace, adrenaline_trace, target_rate = scenario

    # Population-level fitness: vmap the single-critter evaluator.
    evaluate_pop = jax.jit(
        jax.vmap(
            lambda g: _evaluate_single(
                g,
                i_ext_trace,
                valence_trace,
                adrenaline_trace,
                target_rate,
            )
        )
    )

    rng = jax.random.PRNGKey(seed + 100)
    rng, pop_key = jax.random.split(rng)
    population = random_population(
        POP_SIZE, N_NEURONS, N_NEURONS, K_SLOTS, pop_key
    )

    # Warmup JIT; don't count against per-gen timing below.
    _ = evaluate_pop(population).block_until_ready()

    best_fit_per_gen: list[float] = []
    mean_fit_per_gen: list[float] = []
    gen_times: list[float] = []

    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"pop={POP_SIZE}, gens={N_GENERATIONS}, "
        f"N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}"
    )
    print(
        f"target rates per segment (Hz): "
        f"{[float(x) for x in target_rate.reshape(4, -1).mean(axis=1)]}"
    )
    print()

    for gen in range(N_GENERATIONS):
        t0 = time.perf_counter()
        fitness = evaluate_pop(population)
        fitness.block_until_ready()
        gen_times.append(time.perf_counter() - t0)

        best = float(fitness.max())
        mean = float(fitness.mean())
        best_fit_per_gen.append(best)
        mean_fit_per_gen.append(mean)
        print(
            f"gen {gen:3d}: best_fit={best:8.2f}  mean_fit={mean:8.2f}  "
            f"eval={gen_times[-1]*1000:6.1f} ms"
        )

        if gen == N_GENERATIONS - 1:
            break
        rng, gen_key = jax.random.split(rng)
        population = _next_generation(
            population, fitness, gen_key, n_pre=N_NEURONS
        )

    # Final: evaluate the best genome's behavior and report.
    fitness_final = evaluate_pop(population)
    best_idx = int(jnp.argmax(fitness_final))
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
    )
    per_step_rate = best_spikes.astype(jnp.float32).mean(axis=1) * 1000.0
    n_windows = target_rate.shape[0]
    achieved = per_step_rate.reshape(n_windows, -1).mean(axis=1)

    print()
    print("--- best genome behavior by adrenaline segment ---")
    n_segments = len(ADRENALINE_PROFILE)
    seg_targets = target_rate.reshape(n_segments, -1).mean(axis=1)
    seg_achieved = achieved.reshape(n_segments, -1).mean(axis=1)
    for i, (adr, tgt, got) in enumerate(
        zip(ADRENALINE_PROFILE, seg_targets, seg_achieved)
    ):
        err = float(abs(got - tgt))
        print(
            f"segment {i}  adr={adr:.2f}  target={float(tgt):5.1f} Hz  "
            f"achieved={float(got):5.1f} Hz  |err|={err:5.2f} Hz"
        )
    print()
    print(
        f"final best fitness = {float(fitness_final.max()):.2f}  "
        f"(mean = {float(fitness_final.mean()):.2f})"
    )
    print(
        f"total eval time: {sum(gen_times):.1f}s  "
        f"median gen: {statistics.median(gen_times)*1000:.1f}ms"
    )


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
