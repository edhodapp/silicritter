"""Step 7e: paired-agent signal-following with CPPN indirect encoding.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Phase 4.1 established that direct-encoding GA cannot reach the N=256
hand-wired optimum (-1.74e-4) regardless of plasticity, because the
search space (~24,000 direct parameters) is too large for a 30-gen
GA to move into structured configurations like "every B slot bound
to A with v near v_max."

This experiment swaps the direct encoding for a CPPN indirect encoder
(src/silicritter/cppn.py). A CPPN with hidden_dim=8 has 51 evolved
weights. A single weight configuration can express "all slots
cross-bound with high v" as a small set of output biases. The search
space collapses by roughly two orders of magnitude; structured
configurations should be reachable inside a 30-gen GA.

Same scenario as step 7 / Phase 4 so comparison is fair: N=256, K=32,
tonic=16 mV, v_max=2.0, T=2000, pop=32, gens=30, plasticity ON.

Hand-wired baseline (Phase 3, plasticity off, all-cross v=2.0):
  fitness = -1.74e-4

Direct-encoding GA at this scale (Phase 4, scaled exploration):
  fitness = -4.17e-4

Target: CPPN GA should reach at or below the hand-wired baseline.

Usage:
    .venv/bin/python experiments/step07e_paired_cppn_n256.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp

from silicritter.cppn import (
    CPPNGenome,
    decode_cppn_to_pool,
    mutate_cppn,
    random_cppn_population,
    tournament_select_cppn,
    uniform_crossover_cppn,
)
from silicritter.lif import init_state
from silicritter.paired import (
    PairedState,
    make_pool_for_partner,
    simulate_paired,
)
from silicritter.plasticity import (
    PlasticNetState,
    STDPParams,
    default_params,
    init_traces,
)
from silicritter.slotpool import SlotPool


# Scale -- match Phase 4 exactly.
N_NEURONS: int = 256
K_SLOTS: int = 32
N_TIMESTEPS: int = 2_000
WINDOW_STEPS: int = 100
V_MAX: float = 2.0

# A's drive + B tonic -- match Phase 4.
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)
B_BASELINE_DRIVE_MV: float = 16.0

# GA hyperparameters
POP_SIZE: int = 32
N_GENERATIONS: int = 30
TOURNAMENT_SIZE: int = 3
ELITE_COUNT: int = 2

# CPPN hyperparameters
CPPN_HIDDEN_DIM: int = 8
CPPN_INIT_SCALE: float = 1.0
CPPN_MUTATE_SIGMA: float = 0.15


Scenario = tuple[
    SlotPool,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


def _stdp_params() -> STDPParams:
    """STDP params with v_max raised to V_MAX."""
    return default_params()._replace(v_max=V_MAX)


def _build_scenario() -> Scenario:
    """Build fixed A scaffold + drive traces. Same shape as step 7 / 7e."""
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
    pool_a = make_pool_for_partner(N_NEURONS, K_SLOTS, jax.random.PRNGKey(777))
    return (
        pool_a,
        i_ext_a_trace, i_ext_b_trace,
        valence_a_trace, valence_b_trace,
        adrenaline_trace, adrenaline_trace,
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
    genome_b: CPPNGenome,
    scenario: Scenario,
) -> jax.Array:
    """Run one paired critter (CPPN-decoded B); return fitness."""
    pool_a, i_ext_a, i_ext_b, val_a, val_b, adr_a, adr_b = scenario
    pool_b = decode_cppn_to_pool(
        genome_b, N_NEURONS, 2 * N_NEURONS, K_SLOTS, V_MAX
    )
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
        _stdp_params(),
    )
    return _prediction_fitness(spikes_a, spikes_b)


def _next_generation(
    population: CPPNGenome,
    fitness: jax.Array,
    rng: jax.Array,
) -> CPPNGenome:
    """Elite + tournament + crossover + mutate, all on CPPN genomes."""
    k_sel, k_pair, k_cross, k_mut = jax.random.split(rng, 4)
    safe_fitness = jnp.where(
        jnp.isnan(fitness), jnp.float32(-jnp.inf), fitness
    )
    elite_idx = jnp.argsort(-safe_fitness)[:ELITE_COUNT]
    elites = CPPNGenome(
        w1=population.w1[elite_idx],
        w2=population.w2[elite_idx],
    )
    n_offspring = POP_SIZE - ELITE_COUNT
    parents_a = tournament_select_cppn(
        population, fitness, n_offspring, TOURNAMENT_SIZE, k_sel
    )
    parents_b = tournament_select_cppn(
        population, fitness, n_offspring, TOURNAMENT_SIZE, k_pair
    )
    cross_keys = jax.random.split(k_cross, n_offspring)
    children = jax.vmap(uniform_crossover_cppn)(
        parents_a, parents_b, cross_keys
    )
    mut_keys = jax.random.split(k_mut, n_offspring)
    mutated = jax.vmap(
        lambda g, k: mutate_cppn(g, k, CPPN_MUTATE_SIGMA)
    )(children, mut_keys)
    return CPPNGenome(
        w1=jnp.concatenate([elites.w1, mutated.w1], axis=0),
        w2=jnp.concatenate([elites.w2, mutated.w2], axis=0),
    )


def _report_best(
    best_genome: CPPNGenome,
    scenario: Scenario,
) -> None:
    """Re-run best genome and print window-by-window A vs B rates."""
    pool_a, i_ext_a, i_ext_b, val_a, val_b, adr_a, adr_b = scenario
    pool_b = decode_cppn_to_pool(
        best_genome, N_NEURONS, 2 * N_NEURONS, K_SLOTS, V_MAX
    )
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
        _stdp_params(),
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
    # Also inspect the CPPN-decoded pool to see what the best genome
    # actually wired.
    n_cross = int(jnp.sum(pool_b.pre_ids >= N_NEURONS))
    total_slots = N_NEURONS * K_SLOTS
    mean_v = float(jnp.mean(pool_b.v))
    max_v = float(jnp.max(pool_b.v))
    cross_pct = 100.0 * n_cross / total_slots
    print(
        f"best genome decoded pool: "
        f"{n_cross}/{total_slots} cross-bound ({cross_pct:.1f}%), "
        f"v mean {mean_v:.3f}, v max {max_v:.3f}"
    )
    print()
    print("--- best genome window-by-window ---")
    print(
        " window   A rate (Hz)   B rate (Hz)   |B(t) - A(t+1)|"
    )
    for w in range(n_windows):
        a_next = float(a_rate_hz[w + 1]) if w + 1 < n_windows else float("nan")
        err = (
            abs(float(b_rate_hz[w]) - a_next)
            if w + 1 < n_windows else float("nan")
        )
        err_str = "  n/a" if w + 1 >= n_windows else f"{err:8.2f}"
        print(
            f"  {w:3d}     {float(a_rate_hz[w]):8.1f}      "
            f"{float(b_rate_hz[w]):8.1f}       {err_str}"
        )


def run(seed: int = 0) -> None:
    """Execute CPPN-encoded paired-agent GA and report results."""
    scenario = _build_scenario()

    evaluate_pop = jax.jit(
        jax.vmap(lambda g: _evaluate_single(g, scenario))
    )

    rng = jax.random.PRNGKey(seed + 200)
    rng, pop_key = jax.random.split(rng)
    population = random_cppn_population(
        POP_SIZE, CPPN_HIDDEN_DIM, pop_key, CPPN_INIT_SCALE
    )
    _ = evaluate_pop(population).block_until_ready()

    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"pop={POP_SIZE}, gens={N_GENERATIONS}, "
        f"N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}"
    )
    print(
        f"CPPN hidden_dim={CPPN_HIDDEN_DIM}, "
        f"init_scale={CPPN_INIT_SCALE}, "
        f"mutate_sigma={CPPN_MUTATE_SIGMA}"
    )
    print(f"v_max={V_MAX}, tonic={B_BASELINE_DRIVE_MV} mV")
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
            f"gen {gen:3d}: best_fit={best:10.3e}  "
            f"mean_fit={mean:10.3e}  "
            f"eval={gen_times[-1]*1000:7.1f} ms"
        )
        if gen == N_GENERATIONS - 1:
            break
        rng, gen_key = jax.random.split(rng)
        population = _next_generation(population, fitness, gen_key)

    final_fitness = evaluate_pop(population)
    best_idx = int(jnp.argmax(final_fitness))
    best_genome = CPPNGenome(
        w1=population.w1[best_idx],
        w2=population.w2[best_idx],
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
    print()
    print(
        "baselines:\n"
        "  step 7 (N=32, direct):               -5.18e-4\n"
        "  Phase 4 (N=256, direct, scaled):     -4.17e-4\n"
        "  Phase 3 hand-wired (plasticity off): -1.74e-4\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)
