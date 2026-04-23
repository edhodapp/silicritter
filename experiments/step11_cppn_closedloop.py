"""Step 11: CPPN GA with E/I substrate + closed-loop adrenaline.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 7e evolved B's pool via CPPN and reached step 9's static ceiling
(roughly -1.7e-4) without E/I or closed-loop machinery. Step 10 showed
that closed-loop adrenaline on a *hand-wired* cross-E-only B pool
(step 9's static best) breaks that ceiling to ~-5.6e-5 -- a ~3x
improvement driven entirely by the controller.

Step 11 asks: does the GA find anything better than the hand-wired
configuration when it evolves B *with the controller on*? Two
conditions, both with the E/I substrate from step 9:

  A. open-loop (const adr=1.0): CPPN GA + E/I, no controller
  B. closed-loop (gain=50):     CPPN GA + E/I, controller on

Expected:
  A ~ -1.56e-4  (matches step 9 hand-wired best; CPPN can reach the
                 static ceiling but not break it)
  B lower than step 10's hand-wired -5.6e-5, OR roughly equal to it.

A "roughly equal" B is a legitimate null result: the controller
saturates at ADR_MAX and the GA can't do anything about that with
pool topology alone. A "lower B" would mean the GA discovered a
topology that better matches the controller's dynamics -- e.g.,
mixed cross/recurrent that lets adrenaline operate away from the
rail.

Usage:
    .venv/bin/python experiments/step11_cppn_closedloop.py \\
        --gens=30 --pop=32 --seed=0
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from silicritter.closedloop import ControllerParams, simulate_closedloop
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
from silicritter.slotpool import SlotPool, assign_ei_identity


N_NEURONS: int = 256
K_SLOTS: int = 32
N_TIMESTEPS: int = 2_000
WINDOW_STEPS: int = 100
V_MAX: float = 2.0
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)
B_BASELINE_DRIVE_MV: float = 16.0
INHIBITORY_FRACTION: float = 0.2
I_WEIGHT_MULTIPLIER: float = 4.0

# Controller: step 10's defaults at gain=50 (the saturation regime).
CONTROLLER_DECAY: float = 0.98
BASELINE_ADRENALINE: float = 1.0
CLOSED_LOOP_GAIN: float = 50.0
ADR_MIN: float = 0.5
ADR_MAX: float = 3.0

# CPPN + GA hyperparameters (match step 7e).
CPPN_HIDDEN_DIM: int = 8
CPPN_INIT_SCALE: float = 1.0
CPPN_MUTATE_SIGMA: float = 0.15
TOURNAMENT_SIZE: int = 3
ELITE_COUNT: int = 2


Scenario = tuple[
    SlotPool,
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
]


def _stdp_params() -> STDPParams:
    return default_params()._replace(v_max=V_MAX)


def _ctrl_params(gain: float) -> ControllerParams:
    return ControllerParams(
        decay=CONTROLLER_DECAY, baseline=BASELINE_ADRENALINE, gain=gain,
        adr_min=ADR_MIN, adr_max=ADR_MAX,
    )


def _build_scenario() -> Scenario:
    """Pool_a, drive traces, valences, adrenaline_a. Identical across seeds."""
    seg_len = N_TIMESTEPS // len(A_DRIVE_PROFILE)
    i_ext_a = jnp.concatenate(
        [
            jnp.full((seg_len, N_NEURONS), level, dtype=jnp.float32)
            for level in A_DRIVE_PROFILE
        ]
    )
    i_ext_b = jnp.full(
        (N_TIMESTEPS, N_NEURONS), B_BASELINE_DRIVE_MV, dtype=jnp.float32,
    )
    valence = jnp.zeros((N_TIMESTEPS,), dtype=jnp.float32)
    adrenaline_a = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777)
    )
    return (pool_a, i_ext_a, i_ext_b, valence, valence, adrenaline_a)


def _build_state(pool_a: SlotPool, pool_b: SlotPool) -> PairedState:
    return PairedState(
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


def _prediction_fitness(
    spikes_a: jax.Array, spikes_b: jax.Array,
) -> jax.Array:
    """Negative MSE: B's window-t rate vs A's window-(t+1) rate."""
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    a_rate = spikes_a.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    b_rate = spikes_b.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    return -jnp.mean((b_rate[:-1] - a_rate[1:]) ** 2)


def _evaluate_one(
    genome_b: CPPNGenome,
    scenario: Scenario,
    a_is_inh: jax.Array,
    b_is_inh: jax.Array,
    closed_loop: bool,
) -> jax.Array:
    """Decode B's pool from genome; run one sim; return fitness."""
    pool_a, i_ext_a, i_ext_b, val_a, val_b, adr_a = scenario
    pool_b = decode_cppn_to_pool(
        genome_b, N_NEURONS, 2 * N_NEURONS, K_SLOTS, V_MAX
    )
    state = _build_state(pool_a, pool_b)
    if closed_loop:
        spikes_a, spikes_b, _ = simulate_closedloop(
            state, _ctrl_params(CLOSED_LOOP_GAIN),
            i_ext_a, i_ext_b, val_a, val_b, adr_a,
            _stdp_params(),
            gain_mode="tau_m_scale",
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=b_is_inh,
            i_weight_multiplier=I_WEIGHT_MULTIPLIER,
        )
    else:
        adr_b = jnp.full_like(adr_a, BASELINE_ADRENALINE)
        _, spikes_a, spikes_b = simulate_paired(
            state, i_ext_a, i_ext_b, val_a, val_b, adr_a, adr_b,
            _stdp_params(),
            gain_mode="tau_m_scale",
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=b_is_inh,
            i_weight_multiplier=I_WEIGHT_MULTIPLIER,
        )
    return _prediction_fitness(spikes_a, spikes_b)


def _next_generation(
    population: CPPNGenome,
    fitness: jax.Array,
    rng: jax.Array,
    pop_size: int,
) -> CPPNGenome:
    k_sel, k_pair, k_cross, k_mut = jax.random.split(rng, 4)
    safe_fitness = jnp.where(
        jnp.isnan(fitness), jnp.float32(-jnp.inf), fitness
    )
    elite_idx = jnp.argsort(-safe_fitness)[:ELITE_COUNT]
    elites = CPPNGenome(
        w1=population.w1[elite_idx],
        w2=population.w2[elite_idx],
    )
    n_offspring = pop_size - ELITE_COUNT
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


def _evolve(
    label: str,
    scenario: Scenario,
    a_is_inh: jax.Array,
    b_is_inh: jax.Array,
    closed_loop: bool,
    pop_size: int,
    n_generations: int,
    seed: int,
) -> tuple[CPPNGenome, float]:
    """Run a CPPN GA and return (best_genome, best_fitness)."""
    evaluate_pop = jax.jit(jax.vmap(
        lambda g: _evaluate_one(g, scenario, a_is_inh, b_is_inh, closed_loop)
    ))
    rng = jax.random.PRNGKey(seed + 200)
    rng, pop_key = jax.random.split(rng)
    population = random_cppn_population(
        pop_size, CPPN_HIDDEN_DIM, pop_key, CPPN_INIT_SCALE
    )
    _ = evaluate_pop(population).block_until_ready()
    gen_times: list[float] = []
    best = float("-inf")
    for gen in range(n_generations):
        t0 = time.perf_counter()
        fitness = evaluate_pop(population)
        fitness.block_until_ready()
        gen_times.append(time.perf_counter() - t0)
        best = float(fitness.max())
        mean = float(fitness.mean())
        print(
            f"[{label}] gen {gen:3d}: best_fit={best:10.3e}  "
            f"mean_fit={mean:10.3e}  eval={gen_times[-1]*1000:7.1f} ms"
        )
        if gen == n_generations - 1:
            break
        rng, gen_key = jax.random.split(rng)
        population = _next_generation(population, fitness, gen_key, pop_size)
    final_fitness = evaluate_pop(population)
    best_idx = int(jnp.argmax(final_fitness))
    best_genome = CPPNGenome(
        w1=population.w1[best_idx], w2=population.w2[best_idx],
    )
    print(
        f"[{label}] total eval time: {sum(gen_times):.1f}s  "
        f"median gen: {statistics.median(gen_times)*1000:.1f}ms"
    )
    return best_genome, float(final_fitness.max())


def _describe_pool(pool_b: SlotPool) -> str:
    n_cross = int(jnp.sum(pool_b.pre_ids >= N_NEURONS))
    total = N_NEURONS * K_SLOTS
    cross_pct = 100.0 * n_cross / total
    mean_v = float(jnp.mean(pool_b.v))
    max_v = float(jnp.max(pool_b.v))
    return (
        f"{n_cross}/{total} cross-bound ({cross_pct:.1f}%), "
        f"v mean {mean_v:.3f}, max {max_v:.3f}"
    )


def run(
    seed: int = 0, pop_size: int = 32, n_generations: int = 30,
) -> None:
    scenario = _build_scenario()
    a_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)

    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}, "
        f"E:I = {1-INHIBITORY_FRACTION:.0%}:{INHIBITORY_FRACTION:.0%}, "
        f"i_mult={I_WEIGHT_MULTIPLIER}, gain_mode=tau_m_scale"
    )
    print(f"A drive profile (mV): {list(A_DRIVE_PROFILE)}")
    print(
        f"GA: pop={pop_size}, gens={n_generations}, "
        f"CPPN hidden={CPPN_HIDDEN_DIM}, mutate_sigma={CPPN_MUTATE_SIGMA}"
    )
    print()

    open_best, open_fit = _evolve(
        "open-loop", scenario, a_is_inh, b_is_inh,
        closed_loop=False,
        pop_size=pop_size, n_generations=n_generations, seed=seed,
    )
    print()
    closed_best, closed_fit = _evolve(
        "closed-loop", scenario, a_is_inh, b_is_inh,
        closed_loop=True,
        pop_size=pop_size, n_generations=n_generations, seed=seed,
    )
    print()
    open_pool = decode_cppn_to_pool(
        open_best, N_NEURONS, 2 * N_NEURONS, K_SLOTS, V_MAX
    )
    closed_pool = decode_cppn_to_pool(
        closed_best, N_NEURONS, 2 * N_NEURONS, K_SLOTS, V_MAX
    )
    print(
        f"open-loop   best: {open_fit:10.3e}   "
        f"pool: {_describe_pool(open_pool)}"
    )
    print(
        f"closed-loop best: {closed_fit:10.3e}   "
        f"pool: {_describe_pool(closed_pool)}"
    )
    print()
    print("baselines:")
    print("  step  9 hand-wired cross-E-only, const adr:  -1.56e-4")
    print("  step 10 hand-wired cross-E-only, closed-loop (gain=50): -5.60e-5")
    print("  step  7e CPPN GA (no E/I, const adr):        -1.70e-4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--gens", type=int, default=30)
    args = parser.parse_args()
    run(seed=args.seed, pop_size=args.pop, n_generations=args.gens)
