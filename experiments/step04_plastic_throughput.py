"""Step 4 throughput measurement: slot-pool + three-factor STDP on the GPU.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Runs the same N x K x T configuration as step 3 but with three-factor
STDP plasticity active (constant +1 valence) on top of the slot pool.
Prints throughput plus summary statistics of the learned weight
distribution, to show that plasticity is actually moving the pool.

Usage:
    .venv/bin/python experiments/step04_plastic_throughput.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp

from silicritter.lif import DT_MS, init_state
from silicritter.plasticity import (
    PlasticNetState,
    default_params,
    init_traces,
    simulate_plastic,
)
from silicritter.slotpool import init_random


N_REPEATS: int = 5


def _build_inputs(
    n_neurons: int,
    slots_per_post: int,
    n_timesteps: int,
    seed: int,
) -> tuple[PlasticNetState, jax.Array, jax.Array, jax.Array]:
    """Construct initial state, drive, valence, and adrenaline traces."""
    rng = jax.random.PRNGKey(seed)
    pool_key, drive_key = jax.random.split(rng)
    pool = init_random(
        n_post=n_neurons,
        n_pre=n_neurons,
        slots_per_post=slots_per_post,
        rng=pool_key,
        weight_scale=0.05,
    )
    i_ext_trace = jax.random.uniform(
        drive_key,
        (n_timesteps, n_neurons),
        minval=17.0,
        maxval=25.0,
        dtype=jnp.float32,
    )
    valence_trace = jnp.ones((n_timesteps,), dtype=jnp.float32)
    # Baseline adrenaline preserves step-4 dynamics unchanged.
    adrenaline_trace = jnp.ones((n_timesteps,), dtype=jnp.float32)
    state = PlasticNetState(
        lif=init_state(n_neurons),
        pool=pool,
        traces=init_traces(n_pre=n_neurons, n_post=n_neurons),
    )
    return state, i_ext_trace, valence_trace, adrenaline_trace


def run(
    n_neurons: int = 1024,
    slots_per_post: int = 64,
    n_timesteps: int = 10_000,
    seed: int = 0,
) -> None:
    """Warm up JIT, then run N_REPEATS timed passes; print metrics."""
    state, i_ext_trace, valence_trace, adrenaline_trace = _build_inputs(
        n_neurons, slots_per_post, n_timesteps, seed
    )
    params = default_params()

    def _sim(
        s: PlasticNetState,
        i_ext: jax.Array,
        val: jax.Array,
        adr: jax.Array,
    ) -> tuple[PlasticNetState, jax.Array]:
        return simulate_plastic(s, i_ext, val, adr, params)

    sim_jit = jax.jit(_sim)

    # Warmup also primes the accumulators, avoiding an Optional dance.
    final_state, final_spikes = sim_jit(
        state, i_ext_trace, valence_trace, adrenaline_trace
    )
    final_spikes.block_until_ready()

    elapsed_s: list[float] = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        final_state, final_spikes = sim_jit(
            state, i_ext_trace, valence_trace, adrenaline_trace
        )
        final_spikes.block_until_ready()
        elapsed_s.append(time.perf_counter() - t0)
    neuron_steps = n_neurons * n_timesteps
    slot_ops = n_neurons * slots_per_post * n_timesteps
    min_s = min(elapsed_s)
    med_s = statistics.median(elapsed_s)
    mean_rate_hz = float(final_spikes.mean()) * 1000.0 / DT_MS

    initial_v = state.pool.v
    final_v = final_state.pool.v
    dv = final_v - initial_v

    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(f"N={n_neurons}, K={slots_per_post}, "
          f"T={n_timesteps} steps "
          f"({n_timesteps * DT_MS:.0f} ms simulated)")
    print(f"slot evaluations per step: {n_neurons * slots_per_post}")
    print(f"repeats: {N_REPEATS}")
    print(f"elapsed (min):    {min_s * 1000:.1f} ms")
    print(f"elapsed (median): {med_s * 1000:.1f} ms")
    print(f"throughput (min):    "
          f"{neuron_steps / min_s:.3e} neuron-steps/s")
    print(f"throughput (median): "
          f"{neuron_steps / med_s:.3e} neuron-steps/s")
    print(f"slot-eval throughput (median): "
          f"{slot_ops / med_s:.3e} slot-evals/s")
    print(f"mean firing rate: {mean_rate_hz:.1f} Hz")
    print("--- weight statistics ---")
    print(f"initial v  mean: {float(initial_v.mean()):.4f}, "
          f"min: {float(initial_v.min()):.4f}, "
          f"max: {float(initial_v.max()):.4f}")
    print(f"final   v  mean: {float(final_v.mean()):.4f}, "
          f"min: {float(final_v.min()):.4f}, "
          f"max: {float(final_v.max()):.4f}")
    print(f"|dv| mean: {float(jnp.abs(dv).mean()):.4f}, "
          f"max: {float(jnp.abs(dv).max()):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-neurons", type=int, default=1024)
    parser.add_argument("--slots-per-post", type=int, default=64)
    parser.add_argument("--n-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        n_neurons=args.n_neurons,
        slots_per_post=args.slots_per_post,
        n_timesteps=args.n_timesteps,
        seed=args.seed,
    )
