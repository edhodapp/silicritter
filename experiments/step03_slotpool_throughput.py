"""Step 3 throughput measurement: slot-pool synapses + LIF on the GPU.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Runs a slot-pool forward simulation for comparison against the dense
step 2 baseline. Network size is the same (N = 1024) but synapses are
represented as K = 64 slots per postsynaptic neuron with random bindings
-- about 6% of the potential dense connectivity -- so per-step synaptic
input is O(N * K) rather than O(N * N).

Usage:
    .venv/bin/python experiments/step03_slotpool_throughput.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp

from silicritter.lif import DT_MS, LIFState, init_state
from silicritter.slotpool import SlotPool, init_random, simulate


N_REPEATS: int = 5


def _build_inputs(
    n_neurons: int,
    slots_per_post: int,
    n_timesteps: int,
    seed: int,
) -> tuple[LIFState, SlotPool, jax.Array]:
    """Construct the initial LIF state, a random slot pool, and drive."""
    rng = jax.random.PRNGKey(seed)
    pool_key, drive_key = jax.random.split(rng)
    pool = init_random(
        n_post=n_neurons,
        n_pre=n_neurons,
        slots_per_post=slots_per_post,
        rng=pool_key,
        weight_scale=0.05,
    )
    # Match the step 2 drive regime: feedforward equilibrium V_rest + i_ext
    # with V_rest = -65 mV, V_thresh = -50 mV requires i_ext > 15 mV to
    # drive a cell to threshold in isolation. Recurrent slot-pool input
    # shifts this; [17, 25] mV keeps the population in a firing regime.
    i_ext_trace = jax.random.uniform(
        drive_key,
        (n_timesteps, n_neurons),
        minval=17.0,
        maxval=25.0,
        dtype=jnp.float32,
    )
    state = init_state(n_neurons)
    return state, pool, i_ext_trace


def run(
    n_neurons: int = 1024,
    slots_per_post: int = 64,
    n_timesteps: int = 10_000,
    seed: int = 0,
) -> None:
    """Warm up JIT, then run N_REPEATS timed passes; report min/median."""
    state, pool, i_ext_trace = _build_inputs(
        n_neurons, slots_per_post, n_timesteps, seed
    )
    sim_jit = jax.jit(simulate)

    _, warm_spikes = sim_jit(state, pool, i_ext_trace)
    warm_spikes.block_until_ready()

    elapsed_s: list[float] = []
    final_spikes: jax.Array | None = None
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _, final_spikes = sim_jit(state, pool, i_ext_trace)
        final_spikes.block_until_ready()
        elapsed_s.append(time.perf_counter() - t0)

    neuron_steps = n_neurons * n_timesteps
    slot_ops = n_neurons * slots_per_post * n_timesteps
    min_s = min(elapsed_s)
    med_s = statistics.median(elapsed_s)
    assert final_spikes is not None
    mean_rate_hz = float(final_spikes.mean()) * 1000.0 / DT_MS

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
