"""Step 2 throughput measurement: LIF forward-sim on the GPU.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Runs a single-population LIF simulation of N neurons for T timesteps and
reports wall-clock throughput in neuron-steps / second. This is a
baseline sanity check that the JAX + CUDA path is exercised end-to-end.

Usage:
    .venv/bin/python experiments/step02_throughput.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from silicritter.lif import DT_MS, LIFState, init_state, simulate


N_REPEATS: int = 5


def _build_inputs(
    n_neurons: int, n_timesteps: int, seed: int
) -> tuple[LIFState, jax.Array, jax.Array]:
    """Construct the initial state, random weights, and constant drive."""
    rng = jax.random.PRNGKey(seed)
    w_key, drive_key = jax.random.split(rng)
    # Sparse-ish random weights centered at zero, small magnitude.
    weights = (
        jax.random.normal(w_key, (n_neurons, n_neurons), dtype=jnp.float32)
        * 0.05
    )
    # Feedforward-only LIF equilibrium is V_rest + i_ext; with V_rest = -65
    # and V_thresh = -50 a cell needs i_ext > 15 mV to reach threshold in
    # steady state. Recurrent synaptic input shifts this, but [17, 25] mV
    # keeps the population firmly in the firing regime.
    i_ext_trace = jax.random.uniform(
        drive_key,
        (n_timesteps, n_neurons),
        minval=17.0,
        maxval=25.0,
        dtype=jnp.float32,
    )
    state = init_state(n_neurons)
    return state, weights, i_ext_trace


def run(
    n_neurons: int = 1024,
    n_timesteps: int = 10_000,
    seed: int = 0,
) -> None:
    """Warm up JIT, then run N_REPEATS timed passes; report min/median."""
    state, weights, i_ext_trace = _build_inputs(n_neurons, n_timesteps, seed)
    sim_jit = jax.jit(simulate)

    # Warmup triggers JIT compilation; shape/dtype match the timed inputs
    # so the compiled kernel is reused across repeats.
    _, warm_spikes = sim_jit(state, weights, i_ext_trace)
    warm_spikes.block_until_ready()

    elapsed_s: list[float] = []
    final_spikes: jax.Array | None = None
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _, final_spikes = sim_jit(state, weights, i_ext_trace)
        final_spikes.block_until_ready()
        elapsed_s.append(time.perf_counter() - t0)

    neuron_steps = n_neurons * n_timesteps
    min_s = min(elapsed_s)
    med_s = sorted(elapsed_s)[N_REPEATS // 2]
    # final_spikes is guaranteed non-None after the loop (N_REPEATS > 0).
    assert final_spikes is not None
    mean_rate_hz = float(final_spikes.mean()) * 1000.0 / DT_MS

    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(f"N={n_neurons}, T={n_timesteps} steps "
          f"({n_timesteps * DT_MS:.0f} ms simulated)")
    print(f"repeats: {N_REPEATS}")
    print(f"elapsed (min):    {min_s * 1000:.1f} ms")
    print(f"elapsed (median): {med_s * 1000:.1f} ms")
    print(f"throughput (min):    {neuron_steps / min_s:.3e} neuron-steps/s")
    print(f"throughput (median): {neuron_steps / med_s:.3e} neuron-steps/s")
    print(f"mean firing rate: {mean_rate_hz:.1f} Hz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-neurons", type=int, default=1024)
    parser.add_argument("--n-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        n_neurons=args.n_neurons,
        n_timesteps=args.n_timesteps,
        seed=args.seed,
    )
