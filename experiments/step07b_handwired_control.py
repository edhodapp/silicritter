"""Step 7.5: hand-wired predictor control for the paired signal-following task.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

The step-7 GA plateaued at fitness ~ -5.18e-4, and it was not clear
whether the plateau reflects a GA limit (search found no better
configuration) or an architecture / task limit (no B pool exists
that predicts A well at this scale). This experiment answers the
question by *hand-wiring* several B pool configurations, running
each through the identical step-7 scenario, and comparing fitness.

If any hand-wired B beats the GA plateau by a meaningful margin, the
task is solvable at this architecture and the GA encoding is the
limiter. If no hand-wired B beats the plateau, the task itself is
either unsolvable at this scale or the GA is already near the
optimum of a flat fitness landscape.

All hand-wired pools freeze plasticity (plasticity_rate = 0) so the
static configuration persists through the sim without being reshaped
by STDP. That makes the control purely about whether the initial
configuration can solve the task, independent of lifetime learning.

Usage:
    .venv/bin/python experiments/step07b_handwired_control.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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


# Must match step 7 exactly for the comparison to be valid.
N_NEURONS: int = 32
K_SLOTS: int = 8
N_TIMESTEPS: int = 2_000
WINDOW_STEPS: int = 100
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)
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
    """Same scenario as step 7 -- same fixed A pool, drive, profiles."""
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
    # Valence for B is 0.0 here because plasticity_rate=0 freezes weights
    # regardless; keeping both zero makes the intent explicit.
    valence_b_trace = jnp.zeros((N_TIMESTEPS,), dtype=jnp.float32)
    adrenaline_trace = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    pool_a = make_pool_for_partner(N_NEURONS, K_SLOTS, jax.random.PRNGKey(777))
    return (
        pool_a,
        i_ext_a_trace, i_ext_b_trace,
        valence_a_trace, valence_b_trace,
        adrenaline_trace, adrenaline_trace,
    )


def _frozen_pool(
    pre_ids: jax.Array, v_matrix: jax.Array
) -> SlotPool:
    """Pool with given pre_ids / v; plasticity_rate=0 so STDP is inert."""
    return SlotPool(
        pre_ids=pre_ids,
        v=v_matrix,
        plasticity_rate=jnp.zeros_like(v_matrix),
        active=jnp.ones_like(v_matrix, dtype=jnp.bool_),
        release_counter=jnp.zeros_like(pre_ids, dtype=jnp.int32),
    )


def _make_silent_pool() -> SlotPool:
    """Every slot bound somewhere but v=0 everywhere -> no synaptic input."""
    pre_ids = jnp.zeros((N_NEURONS, K_SLOTS), dtype=jnp.int32)
    v = jnp.zeros((N_NEURONS, K_SLOTS), dtype=jnp.float32)
    return _frozen_pool(pre_ids, v)


def _make_cross_pool(cross_v: float, seed: int) -> SlotPool:
    """Every slot bound to a random A-side pre (index in [N, 2N))."""
    rng = jax.random.PRNGKey(seed)
    pre_ids = jax.random.randint(
        rng,
        (N_NEURONS, K_SLOTS),
        minval=N_NEURONS,
        maxval=2 * N_NEURONS,
        dtype=jnp.int32,
    )
    v = jnp.full((N_NEURONS, K_SLOTS), cross_v, dtype=jnp.float32)
    return _frozen_pool(pre_ids, v)


def _make_recurrent_pool(self_v: float, seed: int) -> SlotPool:
    """Every slot bound to a random own-side pre (index in [0, N))."""
    rng = jax.random.PRNGKey(seed)
    pre_ids = jax.random.randint(
        rng,
        (N_NEURONS, K_SLOTS),
        minval=0,
        maxval=N_NEURONS,
        dtype=jnp.int32,
    )
    v = jnp.full((N_NEURONS, K_SLOTS), self_v, dtype=jnp.float32)
    return _frozen_pool(pre_ids, v)


def _prediction_fitness(
    spikes_a: jax.Array, spikes_b: jax.Array
) -> float:
    """Negative MSE between B at window t and A at window t+1."""
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    a_rate = spikes_a.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    b_rate = spikes_b.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    return float(-jnp.mean((b_rate[:-1] - a_rate[1:]) ** 2))


def _evaluate(
    pool_b: SlotPool,
    scenario: Scenario,
) -> tuple[float, jax.Array, jax.Array]:
    """Run paired sim with the given hand-wired B; return fitness + rates."""
    pool_a, i_ext_a, i_ext_b, val_a, val_b, adr_a, adr_b = scenario
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
    fitness = _prediction_fitness(spikes_a, spikes_b)
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
    return fitness, a_rate_hz, b_rate_hz


def run(seed: int = 0) -> None:
    """Run each hand-wired control, print fitnesses, compare to GA plateau."""
    scenario = _build_scenario()
    ga_plateau = -5.18e-4  # from step 7 perf_history

    controls: list[tuple[str, SlotPool]] = [
        ("silent (v = 0)", _make_silent_pool()),
        (
            "all-recurrent v = 0.05",
            _make_recurrent_pool(0.05, seed + 1),
        ),
        (
            "all-recurrent v = 0.10",
            _make_recurrent_pool(0.10, seed + 2),
        ),
        (
            "all-cross v = 0.05",
            _make_cross_pool(0.05, seed + 3),
        ),
        (
            "all-cross v = 0.10",
            _make_cross_pool(0.10, seed + 4),
        ),
        (
            "all-cross v = 0.20",
            _make_cross_pool(0.20, seed + 5),
        ),
        (
            "all-cross v = 0.30",
            _make_cross_pool(0.30, seed + 6),
        ),
    ]

    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}, "
        f"A profile = {list(A_DRIVE_PROFILE)}, "
        f"B tonic = {B_BASELINE_DRIVE_MV} mV"
    )
    print(f"GA plateau (step 7): fitness = {ga_plateau:.3e}")
    print()

    header = (
        "config".ljust(26)
        + " | " + "fitness".rjust(12)
        + " | " + "vs. GA".rjust(10)
        + " | " + "B rates (Hz) by segment".ljust(30)
    )
    print(header)
    print("-" * len(header))

    results: list[tuple[str, float, tuple[float, ...]]] = []
    for name, pool_b in controls:
        fitness, a_rate_hz, b_rate_hz = _evaluate(pool_b, scenario)
        # Summarize B's rate per segment (4 segments, 5 windows each).
        seg_rates = tuple(
            float(b_rate_hz.reshape(4, -1).mean(axis=1)[i])
            for i in range(4)
        )
        vs_ga = fitness / ga_plateau  # >1 means better than GA
        vs_tag = (
            f"{vs_ga:.2f}x" if fitness > ga_plateau else f"{vs_ga:.2f}x (worse)"
        )
        seg_str = "[" + ", ".join(f"{r:5.1f}" for r in seg_rates) + "]"
        print(
            f"{name:<26} | {fitness:12.3e} | "
            f"{vs_tag:>10} | {seg_str:<30}"
        )
        results.append((name, fitness, seg_rates))

    # For reference, print A's per-segment rates too.
    _, a_rate_hz, _ = _evaluate(controls[0][1], scenario)
    a_seg = tuple(
        float(a_rate_hz.reshape(4, -1).mean(axis=1)[i]) for i in range(4)
    )
    print()
    print(f"A's per-segment firing rates: [{a_seg[0]:.1f}, "
          f"{a_seg[1]:.1f}, {a_seg[2]:.1f}, {a_seg[3]:.1f}] Hz")

    best = max(results, key=lambda r: r[1])
    print()
    print(f"best hand-wired: '{best[0]}' at fitness = {best[1]:.3e}")
    # ga_plateau is negative; a 2x-better fitness is half the magnitude,
    # i.e., best > ga_plateau / 2 (less negative). E.g. ga_plateau =
    # -5.18e-4, half-magnitude = -2.59e-4.
    if best[1] > ga_plateau / 2.0:
        print(
            "VERDICT: hand-wired configurations beat the GA plateau by "
            "more than 2x. The GA is the limiter, not the task or "
            "architecture; indirect encoding or a different optimizer is "
            "likely to help."
        )
    else:
        print(
            "VERDICT: hand-wired controls do not meaningfully beat the GA "
            "plateau. The task at this architecture is near its limit; "
            "the GA encoding is not the main bottleneck. Improving the "
            "result needs architectural changes (scale, cross-weight "
            "budget, tonic drive), not a better optimizer."
        )


if __name__ == "__main__":
    run()
