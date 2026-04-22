"""Step 6: structural plasticity (slot release) dynamics.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Runs a plastic slot-pool network with structural release enabled and
tracks how the pool contracts over time. The sim is chunked into
measurement windows so the active-slot count can be sampled as a
trajectory without extending simulate_plastic's interface.

The experiment compares two valence regimes and reports the release
trajectory under each:

1. **valence = +1** (STDP in the nominal direction: LTP on
   pre-then-post, LTD on post-then-pre).
2. **valence = -1** (STDP rules sign-flipped: LTD on pre-then-post,
   LTP on post-then-pre).

Both use adrenaline = 1.0 and gain_mode = "multiplicative". Default
STDP params have a_minus (0.012) > a_plus (0.010), so the rule is
asymmetric: uncorrelated activity drifts weights DOWN under valence
= +1 and UP under valence = -1. The labels "LTP-biased" and
"LTD-biased" would be misleading at this asymmetry -- we therefore
report the valence sign directly and discuss the resulting net drift
in perf_history.md.

Usage:
    .venv/bin/python experiments/step06_structural_release.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.lif import init_state
from silicritter.plasticity import (
    PlasticNetState,
    default_params,
    init_traces,
    simulate_plastic,
)
from silicritter.slotpool import init_random
from silicritter.structural import StructuralParams


N_NEURONS: int = 64
K_SLOTS: int = 16
N_CHUNKS: int = 20
CHUNK_STEPS: int = 250
TOTAL_STEPS: int = N_CHUNKS * CHUNK_STEPS  # 5 000


def _build_initial_state(seed: int) -> PlasticNetState:
    """Return a fresh PlasticNetState with a dense active pool."""
    pool = init_random(
        n_post=N_NEURONS,
        n_pre=N_NEURONS,
        slots_per_post=K_SLOTS,
        rng=jax.random.PRNGKey(seed),
        weight_scale=0.05,
    )
    return PlasticNetState(
        lif=init_state(N_NEURONS),
        pool=pool,
        traces=init_traces(n_pre=N_NEURONS, n_post=N_NEURONS),
    )


def _run_trajectory(
    initial_state: PlasticNetState,
    valence_sign: float,
    seed: int,
) -> list[int]:
    """Simulate in N_CHUNKS windows; return active-slot count per window."""
    rng = jax.random.PRNGKey(seed + 1000)
    state = initial_state
    active_over_time: list[int] = [int(state.pool.active.sum())]
    stdp_params = default_params()
    structural_params = StructuralParams(
        v_release_threshold=0.01,
        release_dwell_steps=100,
    )

    for _ in range(N_CHUNKS):
        rng, k = jax.random.split(rng)
        i_ext_trace = jax.random.uniform(
            k,
            (CHUNK_STEPS, N_NEURONS),
            minval=17.0,
            maxval=22.0,
            dtype=jnp.float32,
        )
        valence_trace = jnp.full(
            (CHUNK_STEPS,), valence_sign, dtype=jnp.float32
        )
        adrenaline_trace = jnp.ones((CHUNK_STEPS,), dtype=jnp.float32)
        state, _ = simulate_plastic(
            state,
            i_ext_trace,
            valence_trace,
            adrenaline_trace,
            stdp_params,
            gain_mode="multiplicative",
            structural_params=structural_params,
        )
        active_over_time.append(int(state.pool.active.sum()))
    return active_over_time


def _format_trajectory(label: str, counts: list[int]) -> str:
    """Pretty-print one condition's active-slot trajectory."""
    header = (
        f"{label:<20} initial={counts[0]:4d}  "
        f"final={counts[-1]:4d}  "
        f"released={counts[0] - counts[-1]:4d}"
    )
    # Compact bar chart across chunks.
    max_count = max(counts) if max(counts) > 0 else 1
    bars = " ".join(
        str(int(round(10 * c / max_count))) for c in counts
    )
    prefix = " " * 20
    return header + "\n" + prefix + " per-chunk active/10: " + bars


def run(seed: int = 0) -> None:
    """Run LTP- and LTD-biased trajectories; print comparison."""
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"N={N_NEURONS}, K={K_SLOTS}, "
        f"chunks={N_CHUNKS} x {CHUNK_STEPS} steps "
        f"(total T={TOTAL_STEPS})"
    )
    print(
        "structural: v_release_threshold=0.01, "
        "release_dwell_steps=100"
    )
    print()

    initial_state = _build_initial_state(seed)

    print("--- running valence = +1 ---")
    pos_counts = _run_trajectory(initial_state, +1.0, seed)
    print(_format_trajectory("valence = +1", pos_counts))
    print()

    print("--- running valence = -1 ---")
    neg_counts = _run_trajectory(initial_state, -1.0, seed)
    print(_format_trajectory("valence = -1", neg_counts))
    print()

    total = N_NEURONS * K_SLOTS
    print(f"pool capacity: {total}")
    print(
        f"valence = +1 retention: "
        f"{100.0 * pos_counts[-1] / total:5.1f}%  "
        f"({pos_counts[-1]} / {total} slots)"
    )
    print(
        f"valence = -1 retention: "
        f"{100.0 * neg_counts[-1] / total:5.1f}%  "
        f"({neg_counts[-1]} / {total} slots)"
    )


if __name__ == "__main__":
    run()
