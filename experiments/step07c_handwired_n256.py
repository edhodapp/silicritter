"""Step 7c: hand-wired control at N=256 with adjusted architectural knobs.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Phase 3 of the N=256 re-scale plan. Step 7.5 closed with the verdict
that the step-7 signal-following task was architecture-limited at
N=32: no hand-wired B configuration could move B's firing rate away
from ~20 Hz regardless of cross-weight strength, because the 16 mV
tonic drive drowned the cross-input. This experiment tests whether
the architectural parameters proposed at the close of step 7.5 --
raise N, raise K, drop tonic to 0, raise v_max -- give the cross
path the leverage it needs.

Adjusted knobs vs. step 7b:

- N: 32 -> 256
- K: 8 -> 32
- B tonic drive: 16 mV -> 0 mV
- v_max: 0.5 -> 2.0

A's drive profile (18, 22, 19, 24 mV) and the 4-segment structure are
held constant so fitness comparisons are meaningful against prior
runs. A's scaffold is a fixed random pool at N=256, K=32 shared
across all configurations.

Success criteria (two, both matter):

1. *Within a single configuration*, does B's per-segment firing rate
   track A's per-segment variation? This is the prediction
   capability we'd need for the full GA task.
2. *Across configurations*, does B's rate shape change with
   cross-weight strength? This is the architectural leverage --
   whether the pool configuration actually matters.

If both hold, architecture has room; Phase 4 GA is worth running.
If neither holds, the architectural parameter change did not fix
things; we pivot to a different scale, task, or mechanism.

Usage:
    .venv/bin/python experiments/step07c_handwired_n256.py
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
    STDPParams,
    default_params,
    init_traces,
)
from silicritter.slotpool import SlotPool


# Adjusted architectural parameters. Everything else matches step 7b.
N_NEURONS: int = 256
K_SLOTS: int = 32
N_TIMESTEPS: int = 2_000
WINDOW_STEPS: int = 100
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)
# The step 7.5 close recommended "tonic=0" but the analytical fallout
# at this scale rules it out: at K=32 with A firing at ~4% per step,
# even v_max=2 yields only ~2.5 mV average cross input -- far below
# the ~15 mV B needs to cross threshold. So we sweep a range of tonic
# values from 0 to just above threshold; the right choice is the one
# where cross-input modulation produces the biggest change in B's
# firing rate across configurations. Overridden per-sweep below.
B_BASELINE_DRIVE_MV: float = 0.0
# v_max raised from the default 0.5 so stronger cross-weights can be
# exercised within the STDP clipping envelope.
V_MAX_OVERRIDE: float = 2.0

# Tonic sweep: zero (pure cross), subthreshold (under 15 mV), and the
# step 7 value (just over threshold). We run the full cross-weight
# sweep at each tonic value and look for a regime with real leverage.
TONIC_SWEEP: tuple[float, ...] = (0.0, 8.0, 14.0, 16.0)


Scenario = tuple[
    SlotPool,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


def _stdp_params_with_v_max() -> STDPParams:
    """Return STDPParams with v_max overridden to the Phase-3 value."""
    base = default_params()
    return base._replace(v_max=V_MAX_OVERRIDE)


def _build_scenario() -> Scenario:
    """Build the Phase 3 scenario traces."""
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
    valence_b_trace = jnp.zeros((N_TIMESTEPS,), dtype=jnp.float32)
    adrenaline_trace = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    pool_a = make_pool_for_partner(N_NEURONS, K_SLOTS, jax.random.PRNGKey(777))
    return (
        pool_a,
        i_ext_a_trace, i_ext_b_trace,
        valence_a_trace, valence_b_trace,
        adrenaline_trace, adrenaline_trace,
    )


def _frozen_pool(pre_ids: jax.Array, v_matrix: jax.Array) -> SlotPool:
    """Pool with given pre_ids / v; plasticity_rate=0 freezes STDP."""
    return SlotPool(
        pre_ids=pre_ids,
        v=v_matrix,
        plasticity_rate=jnp.zeros_like(v_matrix),
        active=jnp.ones_like(v_matrix, dtype=jnp.bool_),
        release_counter=jnp.zeros_like(pre_ids, dtype=jnp.int32),
    )


def _make_silent_pool() -> SlotPool:
    """Zero weights throughout."""
    pre_ids = jnp.zeros((N_NEURONS, K_SLOTS), dtype=jnp.int32)
    v = jnp.zeros((N_NEURONS, K_SLOTS), dtype=jnp.float32)
    return _frozen_pool(pre_ids, v)


def _make_cross_pool(cross_v: float, seed: int) -> SlotPool:
    """All slots bound to random A-side pres (index in [N, 2N))."""
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
    """All slots bound to own-side pres (index in [0, N))."""
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
    pool_b: SlotPool, scenario: Scenario,
) -> tuple[float, jax.Array, jax.Array]:
    """Run paired sim with hand-wired B; return fitness + per-seg rates."""
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
        _stdp_params_with_v_max(),
    )
    fitness = _prediction_fitness(spikes_a, spikes_b)
    n_segments = len(A_DRIVE_PROFILE)
    a_rate_hz = (
        spikes_a.astype(jnp.float32)
        .reshape(n_segments, -1, N_NEURONS)
        .mean(axis=(1, 2))
    ) * 1000.0
    b_rate_hz = (
        spikes_b.astype(jnp.float32)
        .reshape(n_segments, -1, N_NEURONS)
        .mean(axis=(1, 2))
    ) * 1000.0
    return fitness, a_rate_hz, b_rate_hz


def _build_controls(seed: int) -> list[tuple[str, SlotPool]]:
    """Seven hand-wired B configurations shared across tonic sweeps."""
    return [
        ("silent (v = 0)", _make_silent_pool()),
        ("all-recurrent v = 0.10", _make_recurrent_pool(0.10, seed + 1)),
        ("all-recurrent v = 0.50", _make_recurrent_pool(0.50, seed + 2)),
        ("all-cross v = 0.05", _make_cross_pool(0.05, seed + 3)),
        ("all-cross v = 0.50", _make_cross_pool(0.50, seed + 4)),
        ("all-cross v = 1.00", _make_cross_pool(1.00, seed + 5)),
        ("all-cross v = 1.50", _make_cross_pool(1.50, seed + 6)),
        ("all-cross v = 2.00", _make_cross_pool(2.00, seed + 7)),
    ]


Record = tuple[str, float, tuple[float, ...], float]


def _sweep_one_tonic(
    seed: int,
    tonic_mv: float,
) -> tuple[list[Record], tuple[float, ...]]:
    """Run all configurations at one tonic; return records + A rates."""
    # pylint: disable=global-statement
    global B_BASELINE_DRIVE_MV
    B_BASELINE_DRIVE_MV = tonic_mv

    scenario = _build_scenario()
    controls = _build_controls(seed)
    records: list[tuple[str, float, tuple[float, ...], float]] = []
    a_seg_cache: jax.Array | None = None
    for name, pool_b in controls:
        fitness, a_rate_hz, b_rate_hz = _evaluate(pool_b, scenario)
        if a_seg_cache is None:
            a_seg_cache = a_rate_hz
        b_seg = tuple(float(x) for x in b_rate_hz)
        b_range = float(b_rate_hz.max() - b_rate_hz.min())
        records.append((name, fitness, b_seg, b_range))
    assert a_seg_cache is not None
    a_seg = tuple(float(x) for x in a_seg_cache)
    return records, a_seg


def _print_sweep(
    tonic_mv: float,
    records: list[Record],
    a_seg: tuple[float, ...],
) -> None:
    """Print the per-configuration table for a single tonic value."""
    print(f"--- tonic = {tonic_mv:.1f} mV ---")
    header_cols = [
        "config".ljust(26),
        "fitness".rjust(12),
        "B seg 0".rjust(8),
        "B seg 1".rjust(8),
        "B seg 2".rjust(8),
        "B seg 3".rjust(8),
        "B range".rjust(8),
    ]
    header = " | ".join(header_cols)
    print(header)
    print("-" * len(header))
    for name, fitness, b_seg, b_range in records:
        row = " | ".join(
            [
                name.ljust(26),
                f"{fitness:12.3e}",
                f"{b_seg[0]:8.1f}",
                f"{b_seg[1]:8.1f}",
                f"{b_seg[2]:8.1f}",
                f"{b_seg[3]:8.1f}",
                f"{b_range:8.2f}",
            ]
        )
        print(row)
    a_range = max(a_seg) - min(a_seg)
    max_b_range = max(r[3] for r in records)
    best = max(records, key=lambda r: r[1])
    print(
        f"best fitness: '{best[0]}' at {best[1]:.3e}  |  "
        f"max B range: {max_b_range:.2f} Hz  "
        f"(A range: {a_range:.1f} Hz)"
    )
    print()


SweepEntry = tuple[float, list[Record], tuple[float, ...]]


def _verdict_across_tonics(sweep_results: list[SweepEntry]) -> None:
    """Issue a single leverage verdict across all tonic sweeps."""
    a_range = max(sweep_results[0][2]) - min(sweep_results[0][2])
    best_by_tonic: list[tuple[float, str, float, float]] = []
    for tonic_mv, records, _ in sweep_results:
        best = max(records, key=lambda r: r[1])
        max_b_range = max(r[3] for r in records)
        best_by_tonic.append((tonic_mv, best[0], best[1], max_b_range))

    print("--- across-tonic summary ---")
    print(
        "tonic (mV) | best config                  "
        "| best fitness |  max B range"
    )
    print("-" * 78)
    for tonic_mv, cfg, fit, rng in best_by_tonic:
        print(
            f"{tonic_mv:10.1f} | {cfg:<28} | {fit:12.3e} | "
            f"{rng:8.2f} Hz"
        )
    print()

    overall_best = max(best_by_tonic, key=lambda t: t[2])
    print(
        f"overall best: tonic={overall_best[0]:.1f} mV, "
        f"'{overall_best[1]}' at fitness = {overall_best[2]:.3e}  "
        f"(max B range = {overall_best[3]:.2f} Hz)"
    )
    print()

    within_leverage = overall_best[3] >= 0.2 * a_range
    all_fitnesses = [fit for _, _, fit, _ in best_by_tonic]
    across_leverage = (max(all_fitnesses) - min(all_fitnesses)) >= 1e-4

    if within_leverage and across_leverage:
        print(
            "VERDICT: architectural leverage achieved somewhere in the "
            "tonic sweep. Phase 4 (GA re-run at N=256) is worth "
            f"proceeding with, using tonic={overall_best[0]:.1f} mV as "
            "the operating regime."
        )
    elif across_leverage:
        print(
            "VERDICT: partial leverage. Fitness differs across tonic "
            "values but no configuration tracks A strongly. GA may help "
            "but may plateau short of strong prediction."
        )
    else:
        print(
            "VERDICT: no leverage at any tonic in the sweep. Pivot: "
            "higher N, larger K, different cross-weight layout, or a "
            "different task."
        )


def run(seed: int = 0) -> None:
    """Run the Phase 3 tonic sweep and report architectural-leverage verdict."""
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(f"scenario: N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}")
    print(
        f"v_max={V_MAX_OVERRIDE} (was 0.5); "
        f"tonic sweep over {list(TONIC_SWEEP)} mV"
    )
    print(f"A drive profile (mV): {list(A_DRIVE_PROFILE)}")
    print()

    sweep_results: list[SweepEntry] = []
    for tonic_mv in TONIC_SWEEP:
        records, a_seg = _sweep_one_tonic(seed, tonic_mv)
        _print_sweep(tonic_mv, records, a_seg)
        sweep_results.append((tonic_mv, records, a_seg))

    _verdict_across_tonics(sweep_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)
