"""Block 10: N=100 multi-seed E/I comparison at step 13's grid.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Confirms D008's (i_frac=0.2, i_mult=8.0) E/I operating-point choice
at N=100, supersedes step 13's existing N=20 anchor. Per the
revalidation plan (project_revalidation_plan memory): "Block 10:
step 13 E/I (0.2,8.0) vs (0.2,4.0), N=100."

Sweep grid:
    i_mult in {4.0, 8.0}             (canonical vs D008 candidate)
    seeds 0, 37, 74, ..., 3663       (100 seeds, stride 37)
    conditions: open_loop, closed_loop (gain=50)

Total runs: 2 * 100 * 2 = 400. Wall-time on GTX 1050 Mobile ~30-60 min
(per-run wall ~5-9 s at T=2000 based on step 13 anchor timing).

Outputs:
    overnight_results/block10_step13_ei_n100.csv
        seed,i_mult,condition,fitness,wall_sec
    overnight_results/block10_step13_ei_n100.log

Resumable via _completed_pairs. Resumes naturally after laptop sleep
or process kill.

Why output_mode="rate" even at T=2000 (small enough to fit raster
trivially): consistency with Phase 2 / Block 9, less memory churn,
mathematically identical for the windowed-mean fitness used here.

Usage:
    .venv/bin/python experiments/block10_step13_ei_n100.py
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp

from silicritter.closedloop import (
    ControllerParams,
    simulate_closedloop,
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
from silicritter.slotpool import (
    SlotPool,
    assign_ei_identity,
)


# Network and scenario constants - identical to step 13 so per-step
# dynamics match step 13's existing anchors at the same loop seeds.
N_NEURONS: int = 256
K_SLOTS: int = 32
T_STEPS: int = 2_000
WINDOW_STEPS: int = 100
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)
B_BASELINE_DRIVE_MV: float = 16.0
V_MAX: float = 2.0
CONTROLLER_DECAY: float = 0.98
BASELINE_ADRENALINE: float = 1.0
ADR_MIN: float = 0.5
ADR_MAX: float = 3.0
CLOSED_LOOP_GAIN: float = 50.0

# Block 10 sweep configuration. i_frac fixed; i_mult varies between
# canonical (4.0) and D008 candidate (8.0).
I_FRAC: float = 0.2
I_MULTS: tuple[float, ...] = (4.0, 8.0)
N_SEEDS: int = 100
SEED_BASE: int = 0
SEED_STRIDE: int = 37
CONDITIONS: tuple[str, ...] = ("open_loop", "closed_loop")

CSV_PATH: Path = Path("overnight_results/block10_step13_ei_n100.csv")
LOG_PATH: Path = Path("overnight_results/block10_step13_ei_n100.log")

MAX_CONSECUTIVE_FAILURES: int = 5


def _stdp_params() -> STDPParams:
    return default_params()._replace(v_max=V_MAX)


def _ctrl_params() -> ControllerParams:
    return ControllerParams(
        decay=CONTROLLER_DECAY, baseline=BASELINE_ADRENALINE,
        gain=CLOSED_LOOP_GAIN,
        adr_min=ADR_MIN, adr_max=ADR_MAX,
    )


def _make_cross_e_only_pool(seed: int) -> SlotPool:
    """Step 13's hand-wired cross-E-only configuration: all B slots
    bound to A's E neurons (which are the first N*(1-i_frac) of A)."""
    rng = jax.random.PRNGKey(seed)
    n_excitatory = N_NEURONS - int(N_NEURONS * I_FRAC)
    pre_ids = jax.random.randint(
        rng, (N_NEURONS, K_SLOTS),
        minval=N_NEURONS,
        maxval=N_NEURONS + n_excitatory,
        dtype=jnp.int32,
    )
    v = jnp.full((N_NEURONS, K_SLOTS), V_MAX, dtype=jnp.float32)
    return SlotPool(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=jnp.zeros_like(v),
        active=jnp.ones_like(v, dtype=jnp.bool_),
        release_counter=jnp.zeros_like(pre_ids, dtype=jnp.int32),
    )


def _build_initial_state(pool_a: SlotPool, pool_b: SlotPool) -> PairedState:
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


def _build_traces() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-step scalar drive traces (broadcast to N inside the scan).

    Same shape as Phase 2 / Block 9; T_STEPS is small here but the
    scalar pattern keeps allocation overhead low across 400 calls.
    """
    assert T_STEPS % len(A_DRIVE_PROFILE) == 0, (
        f"T_STEPS={T_STEPS} not divisible by A drive profile length "
        f"{len(A_DRIVE_PROFILE)}"
    )
    seg_len = T_STEPS // len(A_DRIVE_PROFILE)
    i_ext_a = jnp.repeat(
        jnp.asarray(A_DRIVE_PROFILE, dtype=jnp.float32), seg_len,
    )
    i_ext_b = jnp.full((T_STEPS,), B_BASELINE_DRIVE_MV, dtype=jnp.float32)
    valence = jnp.zeros((T_STEPS,), dtype=jnp.float32)
    adrenaline_a = jnp.ones((T_STEPS,), dtype=jnp.float32)
    return i_ext_a, i_ext_b, valence, adrenaline_a


SetupBundle = tuple[
    PairedState, jax.Array, jax.Array,
    jax.Array, jax.Array, jax.Array, jax.Array,
]


def _setup_for_seed(seed: int) -> SetupBundle:
    """Build initial state, E/I masks, and input traces for one seed.

    pool_a uses fixed PRNGKey(777) to match step 10 / step 13; pool_b
    uses PRNGKey(seed + 1) for the same reason (matches step 13's
    anchor RNG state at corresponding loop seeds).
    """
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777),
    )
    pool_b = _make_cross_e_only_pool(seed + 1)
    initial_state = _build_initial_state(pool_a, pool_b)
    a_is_inh = assign_ei_identity(N_NEURONS, I_FRAC)
    b_is_inh = assign_ei_identity(N_NEURONS, I_FRAC)
    i_ext_a, i_ext_b, valence, adrenaline_a = _build_traces()
    return (
        initial_state, a_is_inh, b_is_inh,
        i_ext_a, i_ext_b, valence, adrenaline_a,
    )


def _prediction_fitness_from_rates(
    rate_a: jax.Array, rate_b: jax.Array,
) -> float:
    """Windowed-mean prediction fitness from per-step rate scalars.

    Same math as Phase 2 / Block 9, but T_STEPS is fixed at module
    level here (this experiment doesn't sweep T) so the function takes
    only the rate traces.
    """
    assert T_STEPS % WINDOW_STEPS == 0, (
        f"T_STEPS={T_STEPS} not divisible by WINDOW_STEPS={WINDOW_STEPS}"
    )
    n_windows = T_STEPS // WINDOW_STEPS
    assert n_windows > 1, (
        f"n_windows={n_windows} would give an empty lead-lag diff and "
        f"propagate NaN; need T_STEPS > WINDOW_STEPS for fitness to be "
        f"defined"
    )
    a_rate_w = rate_a.reshape(n_windows, WINDOW_STEPS).mean(axis=1)
    b_rate_w = rate_b.reshape(n_windows, WINDOW_STEPS).mean(axis=1)
    return float(-jnp.mean((b_rate_w[:-1] - a_rate_w[1:]) ** 2))


def _run_open_loop(
    setup: SetupBundle, i_mult: float,
) -> tuple[jax.Array, jax.Array, float]:
    """Open-loop run at given i_mult."""
    (initial_state, a_is_inh, b_is_inh,
     i_ext_a, i_ext_b, valence, adrenaline_a) = setup
    adrenaline_b = jnp.full_like(adrenaline_a, BASELINE_ADRENALINE)
    t0 = time.time()
    _, rate_a, rate_b = simulate_paired(
        initial_state,
        i_ext_a, i_ext_b, valence, valence, adrenaline_a, adrenaline_b,
        _stdp_params(),
        gain_mode="tau_m_scale",
        a_is_inhibitory=a_is_inh,
        b_is_inhibitory=b_is_inh,
        i_weight_multiplier=i_mult,
        output_mode="rate",
    )
    rate_a.block_until_ready()
    rate_b.block_until_ready()
    return rate_a, rate_b, time.time() - t0


def _run_closed_loop(
    setup: SetupBundle, i_mult: float,
) -> tuple[jax.Array, jax.Array, float]:
    """Closed-loop run at given i_mult."""
    (initial_state, a_is_inh, b_is_inh,
     i_ext_a, i_ext_b, valence, adrenaline_a) = setup
    t0 = time.time()
    _, rate_a, rate_b, _ = simulate_closedloop(
        initial_state, _ctrl_params(),
        i_ext_a, i_ext_b, valence, valence, adrenaline_a,
        _stdp_params(),
        gain_mode="tau_m_scale",
        a_is_inhibitory=a_is_inh,
        b_is_inhibitory=b_is_inh,
        i_weight_multiplier=i_mult,
        output_mode="rate",
    )
    rate_a.block_until_ready()
    rate_b.block_until_ready()
    return rate_a, rate_b, time.time() - t0


def _evaluate_one(
    seed: int, i_mult: float, conditions_to_run: list[str],
) -> dict[str, tuple[float, float]]:
    """Run requested conditions for (seed, i_mult).

    Returns ``{condition: (fitness, wall_sec)}``.
    """
    setup = _setup_for_seed(seed)
    results: dict[str, tuple[float, float]] = {}
    if "open_loop" in conditions_to_run:
        rate_a, rate_b, wall = _run_open_loop(setup, i_mult)
        fitness = _prediction_fitness_from_rates(rate_a, rate_b)
        results["open_loop"] = (fitness, wall)
    if "closed_loop" in conditions_to_run:
        rate_a, rate_b, wall = _run_closed_loop(setup, i_mult)
        fitness = _prediction_fitness_from_rates(rate_a, rate_b)
        results["closed_loop"] = (fitness, wall)
    return results


def _completed_pairs(csv_path: Path) -> set[tuple[int, float, str]]:
    """Read CSV, return set of (seed, i_mult, condition) already done."""
    if not csv_path.exists():
        return set()
    completed: set[tuple[int, float, str]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((
                int(row["seed"]),
                float(row["i_mult"]),
                row["condition"],
            ))
    return completed


def _append_row(
    csv_path: Path, seed: int, i_mult: float,
    condition: str, fitness: float, wall_sec: float,
) -> None:
    """Append one result row, writing the CSV header on first write.

    Header detection uses ``stat().st_size == 0`` (not just ``not
    exists()``): if a prior run created the file but crashed before
    writing the header (or even before the first row), the file
    survives at zero bytes. Without this guard the next run would
    skip writing the header, and ``_completed_pairs`` would
    misinterpret the first real row as the header on resume.
    """
    needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(
                ["seed", "i_mult", "condition", "fitness", "wall_sec"],
            )
        writer.writerow([
            seed, i_mult, condition,
            f"{fitness:.6e}", f"{wall_sec:.2f}",
        ])


def _log(message: str) -> None:
    """Append timestamped line to LOG_PATH and stdout."""
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _expected_total() -> int:
    """Total (seed, i_mult, condition) tuples in the full sweep grid."""
    return len(I_MULTS) * N_SEEDS * len(CONDITIONS)


def _conditions_for_seed_imult(
    seed: int, i_mult: float,
    completed: set[tuple[int, float, str]],
) -> list[str]:
    """Conditions still to run for (seed, i_mult) given completed."""
    return [
        c for c in CONDITIONS
        if (seed, i_mult, c) not in completed
    ]


def main() -> None:
    """Drive the N=100 E/I sweep, resumable from a partial CSV.

    The MAX_CONSECUTIVE_FAILURES counter aborts after 5 successive
    _evaluate_one failures - prevents silent burning through the rest
    of a 400-row grid if the configuration is broken.
    """
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_pairs(CSV_PATH)
    expected = _expected_total()
    _log(
        f"block 10 N=100 E/I sweep: i_mults={list(I_MULTS)}, "
        f"i_frac={I_FRAC}, T={T_STEPS}, gain={CLOSED_LOOP_GAIN}, "
        f"N={N_SEEDS} seeds"
    )
    _log(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    _log(f"already complete: {len(completed)} / {expected}")
    overall_start = time.time()
    consecutive_failures = 0
    for i_mult in I_MULTS:
        for i in range(N_SEEDS):
            seed = SEED_BASE + i * SEED_STRIDE
            todo = _conditions_for_seed_imult(seed, i_mult, completed)
            if not todo:
                continue
            try:
                results = _evaluate_one(seed, i_mult, todo)
                consecutive_failures = 0
            except (RuntimeError, MemoryError) as exc:
                _log(
                    f"FAILED seed={seed} i_mult={i_mult} "
                    f"todo={todo}: {type(exc).__name__}: {exc}"
                )
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    _log(
                        f"ABORTING: {consecutive_failures} consecutive "
                        f"failures - configuration appears broken; "
                        f"raising last exception"
                    )
                    raise
                continue
            for condition, (fitness, wall) in results.items():
                _append_row(
                    CSV_PATH, seed, i_mult, condition, fitness, wall,
                )
                _log(
                    f"  done seed={seed} i_mult={i_mult} {condition} "
                    f"fitness={fitness:.3e} wall={wall:.1f}s"
                )
    _log(f"block 10 complete: total wall {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
