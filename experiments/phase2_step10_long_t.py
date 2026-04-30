"""Phase 2: long-T reproducer for step 10's closed-loop adrenaline result.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Sweeps T in {10k, 100k, 1M, 10M} at N=5 seeds x 4 conditions
(open-loop, closed-loop gain in {10, 50, 200}) to confirm step 10's
headline number (~-5.6e-5 at gain=200) holds across training durations.

Background: step 10's original sweep ran at T=2000 (the smoke-test
duration). Block 7 of the original revalidation overnight batch
collapsed an N=5 result down to z=0.53 null at N=20, triggering an
audit of every load-bearing claim. This Phase 2 run is the first long-T
look at step 10 - if the headline holds at T=10M with N=5, it's a
green-light to commit the much larger Block 9 (N=500, durations
10k-10M) cycles. If it doesn't, we re-anchor before scaling.

Memory: simulate_paired and simulate_closedloop return full per-step
spike rasters of shape (T, n_neurons) under the default
``output_mode="raster"``, which OOMs the GTX 1050 Mobile's 4 GB VRAM
at T=10M (5 GB raster). This script passes ``output_mode="rate"`` so
each scan output is a per-step population-mean scalar (T,), bringing
the trace size down to ~40 MB at T=10M. Math is identical for the
windowed-mean fitness used here.

Outputs:
    overnight_results/phase2_step10_long_t.csv
        seed,t_steps,condition,fitness,wall_sec
    overnight_results/phase2_step10_long_t.log
        time-stamped narrative for tail -f monitoring

Resumable: skips (seed, t_steps, condition) tuples already in the CSV.
Re-launching after a crash picks up where it left off.

Usage:
    .venv/bin/python experiments/phase2_step10_long_t.py
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


# Network and scenario constants - identical to step 10 so per-step
# dynamics match bit-exact at any T.
N_NEURONS: int = 256
K_SLOTS: int = 32
WINDOW_STEPS: int = 100
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)
B_BASELINE_DRIVE_MV: float = 16.0
V_MAX: float = 2.0
INHIBITORY_FRACTION: float = 0.2
I_WEIGHT_MULTIPLIER: float = 4.0
CONTROLLER_DECAY: float = 0.98
BASELINE_ADRENALINE: float = 1.0
ADR_MIN: float = 0.5
ADR_MAX: float = 3.0

# Phase 2 sweep configuration.
DURATIONS: tuple[int, ...] = (10_000, 100_000, 1_000_000, 10_000_000)
N_SEEDS: int = 5
SEED_BASE: int = 0
# Stride between seeds matches step 10's _run_multi_seed; ensures
# independent draws while allowing direct comparison to step 10's
# existing N=5 anchors at T=2000.
SEED_STRIDE: int = 37
GAINS: tuple[float, ...] = (10.0, 50.0, 200.0)
OPEN_LOOP_LABEL: str = "open_loop"

CSV_PATH: Path = Path("overnight_results/phase2_step10_long_t.csv")
LOG_PATH: Path = Path("overnight_results/phase2_step10_long_t.log")


def _stdp_params() -> STDPParams:
    return default_params()._replace(v_max=V_MAX)


def _make_cross_e_only_pool(cross_v: float, seed: int) -> SlotPool:
    """Step 9's winning configuration: all slots bound to A's E neurons."""
    rng = jax.random.PRNGKey(seed)
    n_excitatory = N_NEURONS - int(N_NEURONS * INHIBITORY_FRACTION)
    pre_ids = jax.random.randint(
        rng, (N_NEURONS, K_SLOTS),
        minval=N_NEURONS,
        maxval=N_NEURONS + n_excitatory,
        dtype=jnp.int32,
    )
    v = jnp.full((N_NEURONS, K_SLOTS), cross_v, dtype=jnp.float32)
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


def _build_traces(
    t_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build per-step scalar drive traces of length t_steps.

    Returns four ``(t_steps,)`` arrays - one scalar per step for each
    of i_ext_a, i_ext_b, valence, adrenaline_a. ``simulate_paired`` and
    ``simulate_closedloop`` broadcast each scalar across the N-neuron
    axis inside their scan bodies.

    Per-step scalars (not full ``(T, N)`` rasters) are mandatory at
    long T: at T=10M, N=256, a ``(T, N)`` float32 array is ~10 GB, so
    two of them OOM the GTX 1050 Mobile's 4 GB VRAM before the scan
    even starts. Per-step scalars are ~40 MB at T=10M.

    Behavior matches step 10 (which uses ``(T, N)`` rasters at its
    smaller T=2000): each neuron sees the segment's drive level on a
    given step, identical to the broadcast result.

    A drive profile has 4 levels splitting t_steps into equal segments;
    t_steps must be a multiple of len(A_DRIVE_PROFILE).
    """
    assert t_steps % len(A_DRIVE_PROFILE) == 0, (
        f"t_steps={t_steps} not divisible by A drive profile length "
        f"{len(A_DRIVE_PROFILE)}"
    )
    seg_len = t_steps // len(A_DRIVE_PROFILE)
    i_ext_a = jnp.repeat(
        jnp.asarray(A_DRIVE_PROFILE, dtype=jnp.float32), seg_len,
    )
    i_ext_b = jnp.full((t_steps,), B_BASELINE_DRIVE_MV, dtype=jnp.float32)
    valence = jnp.zeros((t_steps,), dtype=jnp.float32)
    adrenaline_a = jnp.ones((t_steps,), dtype=jnp.float32)
    return i_ext_a, i_ext_b, valence, adrenaline_a


def _ctrl_params(gain: float) -> ControllerParams:
    return ControllerParams(
        decay=CONTROLLER_DECAY, baseline=BASELINE_ADRENALINE, gain=gain,
        adr_min=ADR_MIN, adr_max=ADR_MAX,
    )


SetupBundle = tuple[
    PairedState, jax.Array, jax.Array,
    jax.Array, jax.Array, jax.Array, jax.Array,
]


def _setup_for_seed(seed: int, t_steps: int) -> SetupBundle:
    """Build initial state, E/I masks, and input traces for one seed/T."""
    # pool_a uses the fixed seed 777 to match step 10's reproducer
    # bit-exactly (step10:_setup_for_seed also pins PRNGKey(777) for
    # pool_a). Multi-seed variability in this sweep deliberately comes
    # only from pool_b - that's the constellation step 10 measured, and
    # Phase 2's job is to confirm step 10's headline at long T, not to
    # explore a different randomization scheme.
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777),
    )
    # The seed+1 offset matches step10:240 for bit-exact reproduction.
    # Original rationale in step 10 is unknown (likely a historical
    # artifact from a draft where pool_a used PRNGKey(seed)); preserved
    # here so Phase 2's RNG state matches step 10 at the same loop seeds.
    pool_b = _make_cross_e_only_pool(V_MAX, seed + 1)
    initial_state = _build_initial_state(pool_a, pool_b)
    a_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    i_ext_a, i_ext_b, valence, adrenaline_a = _build_traces(t_steps)
    return (
        initial_state, a_is_inh, b_is_inh,
        i_ext_a, i_ext_b, valence, adrenaline_a,
    )


def _prediction_fitness_from_rates(
    rate_a: jax.Array, rate_b: jax.Array, t_steps: int,
) -> float:
    """Windowed-mean prediction fitness from per-step population-mean rates.

    Mathematically identical to step 10's ``_prediction_fitness`` when
    given ``rate_x = spikes_x.mean(axis=1)``: reshapes ``(T,)`` to
    ``(n_windows, WINDOW_STEPS)``, means over the WINDOW_STEPS axis,
    then computes ``-E[(b[:-1] - a[1:])**2]`` over the resulting
    windows.
    """
    assert t_steps % WINDOW_STEPS == 0, (
        f"t_steps={t_steps} not divisible by WINDOW_STEPS={WINDOW_STEPS}"
    )
    n_windows = t_steps // WINDOW_STEPS
    # Defense-in-depth: the lead-lag formula b[:-1] - a[1:] is empty
    # when n_windows == 1, which would make jnp.mean return NaN and
    # silently propagate into the CSV. Unreachable under DURATIONS as
    # configured (smallest T=10k => 100 windows), but worth pinning so
    # a future smaller-T addition fails loudly at the call site.
    assert n_windows > 1, (
        f"n_windows={n_windows} (T={t_steps}, WINDOW_STEPS={WINDOW_STEPS}) "
        f"would give an empty lead-lag diff and propagate NaN; need T > "
        f"WINDOW_STEPS for fitness to be defined"
    )
    a_rate_w = rate_a.reshape(n_windows, WINDOW_STEPS).mean(axis=1)
    b_rate_w = rate_b.reshape(n_windows, WINDOW_STEPS).mean(axis=1)
    return float(-jnp.mean((b_rate_w[:-1] - a_rate_w[1:]) ** 2))


def _run_open_loop(
    setup: SetupBundle,
) -> tuple[jax.Array, jax.Array, float]:
    """Open-loop run (constant adrenaline = baseline on B); return rates."""
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
        i_weight_multiplier=I_WEIGHT_MULTIPLIER,
        output_mode="rate",
    )
    rate_a.block_until_ready()
    rate_b.block_until_ready()
    return rate_a, rate_b, time.time() - t0


def _run_closed_loop(
    setup: SetupBundle, gain: float,
) -> tuple[jax.Array, jax.Array, float]:
    """Run closed-loop at given gain; return rates + wall_sec."""
    (initial_state, a_is_inh, b_is_inh,
     i_ext_a, i_ext_b, valence, adrenaline_a) = setup
    t0 = time.time()
    _, rate_a, rate_b, _ = simulate_closedloop(
        initial_state, _ctrl_params(gain),
        i_ext_a, i_ext_b, valence, valence, adrenaline_a,
        _stdp_params(),
        gain_mode="tau_m_scale",
        a_is_inhibitory=a_is_inh,
        b_is_inhibitory=b_is_inh,
        i_weight_multiplier=I_WEIGHT_MULTIPLIER,
        output_mode="rate",
    )
    rate_a.block_until_ready()
    rate_b.block_until_ready()
    return rate_a, rate_b, time.time() - t0


def _evaluate_one(
    seed: int, t_steps: int, conditions_to_run: list[str],
) -> dict[str, tuple[float, float]]:
    """Run only the conditions in conditions_to_run for (seed, t_steps).

    Returns ``{condition: (fitness, wall_sec)}``. Conditions not requested
    are not computed (lets the resume path skip already-done work).
    """
    setup = _setup_for_seed(seed, t_steps)
    results: dict[str, tuple[float, float]] = {}
    if OPEN_LOOP_LABEL in conditions_to_run:
        rate_a, rate_b, wall = _run_open_loop(setup)
        fitness = _prediction_fitness_from_rates(rate_a, rate_b, t_steps)
        results[OPEN_LOOP_LABEL] = (fitness, wall)
    for gain in GAINS:
        condition = f"gain={gain:g}"
        if condition not in conditions_to_run:
            continue
        rate_a, rate_b, wall = _run_closed_loop(setup, gain)
        fitness = _prediction_fitness_from_rates(rate_a, rate_b, t_steps)
        results[condition] = (fitness, wall)
    return results


def _completed_pairs(csv_path: Path) -> set[tuple[int, int, str]]:
    """Read CSV, return set of (seed, t_steps, condition) already done.

    Returns empty set if the file does not exist (first-launch case).
    """
    if not csv_path.exists():
        return set()
    completed: set[tuple[int, int, str]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((
                int(row["seed"]),
                int(row["t_steps"]),
                row["condition"],
            ))
    return completed


def _append_row(
    csv_path: Path, seed: int, t_steps: int,
    condition: str, fitness: float, wall_sec: float,
) -> None:
    """Append a single result row, writing the CSV header on first write."""
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["seed", "t_steps", "condition", "fitness", "wall_sec"],
            )
        writer.writerow([
            seed, t_steps, condition,
            f"{fitness:.6e}", f"{wall_sec:.2f}",
        ])


def _log(message: str) -> None:
    """Append timestamped line to log_path and stdout.

    Caller is responsible for ensuring ``LOG_PATH.parent`` exists -
    ``main`` does this once at startup. Doing it per call wastes
    syscalls on every line of a multi-thousand-step batch log.
    """
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _expected_total() -> int:
    """Total (seed, t_steps, condition) tuples in the full sweep grid."""
    return len(DURATIONS) * N_SEEDS * (1 + len(GAINS))


def _conditions_for_seed_t(
    seed: int, t_steps: int, completed: set[tuple[int, int, str]],
) -> list[str]:
    """Conditions still to run for (seed, t_steps) given the completed set."""
    all_conditions = [OPEN_LOOP_LABEL] + [f"gain={g:g}" for g in GAINS]
    return [
        c for c in all_conditions
        if (seed, t_steps, c) not in completed
    ]


def main() -> None:
    """Drive the long-T sweep, resumable from a partial CSV."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_pairs(CSV_PATH)
    expected = _expected_total()
    _log(
        f"phase 2 long-T sweep: durations={list(DURATIONS)}, "
        f"N={N_SEEDS} seeds, {len(GAINS) + 1} conditions per (seed, T)"
    )
    _log(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    _log(f"already complete: {len(completed)} / {expected}")
    overall_start = time.time()
    for t_steps in DURATIONS:
        for i in range(N_SEEDS):
            seed = SEED_BASE + i * SEED_STRIDE
            todo = _conditions_for_seed_t(seed, t_steps, completed)
            if not todo:
                _log(f"skip seed={seed} t={t_steps} (all done)")
                continue
            _log(
                f"start seed={seed} t={t_steps} "
                f"({len(todo)} of {len(GAINS) + 1} conditions)"
            )
            try:
                results = _evaluate_one(seed, t_steps, todo)
            except (RuntimeError, MemoryError) as exc:
                # Catch sim-shape failures (CUDA OOM, JAX runtime errors)
                # and continue with the next (seed, t_steps). Do NOT
                # catch AssertionError or ValueError: those signal
                # programmer error in this script's invariants /
                # constants, and per project policy they must propagate
                # so the user sees a loud crash rather than a quietly-
                # logged-and-skipped wrong result.
                _log(
                    f"FAILED seed={seed} t={t_steps}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            for condition, (fitness, wall) in results.items():
                _append_row(
                    CSV_PATH, seed, t_steps, condition, fitness, wall,
                )
                _log(
                    f"  done seed={seed} t={t_steps} {condition} "
                    f"fitness={fitness:.3e} wall={wall:.1f}s"
                )
    _log(f"phase 2 complete: total wall {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
