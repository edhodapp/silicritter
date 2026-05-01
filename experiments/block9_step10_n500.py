"""Block 9: N=500 long-T measurement of step 10's closed-loop result.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

The load-bearing N=500 sweep that anchors step 10's headline closed-
loop adrenaline result. Phase 2 already confirmed at N=5 that step
10's qualitative claim holds at long T (~5.8x reduction from open-
loop to gain=200, with the long-T headline at -2.728e-05 superseding
step 10's originally reported -5.60e-5). Block 9 produces the
variance distribution that N=5 cannot resolve, supporting the
revalidation plan's "N=500 for headline numbers" discipline (see
project_revalidation_plan memory).

Sweep grid:
    durations T in {10k, 100k, 1M, 10M}    (same as Phase 2)
    seeds 0, 37, 74, ..., 18463            (500 seeds, stride 37)
    conditions: open_loop, gain=200        (single gain - the headline)

Total runs: 4 * 500 * 2 = 4000.

Wall-time estimate: ~84 hr on A100 dominated by T=10M (500 seeds *
2 conditions * ~268 s = 75 hr at T=10M alone). Exceeds Colab Pro+'s
~38 A100-hr/month compute budget; AWS spot (g6.xlarge L4 spot
~$0.24/hr) is the right path. Phase 2 already established that the
toolchain handles T=10M cleanly.

Why a single gain=200 (not the full Phase 2 gain sweep): Phase 2
showed gain=50 and gain=200 differ by ~0.4% at T=10M and are bit-
identical at T=1M and shorter. The headline number is gain=200; the
gain=50 / gain=10 trends were already characterized at N=5 in
Phase 2 and don't change qualitatively at N=500. Spending the
compute on those gains again is bad value vs. a tighter variance
bound at the headline.

Outputs:
    overnight_results/block9_step10_n500.csv
        seed,t_steps,condition,fitness,wall_sec
    overnight_results/block9_step10_n500.log
        time-stamped narrative for tail -f monitoring

Resumable: skips (seed, t_steps, condition) tuples already in the CSV.
Mandatory for AWS spot: spot interruption is part of the workflow,
not an exception. Re-launching after interruption picks up where it
stopped.

Usage:
    .venv/bin/python experiments/block9_step10_n500.py
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


# Network and scenario constants - identical to step 10 / Phase 2 so
# per-step dynamics match bit-exact and Block 9 seeds 0..148 align with
# Phase 2's N=5 anchors at the same loop seeds.
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

# Block 9 sweep configuration. N=500, single headline gain. Strides
# match step 10 / Phase 2 so the first 5 seeds (0, 37, 74, 111, 148)
# are the same anchors Phase 2 measured at N=5.
DURATIONS: tuple[int, ...] = (10_000, 100_000, 1_000_000, 10_000_000)
N_SEEDS: int = 500
SEED_BASE: int = 0
SEED_STRIDE: int = 37
GAINS: tuple[float, ...] = (200.0,)
OPEN_LOOP_LABEL: str = "open_loop"

CSV_PATH: Path = Path("overnight_results/block9_step10_n500.csv")
LOG_PATH: Path = Path("overnight_results/block9_step10_n500.log")


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
    long T: at T=10M, N=256, a ``(T, N)`` float32 array is ~10 GB. The
    rate-trace output_mode handles the OUTPUT side; this handles the
    input side. Per-step scalars are ~40 MB at T=10M.

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
    # pool_a uses fixed PRNGKey(777) to match step 10 / Phase 2
    # bit-exactly. Seed-driven variance comes only from pool_b - that's
    # the constellation step 10 measured.
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777),
    )
    # The seed+1 offset matches step10:240 / Phase 2 for bit-exact
    # reproduction of step 10's RNG state at the same loop seeds.
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
    """Windowed-mean prediction fitness from per-step rate scalars."""
    assert t_steps % WINDOW_STEPS == 0, (
        f"t_steps={t_steps} not divisible by WINDOW_STEPS={WINDOW_STEPS}"
    )
    n_windows = t_steps // WINDOW_STEPS
    # n_windows == 1 would give empty lead-lag diff and propagate NaN
    # silently. Unreachable under DURATIONS as configured (smallest
    # T=10k => 100 windows) but pinned for any future smaller-T addition.
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
    """Open-loop run (constant adrenaline = baseline on B)."""
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
    """Closed-loop run at given gain."""
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
    """Run only the conditions in ``conditions_to_run`` for (seed, t_steps).

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
    """Read CSV, return set of (seed, t_steps, condition) already done."""
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
    """Append a single result row, writing the CSV header on first write.

    Header detection uses ``stat().st_size == 0`` (not just ``not
    exists()``): if a prior run created the file but crashed before
    writing the header (or before the first row), the file survives
    at zero bytes. Without this guard the next run would skip writing
    the header, and ``_completed_pairs`` would misinterpret the first
    real row as the header on resume.
    """
    needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(
                ["seed", "t_steps", "condition", "fitness", "wall_sec"],
            )
        writer.writerow([
            seed, t_steps, condition,
            f"{fitness:.6e}", f"{wall_sec:.2f}",
        ])


def _log(message: str) -> None:
    """Append timestamped line to LOG_PATH and stdout.

    Caller (``main``) is responsible for ensuring ``LOG_PATH.parent``
    exists. Doing the mkdir here would waste syscalls on every line of
    a multi-thousand-step batch log.
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
    """Conditions still to run for (seed, t_steps) given completed."""
    all_conditions = [OPEN_LOOP_LABEL] + [f"gain={g:g}" for g in GAINS]
    return [
        c for c in all_conditions
        if (seed, t_steps, c) not in completed
    ]


MAX_CONSECUTIVE_FAILURES: int = 5


def main() -> None:
    """Drive the N=500 long-T sweep, resumable from a partial CSV.

    On AWS spot, a persistent failure (CUDA OOM at T=10M, driver
    error, etc.) would otherwise log 500 FAILED lines and finish with
    a partial CSV - silently burning ~$30 of spot compute. The
    consecutive-failure counter aborts the run after
    ``MAX_CONSECUTIVE_FAILURES`` (= 5) successive ``_evaluate_one``
    failures, preferring a loud crash that the user notices over a
    silent wrong-result completion.
    """
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_pairs(CSV_PATH)
    expected = _expected_total()
    _log(
        f"block 9 N=500 long-T sweep: durations={list(DURATIONS)}, "
        f"N={N_SEEDS} seeds, {len(GAINS) + 1} conditions per (seed, T)"
    )
    _log(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    _log(f"already complete: {len(completed)} / {expected}")
    overall_start = time.time()
    consecutive_failures = 0
    for t_steps in DURATIONS:
        for i in range(N_SEEDS):
            seed = SEED_BASE + i * SEED_STRIDE
            todo = _conditions_for_seed_t(seed, t_steps, completed)
            if not todo:
                continue  # Quiet skip - 4000-row sweep would otherwise spam.
            _log(
                f"start seed={seed} t={t_steps} "
                f"({len(todo)} of {len(GAINS) + 1} conditions)"
            )
            try:
                results = _evaluate_one(seed, t_steps, todo)
                consecutive_failures = 0
            except (RuntimeError, MemoryError) as exc:
                # Catch sim-shape failures (CUDA OOM, JAX runtime errors).
                # Do NOT catch AssertionError or ValueError: those signal
                # programmer error and must propagate so the user sees a
                # loud crash rather than a quietly-logged-and-skipped
                # wrong result.
                _log(
                    f"FAILED seed={seed} t={t_steps}: "
                    f"{type(exc).__name__}: {exc}"
                )
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    _log(
                        f"ABORTING: {consecutive_failures} consecutive "
                        f"failures - the configuration appears broken; "
                        f"raising last exception so an external "
                        f"supervisor can react"
                    )
                    raise
                continue
            for condition, (fitness, wall) in results.items():
                _append_row(
                    CSV_PATH, seed, t_steps, condition, fitness, wall,
                )
                _log(
                    f"  done seed={seed} t={t_steps} {condition} "
                    f"fitness={fitness:.3e} wall={wall:.1f}s"
                )
    _log(f"block 9 complete: total wall {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
