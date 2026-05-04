"""Block 11b: capture genomes, topology stats, eval at 100 scenarios.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Three-phase orchestration that resolves Block 11's three open
scientific findings (per perf_history 2026-05-04 entry):

  Phase 1 - Re-run all 40 Block 11 GAs (deterministic from
            ga_seed) and capture the evolved best genomes. Save
            to a pickle file. Skip Phase 1 if the pickle already
            exists. (~6 min on GTX 1050 if Phase 1 runs.)

  Phase 2 - Decode each evolved genome to a SlotPool and compute
            structural statistics (cross-bound %, cross-E vs
            cross-I split, recurrent split, weight distribution).
            Write per-genome stats to a CSV. Always recomputed -
            instant, fits comfortably in laptop RAM.

  Phase 3 - For each evolved genome, evaluate fitness at 100
            different scenario eval_seeds (varying pool_a's
            PRNGKey from training's PRNGKey(777) baseline). Write
            per-eval rows to a CSV. Resumable. (~20 min for 4000
            evals at ~0.3 s each on the GTX 1050.)

Outputs:
    overnight_results/block11b_genomes.pkl
        dict[(ga_seed, condition)] -> CPPNGenome
    overnight_results/block11b_genome_stats.csv
        per-genome topology statistics
    overnight_results/block11b_eval_seeds.csv
        ga_seed,condition,eval_seed,fitness,wall_sec
    overnight_results/block11b.log
        time-stamped narrative

Why this addresses the open findings from Block 11:

(1) "Extract the best closed-loop GA topology" - Phase 1 captures
    every GA's evolved genome; Phase 2's CSV identifies the best
    by fitness and exposes its structural fingerprint.

(2) "Open-loop +9% GA win - what topology features?" - Phase 2's
    structural stats across all 40 genomes let us see whether
    open-loop winners share specific patterns (e.g. higher cross-E
    %, distinctive weight distributions) vs. hand-wired
    cross-E-only.

(3) "Closed-loop +4% mean win - selection-on-noise or real?" -
    Phase 3 re-evaluates every genome on 100 novel scenarios; if
    the evolved genomes still beat hand-wired on average across
    novel scenarios, the win is real, not selection-bias artifact.

Resumability:
    - Phase 1 skipped if pickle exists.
    - Phase 2 always recomputed (instant).
    - Phase 3 resumes via _completed_evals on (ga_seed, condition,
      eval_seed) tuple.

Usage:
    .venv/bin/python experiments/block11b_genome_analysis.py
"""

from __future__ import annotations

import csv
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

# pylint: disable=import-error
import step11_cppn_closedloop as s11
from silicritter.cppn import CPPNGenome, decode_cppn_to_pool
from silicritter.lif import init_state
from silicritter.paired import (
    PairedState,
    make_pool_for_partner,
)
from silicritter.plasticity import (
    PlasticNetState,
    init_traces,
)
from silicritter.slotpool import SlotPool, assign_ei_identity


# Block 11's grid: 20 ga_seeds x 2 conditions = 40 genomes.
N_GAS: int = 20
SEED_BASE: int = 0
SEED_STRIDE: int = 1
CONDITIONS: tuple[str, ...] = ("open_loop", "closed_loop")

# Block 11b Phase 3: 100 eval_seeds per genome, disjoint from training
# scenario's PRNGKey(777) and from any GA-init seed range.
N_EVAL_SEEDS: int = 100
EVAL_SEED_BASE: int = 1000
EVAL_SEED_STRIDE: int = 37

GENOMES_PKL: Path = Path("overnight_results/block11b_genomes.pkl")
STATS_CSV: Path = Path("overnight_results/block11b_genome_stats.csv")
EVALS_CSV: Path = Path("overnight_results/block11b_eval_seeds.csv")
LOG_PATH: Path = Path("overnight_results/block11b.log")

STATS_KEYS: tuple[str, ...] = (
    "cross_pct", "cross_e_pct", "cross_i_pct", "recurrent_pct",
    "v_mean", "v_std", "v_max", "v_mean_cross_e", "v_mean_recurrent",
)

MAX_CONSECUTIVE_FAILURES: int = 5


# ----- Logging -------------------------------------------------------------


def _log(message: str) -> None:
    """Append a timestamped line to LOG_PATH and stdout."""
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----- Sweep-grid helpers --------------------------------------------------


def _genome_pairs() -> list[tuple[int, str]]:
    """All (ga_seed, condition) pairs from Block 11's grid, in order."""
    return [
        (SEED_BASE + i * SEED_STRIDE, condition)
        for i in range(N_GAS)
        for condition in CONDITIONS
    ]


def _expected_total_evals() -> int:
    """Total (ga_seed, condition, eval_seed) tuples in Phase 3."""
    return N_GAS * len(CONDITIONS) * N_EVAL_SEEDS


def _eval_seeds_to_run(
    ga_seed: int, condition: str,
    completed: set[tuple[int, str, int]],
) -> list[int]:
    """Eval_seeds still to run for (ga_seed, condition)."""
    todo: list[int] = []
    for i in range(N_EVAL_SEEDS):
        eval_seed = EVAL_SEED_BASE + i * EVAL_SEED_STRIDE
        if (ga_seed, condition, eval_seed) not in completed:
            todo.append(eval_seed)
    return todo


# ----- Phase 1: capture genomes via re-running Block 11's GAs --------------


def _capture_one_genome(ga_seed: int, condition: str) -> CPPNGenome:
    """Re-run one GA and return its evolved best genome.

    Identical setup to Block 11's _run_one_ga; returns the genome
    rather than just the fitness. Bit-exact to Block 11's run since
    s11._evolve is deterministic on (seed, scenario, identity, ...).
    """
    scenario = s11._build_scenario()  # pylint: disable=protected-access
    a_is_inh = assign_ei_identity(s11.N_NEURONS, s11.INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(s11.N_NEURONS, s11.INHIBITORY_FRACTION)
    closed_loop = condition == "closed_loop"
    label = f"ga{ga_seed}/{condition}"
    # Block 11 used step 11's defaults (pop=32, gens=30); Block 11b
    # MUST match exactly so re-running produces bit-identical evolved
    # genomes. Hardcoded here rather than reading s11.run.__defaults__
    # to keep the contract explicit (and to silence mypy's
    # "tuple[Any, ...] | None" complaint about the defaults tuple).
    genome, _ = s11._evolve(  # pylint: disable=protected-access
        label, scenario, a_is_inh, b_is_inh,
        closed_loop=closed_loop,
        pop_size=32,
        n_generations=30,
        seed=ga_seed,
    )
    return genome


def _save_genomes(
    pkl_path: Path,
    genomes: dict[tuple[int, str], CPPNGenome],
) -> None:
    """Save dict of evolved genomes to a pickle file.

    JAX arrays inside CPPNGenome serialize via JAX's pickle support
    (they convert to numpy arrays under the hood); load round-trips
    them back to JAX arrays.
    """
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(genomes, f)


def _load_genomes(
    pkl_path: Path,
) -> dict[tuple[int, str], CPPNGenome]:
    """Load saved genome dict."""
    with pkl_path.open("rb") as f:
        loaded: dict[tuple[int, str], CPPNGenome] = pickle.load(f)
    return loaded


def _capture_all_genomes() -> dict[tuple[int, str], CPPNGenome]:
    """Phase 1: re-run all 40 GAs, return {(ga_seed, condition): genome}."""
    genomes: dict[tuple[int, str], CPPNGenome] = {}
    pairs = _genome_pairs()
    _log(f"Phase 1: capturing {len(pairs)} evolved genomes")
    for i, (ga_seed, condition) in enumerate(pairs):
        t0 = time.monotonic()
        genome = _capture_one_genome(ga_seed, condition)
        wall = time.monotonic() - t0
        genomes[(ga_seed, condition)] = genome
        _log(
            f"  Phase 1 [{i + 1}/{len(pairs)}] ga_seed={ga_seed} "
            f"{condition} captured ({wall:.1f}s)"
        )
    return genomes


# ----- Phase 2: structural statistics --------------------------------------


def _genome_stats(genome: CPPNGenome) -> dict[str, float]:
    """Decode genome to a SlotPool and return structural statistics.

    Computed across all (post, slot) entries:
      cross_pct: fraction of slots pointing into the partner space
        ([N, 2N) of the combined raster).
      cross_e_pct / cross_i_pct: fraction pointing to partner E vs I
        neurons. Uses ``assign_ei_identity``'s convention: I neurons
        are the LAST inhibitory_fraction*N indices.
      recurrent_pct: fraction pointing into own [0, N) raster.
      v_mean / v_std / v_max: weight magnitudes.
      v_mean_cross_e / v_mean_recurrent: weight means restricted to
        cross-E / recurrent slot subsets, surfaces "is the GA tuning
        cross-E weights specifically?"
    """
    pool = decode_cppn_to_pool(
        genome, s11.N_NEURONS, 2 * s11.N_NEURONS, s11.K_SLOTS, s11.V_MAX,
    )
    n_total = s11.N_NEURONS * s11.K_SLOTS
    is_cross = pool.pre_ids >= s11.N_NEURONS
    is_recurrent = ~is_cross
    n_cross = int(jnp.sum(is_cross))
    n_recurrent = int(jnp.sum(is_recurrent))
    # Partner E mask: pre_ids in [N, 2N) point to partner indices
    # (pre_ids - N); the partner's I neurons sit at the END of [0, N)
    # per assign_ei_identity (last inhibitory_fraction*N indices).
    n_partner = s11.N_NEURONS
    n_partner_inhib = int(n_partner * s11.INHIBITORY_FRACTION)
    partner_idx = jnp.where(is_cross, pool.pre_ids - n_partner, 0)
    is_cross_i = is_cross & (partner_idx >= (n_partner - n_partner_inhib))
    is_cross_e = is_cross & ~is_cross_i
    n_cross_e = int(jnp.sum(is_cross_e))
    n_cross_i = int(jnp.sum(is_cross_i))
    v = pool.v
    v_cross_e = jnp.where(is_cross_e, v, jnp.float32(0.0))
    v_recurrent = jnp.where(is_recurrent, v, jnp.float32(0.0))
    return {
        "cross_pct": 100.0 * n_cross / n_total,
        "cross_e_pct": 100.0 * n_cross_e / n_total,
        "cross_i_pct": 100.0 * n_cross_i / n_total,
        "recurrent_pct": 100.0 * n_recurrent / n_total,
        "v_mean": float(jnp.mean(v)),
        "v_std": float(jnp.std(v)),
        "v_max": float(jnp.max(v)),
        "v_mean_cross_e": (
            float(jnp.sum(v_cross_e) / max(n_cross_e, 1))
        ),
        "v_mean_recurrent": (
            float(jnp.sum(v_recurrent) / max(n_recurrent, 1))
        ),
    }


def _append_stats_row(
    csv_path: Path, ga_seed: int, condition: str,
    best_fitness: float, stats: dict[str, float],
) -> None:
    """Append per-genome stats row, writing header on first/empty file."""
    needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(
                ["ga_seed", "condition", "best_fitness"]
                + list(STATS_KEYS),
            )
        writer.writerow(
            [ga_seed, condition, f"{best_fitness:.6e}"]
            + [f"{stats[k]:.6f}" for k in STATS_KEYS],
        )


def _read_block11_fitness() -> dict[tuple[int, str], float]:
    """Read Block 11's fitnesses for cross-reference in the stats CSV."""
    block11_csv = Path("overnight_results/block11_cppn_n20.csv")
    if not block11_csv.exists():
        return {}
    out: dict[tuple[int, str], float] = {}
    with block11_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[(int(row["ga_seed"]), row["condition"])] = float(
                row["best_fitness"]
            )
    return out


# ----- Phase 3: evaluate at varying scenarios ------------------------------


def _build_scenario_for_eval_seed(
    eval_seed: int,
) -> tuple[Any, ...]:
    """Build a step-11-shaped scenario whose pool_a uses PRNGKey(eval_seed).

    Mirror of step 11's _build_scenario except the pool_a seed is
    parameterized. Pool_a is the only seed-dependent piece; drives,
    valences, and adrenaline_a are deterministic across eval_seeds.
    """
    seg_len = s11.N_TIMESTEPS // len(s11.A_DRIVE_PROFILE)
    i_ext_a = jnp.concatenate(
        [
            jnp.full(
                (seg_len, s11.N_NEURONS), level, dtype=jnp.float32,
            )
            for level in s11.A_DRIVE_PROFILE
        ]
    )
    i_ext_b = jnp.full(
        (s11.N_TIMESTEPS, s11.N_NEURONS),
        s11.B_BASELINE_DRIVE_MV,
        dtype=jnp.float32,
    )
    valence = jnp.zeros((s11.N_TIMESTEPS,), dtype=jnp.float32)
    adrenaline_a = jnp.ones((s11.N_TIMESTEPS,), dtype=jnp.float32)
    pool_a = make_pool_for_partner(
        s11.N_NEURONS, s11.K_SLOTS, jax.random.PRNGKey(eval_seed),
    )
    return (pool_a, i_ext_a, i_ext_b, valence, valence, adrenaline_a)


def _evaluate_genome_at(
    genome: CPPNGenome, condition: str, eval_seed: int,
) -> tuple[float, float]:
    """Evaluate one (genome, eval_seed) under the given condition.

    Returns (fitness, wall_sec). Delegates the actual sim to
    s11._evaluate_one for cross-step bit-exact consistency.
    """
    scenario = _build_scenario_for_eval_seed(eval_seed)
    a_is_inh = assign_ei_identity(s11.N_NEURONS, s11.INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(s11.N_NEURONS, s11.INHIBITORY_FRACTION)
    closed_loop = condition == "closed_loop"
    t0 = time.monotonic()
    fitness = s11._evaluate_one(  # pylint: disable=protected-access
        genome, scenario, a_is_inh, b_is_inh, closed_loop,
    )
    fitness_value = float(fitness)
    return fitness_value, time.monotonic() - t0


def _completed_evals(
    csv_path: Path,
) -> set[tuple[int, str, int]]:
    """Read evals CSV, return set of (ga_seed, condition, eval_seed) done."""
    if not csv_path.exists():
        return set()
    completed: set[tuple[int, str, int]] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((
                int(row["ga_seed"]),
                row["condition"],
                int(row["eval_seed"]),
            ))
    return completed


def _append_eval_row(
    csv_path: Path, ga_seed: int, condition: str,
    eval_seed: int, fitness: float, wall_sec: float,
) -> None:
    """Append a single eval row, writing header on first/empty file."""
    needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow([
                "ga_seed", "condition", "eval_seed", "fitness",
                "wall_sec",
            ])
        writer.writerow([
            ga_seed, condition, eval_seed,
            f"{fitness:.6e}", f"{wall_sec:.2f}",
        ])


# ----- Main orchestration --------------------------------------------------


def main() -> None:
    """Drive Phase 1 -> Phase 2 -> Phase 3 with per-phase resume."""
    GENOMES_PKL.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: capture or load genomes ---------------------------------
    if GENOMES_PKL.exists():
        _log(f"Phase 1: loading saved genomes from {GENOMES_PKL}")
        genomes = _load_genomes(GENOMES_PKL)
        assert len(genomes) == N_GAS * len(CONDITIONS), (
            f"saved pickle has {len(genomes)} entries, expected "
            f"{N_GAS * len(CONDITIONS)}"
        )
    else:
        genomes = _capture_all_genomes()
        _save_genomes(GENOMES_PKL, genomes)
        _log(f"Phase 1: saved {len(genomes)} genomes to {GENOMES_PKL}")

    # ---- Phase 2: structural stats ----------------------------------------
    _log(f"Phase 2: writing structural stats to {STATS_CSV}")
    block11_fits = _read_block11_fitness()
    # Always rewrite stats CSV (instant; ensures consistency with current
    # _genome_stats schema).
    if STATS_CSV.exists():
        STATS_CSV.unlink()
    for ga_seed, condition in _genome_pairs():
        genome = genomes[(ga_seed, condition)]
        stats = _genome_stats(genome)
        # Use Block 11's recorded fitness for cross-reference; fall back
        # to NaN if Block 11's CSV is missing (shouldn't happen in normal
        # workflow but supports running Block 11b standalone).
        best_fit = block11_fits.get((ga_seed, condition), float("nan"))
        _append_stats_row(STATS_CSV, ga_seed, condition, best_fit, stats)
    _log(f"Phase 2: wrote {N_GAS * len(CONDITIONS)} stats rows")

    # ---- Phase 3: evaluate at 100 eval_seeds per genome -------------------
    completed = _completed_evals(EVALS_CSV)
    expected = _expected_total_evals()
    _log(
        f"Phase 3: evaluating each genome at {N_EVAL_SEEDS} eval_seeds; "
        f"already complete: {len(completed)} / {expected}"
    )
    overall_start = time.monotonic()
    consecutive_failures = 0
    for ga_seed, condition in _genome_pairs():
        genome = genomes[(ga_seed, condition)]
        eval_seeds = _eval_seeds_to_run(ga_seed, condition, completed)
        if not eval_seeds:
            continue
        _log(
            f"  Phase 3 ga_seed={ga_seed} {condition}: "
            f"{len(eval_seeds)} eval_seeds remaining"
        )
        for eval_seed in eval_seeds:
            try:
                fitness, wall = _evaluate_genome_at(
                    genome, condition, eval_seed,
                )
                consecutive_failures = 0
            except (RuntimeError, MemoryError) as exc:
                _log(
                    f"  FAILED ga_seed={ga_seed} {condition} "
                    f"eval_seed={eval_seed}: "
                    f"{type(exc).__name__}: {exc}"
                )
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    _log(
                        f"ABORTING: {consecutive_failures} consecutive "
                        f"failures - configuration appears broken"
                    )
                    raise
                continue
            _append_eval_row(
                EVALS_CSV, ga_seed, condition, eval_seed, fitness, wall,
            )
    total_wall = time.monotonic() - overall_start
    _log(f"Phase 3 complete: total wall {total_wall:.1f}s")

    # PlasticNetState / PairedState / init_state / init_traces are
    # imported above for typing parity with step 11; no Phase 3 use.
    _ = PlasticNetState, PairedState, init_state, init_traces, SlotPool


if __name__ == "__main__":
    main()
