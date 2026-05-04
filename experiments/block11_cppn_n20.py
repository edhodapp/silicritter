"""Block 11: N=20 independent CPPN GAs at step 11's E/I + closed-loop setup.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Per the revalidation plan: ``step 11 CPPN GA, N=20 independent GAs``
- runs step 11's GA 20 times with different starting populations
under both open-loop and closed-loop conditions, producing a
distribution of "GA's best fitness" datapoints that can be
statistically compared to step 10's hand-wired closed-loop
headline (-2.7284e-05 at gain=200, T=10M, N=500 from Block 9).

Block 11 *does not* include the "× N=100 eval seeds" leg from the
plan. That phase would re-evaluate each evolved best genome on 100
varying scenarios; deferred as Block 11b. With Block 11 alone we
get N=20 best-fitness samples per condition for the distribution
comparison.

**Selection-on-noise caveat (load-bearing for interpretation):** the
GA trains on a single deterministic scenario (pool_a uses
PRNGKey(777) inside step 11's _build_scenario) and selects the best
individual at the end of training. The "best" fitness reported is
therefore an *upward-biased* estimator vs the hand-wired comparison
which is not subject to the same selection-on-noise. Concretely:

- If Block 11 reports closed_loop best_fit *worse* than Block 9's
  -2.7284e-05 hand-wired headline, the conclusion is robust ("the
  GA cannot beat hand-wired even with the bias in its favor").
- If Block 11 reports closed_loop best_fit *better* than -2.7284e-05,
  hold the conclusion lightly until Block 11b's eval-seed re-eval
  shows the evolved genomes generalize to unseen scenarios.

The "× N=100 eval seeds" phase exists in the plan precisely to
break this asymmetry. Don't claim a "GA wins" finding from Block
11 alone.

Sweep:
    20 GA seeds × 2 conditions = 40 independent GAs
    each at step 11's defaults: pop=32, gens=30, T=2000
    each GA does 30 × 32 = 960 fitness evaluations (vmap-batched)

Wall-time estimate on the GTX 1050 Mobile: ~40-60 min for the full
sweep. Assumes step 11's _evolve eval cost is ~1-1.5 s per
generation (vmap=32 over T=2000 sims) plus ~30 s JIT compile per
GA.

Outputs:
    overnight_results/block11_cppn_n20.csv
        ga_seed,condition,best_fitness,wall_sec
    overnight_results/block11_cppn_n20.log

Resumable via _completed_pairs on (ga_seed, condition). The script
re-launches step 11's _evolve for any unfinished pair and skips
already-recorded rows.

Why not save the evolved genomes for later analysis: keeps Block
11 simple and headline-focused. If a future Block 11b wants to
re-evaluate or analyze the evolved topologies, re-run the GA at
the same ga_seed - reproducibility is bit-exact since step 11's
_evolve uses ``jax.random.PRNGKey(seed + 200)`` deterministically.

Usage:
    .venv/bin/python experiments/block11_cppn_n20.py
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

# Block 11 reuses step 11's GA implementation directly. step 11 lives
# in the same experiments/ directory and is on the path via
# pyproject.toml's pytest pythonpath setting; outside pytest we
# rely on Python's normal cwd-based import resolution (the script
# is launched from the repo root).
# pylint: disable=import-error
import step11_cppn_closedloop as s11
from silicritter.slotpool import assign_ei_identity


N_GAS: int = 20
SEED_BASE: int = 0
# Step 11's _evolve uses PRNGKey(seed + 200), so any non-overlapping
# integer range produces independent populations. Stride 1 is fine
# here because step 11's offset prevents seed=0 collisions with
# anything else in the codebase using PRNGKey(0).
SEED_STRIDE: int = 1
CONDITIONS: tuple[str, ...] = ("open_loop", "closed_loop")
POP_SIZE: int = 32
N_GENERATIONS: int = 30

CSV_PATH: Path = Path("overnight_results/block11_cppn_n20.csv")
LOG_PATH: Path = Path("overnight_results/block11_cppn_n20.log")

MAX_CONSECUTIVE_FAILURES: int = 5


def _completed_pairs(csv_path: Path) -> set[tuple[int, str]]:
    """Read CSV, return set of (ga_seed, condition) already done."""
    if not csv_path.exists():
        return set()
    completed: set[tuple[int, str]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((int(row["ga_seed"]), row["condition"]))
    return completed


def _append_row(
    csv_path: Path, ga_seed: int, condition: str,
    best_fitness: float, wall_sec: float,
) -> None:
    """Append a single result row, writing header on first write.

    Header detection uses ``stat().st_size == 0`` (not just ``not
    exists()``) so a 0-byte file from a prior crashed run still
    gets a header on the next append (matches Block 9/10's G10-iii
    fix; resume on a corrupted prior run is safe).
    """
    needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(
                ["ga_seed", "condition", "best_fitness", "wall_sec"],
            )
        writer.writerow([
            ga_seed, condition,
            f"{best_fitness:.6e}", f"{wall_sec:.2f}",
        ])


def _log(message: str) -> None:
    """Append timestamped line to LOG_PATH and stdout."""
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _expected_total() -> int:
    """Total (ga_seed, condition) pairs in the full sweep grid."""
    return N_GAS * len(CONDITIONS)


def _gas_to_run(
    completed: set[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Pairs still to run, in deterministic (ga_seed, condition) order."""
    todo: list[tuple[int, str]] = []
    for i in range(N_GAS):
        ga_seed = SEED_BASE + i * SEED_STRIDE
        for condition in CONDITIONS:
            if (ga_seed, condition) not in completed:
                todo.append((ga_seed, condition))
    return todo


def _run_one_ga(ga_seed: int, condition: str) -> tuple[float, float]:
    """Run one GA at the given seed/condition; return (best_fit, wall_sec).

    Delegates to step 11's _evolve, passing through pop_size /
    n_generations / seed unchanged. Block 11 owns only the
    orchestration; the GA itself is step 11's load-bearing code.
    """
    scenario = s11._build_scenario()  # pylint: disable=protected-access
    a_is_inh = assign_ei_identity(s11.N_NEURONS, s11.INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(s11.N_NEURONS, s11.INHIBITORY_FRACTION)
    closed_loop = condition == "closed_loop"
    label = f"ga{ga_seed}/{condition}"
    t0 = time.time()
    _, best_fit = s11._evolve(  # pylint: disable=protected-access
        label, scenario, a_is_inh, b_is_inh,
        closed_loop=closed_loop,
        pop_size=POP_SIZE, n_generations=N_GENERATIONS, seed=ga_seed,
    )
    # Coerce explicitly: s11._evolve returns a Python float in the
    # current code path, but a JAX scalar would silently break
    # _append_row's f"{...:.6e}" formatting on some JAX versions.
    return float(best_fit), time.time() - t0


def main() -> None:
    """Drive the N=20 GA sweep, resumable from a partial CSV."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_pairs(CSV_PATH)
    expected = _expected_total()
    _log(
        f"block 11 CPPN-GA sweep: N={N_GAS} GAs x {len(CONDITIONS)} "
        f"conditions, pop={POP_SIZE}, gens={N_GENERATIONS}"
    )
    _log(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    _log(f"already complete: {len(completed)} / {expected}")
    overall_start = time.time()
    todo = _gas_to_run(completed)
    consecutive_failures = 0
    failed_pairs: list[tuple[int, str]] = []
    for ga_seed, condition in todo:
        _log(f"start ga_seed={ga_seed} {condition}")
        try:
            best_fit, wall = _run_one_ga(ga_seed, condition)
            consecutive_failures = 0
        except (RuntimeError, MemoryError) as exc:
            _log(
                f"FAILED ga_seed={ga_seed} {condition}: "
                f"{type(exc).__name__}: {exc}"
            )
            failed_pairs.append((ga_seed, condition))
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                _log(
                    f"ABORTING: {consecutive_failures} consecutive "
                    f"failures - configuration appears broken; "
                    f"raising last exception"
                )
                raise
            continue
        _append_row(CSV_PATH, ga_seed, condition, best_fit, wall)
        _log(
            f"  done ga_seed={ga_seed} {condition} "
            f"best_fit={best_fit:.3e} wall={wall:.1f}s"
        )
    total_wall = time.time() - overall_start
    if failed_pairs:
        _log(
            f"block 11 finished with {len(failed_pairs)} failed pairs: "
            f"{failed_pairs} - re-run to retry (resume will skip "
            f"completed rows)"
        )
    _log(f"block 11 complete: total wall {total_wall:.1f}s")


if __name__ == "__main__":
    main()
