"""Block 12: N=100 fGn Hurst sweep at H in {0.3, 0.5, 0.7, 0.9}.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Per the revalidation plan: ``Block 12: step 14 fGn H ∈
{0.3,0.5,0.7,0.9}, N=100 per H``. Multi-seed reanchor of step 14's
single-seed Hurst-sweep finding (closed-loop tracking is
H-dependent; H=0.5 is the noise minimum, H=0.7 stumbles, H=0.9
shows a prediction gap). Step 14's N=1 numbers per cell were
informative but variance-unbounded; Block 12 puts SEM on each
(H, condition) cell.

Sweep grid:
    Hurst H in {0.3, 0.5, 0.7, 0.9}  (matches step 14's HURST_GRID)
    N=100 seeds per H                (stride 37; aligns with prior blocks)
    conditions: open_loop, closed_loop (gain=50, D008's i_mult=8.0)

Total runs: 4 * 100 * 2 = 800. At step 14's T=2000 and per-run
wall ~0.3-0.5 s on the GTX 1050, expected total wall ~5-10 min.

Per-run metrics (5 floats from step 14's _run_condition):
    track_fit       -mean((b_rate - a_rate)^2)        (B tracks A now)
    pred_fit        -mean((b[:-1] - a[1:])^2)         (B predicts A's next)
    stim_lag1       Empirical lag-1 autocorr of A's per-window drive
    a_rate_mean     Mean A firing rate (Hz)
    b_rate_mean     Mean B firing rate (Hz)

Outputs:
    overnight_results/block12_fgn_n100.csv
        seed,hurst,condition,track_fit,pred_fit,stim_lag1,
        a_rate_mean,b_rate_mean,wall_sec
    overnight_results/block12_fgn_n100.log

Resumable on (seed, hurst, condition) tuple. MAX_CONSECUTIVE_FAILURES=5
abort pattern matches Blocks 9/10/11.

Cross-file coupling: imports step 14's _run_condition directly.
A regression test pins its (hurst, closed_loop, seed) signature so
step 14 refactor breaks Block 12 at gate time, not at multi-hour
run time.

Usage:
    .venv/bin/python experiments/block12_fgn_n100.py
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

# pylint: disable=import-error
import step14_fgn_stimulus as s14


N_SEEDS: int = 100
SEED_BASE: int = 0
SEED_STRIDE: int = 37
HURST_GRID: tuple[float, ...] = s14.HURST_GRID
CONDITIONS: tuple[str, ...] = ("open_loop", "closed_loop")

CSV_PATH: Path = Path("overnight_results/block12_fgn_n100.csv")
LOG_PATH: Path = Path("overnight_results/block12_fgn_n100.log")

MAX_CONSECUTIVE_FAILURES: int = 5


def _completed_pairs(
    csv_path: Path,
) -> set[tuple[int, float, str]]:
    """Read CSV, return set of (seed, hurst, condition) already done."""
    if not csv_path.exists():
        return set()
    completed: set[tuple[int, float, str]] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((
                int(row["seed"]),
                float(row["hurst"]),
                row["condition"],
            ))
    return completed


def _append_row(
    csv_path: Path,
    seed: int, hurst: float, condition: str,
    track_fit: float, pred_fit: float, stim_lag1: float,
    a_rate_mean: float, b_rate_mean: float,
    wall_sec: float,
) -> None:
    """Append one result row, writing header on first/empty file."""
    needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow([
                "seed", "hurst", "condition",
                "track_fit", "pred_fit", "stim_lag1",
                "a_rate_mean", "b_rate_mean", "wall_sec",
            ])
        writer.writerow([
            seed, hurst, condition,
            f"{track_fit:.6e}", f"{pred_fit:.6e}",
            f"{stim_lag1:.6f}",
            f"{a_rate_mean:.4f}", f"{b_rate_mean:.4f}",
            f"{wall_sec:.2f}",
        ])


def _log(message: str) -> None:
    """Append timestamped line to LOG_PATH and stdout."""
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    line = f"[{ts}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _expected_total() -> int:
    """Total (seed, hurst, condition) tuples in the full sweep grid."""
    return len(HURST_GRID) * N_SEEDS * len(CONDITIONS)


def _conditions_to_run(
    seed: int, hurst: float,
    completed: set[tuple[int, float, str]],
) -> list[str]:
    """Conditions still to run for (seed, hurst) given completed."""
    return [
        c for c in CONDITIONS
        if (seed, hurst, c) not in completed
    ]


def _run_one(
    seed: int, hurst: float, condition: str,
) -> tuple[tuple[float, float, float, float, float], float]:
    """Run one (seed, hurst, condition) cell.

    Delegates to step 14's _run_condition for the actual sim;
    Block 12 owns only the orchestration. Returns (metrics_5tuple,
    wall_sec) where metrics is (track_fit, pred_fit, stim_lag1,
    a_rate_mean, b_rate_mean).

    Floats are coerced to Python float (s14._run_condition already
    returns floats but the explicit coercion guards against future
    JAX-array drift through _append_row's "{:.6e}" formatter).
    """
    closed_loop = condition == "closed_loop"
    t0 = time.monotonic()
    metrics = s14._run_condition(  # pylint: disable=protected-access
        hurst, closed_loop, seed,
    )
    wall = time.monotonic() - t0
    track, pred, lag1, a_mean, b_mean = metrics
    return (
        (float(track), float(pred), float(lag1),
         float(a_mean), float(b_mean)),
        wall,
    )


def main() -> None:
    """Drive the N=100 fGn Hurst sweep, resumable from a partial CSV."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_pairs(CSV_PATH)
    expected = _expected_total()
    _log(
        f"block 12 fGn N=100 sweep: H={list(HURST_GRID)}, "
        f"N={N_SEEDS} seeds, {len(CONDITIONS)} conditions per (seed, H)"
    )
    _log(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    _log(f"already complete: {len(completed)} / {expected}")
    overall_start = time.monotonic()
    consecutive_failures = 0
    failed_pairs: list[tuple[int, float, str]] = []
    for hurst in HURST_GRID:
        for i in range(N_SEEDS):
            seed = SEED_BASE + i * SEED_STRIDE
            todo = _conditions_to_run(seed, hurst, completed)
            if not todo:
                continue
            for condition in todo:
                try:
                    metrics, wall = _run_one(seed, hurst, condition)
                    consecutive_failures = 0
                except (RuntimeError, MemoryError) as exc:
                    _log(
                        f"FAILED seed={seed} H={hurst} {condition}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    failed_pairs.append((seed, hurst, condition))
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        _log(
                            f"ABORTING: {consecutive_failures} "
                            f"consecutive failures - configuration "
                            f"appears broken; raising last exception"
                        )
                        raise
                    continue
                track, pred, lag1, a_mean, b_mean = metrics
                _append_row(
                    CSV_PATH, seed, hurst, condition,
                    track, pred, lag1, a_mean, b_mean, wall,
                )
                _log(
                    f"  done seed={seed} H={hurst} {condition} "
                    f"track={track:.3e} pred={pred:.3e} "
                    f"lag1={lag1:.3f} wall={wall:.1f}s"
                )
    total_wall = time.monotonic() - overall_start
    if failed_pairs:
        _log(
            f"block 12 finished with {len(failed_pairs)} failed pairs"
        )
    _log(f"block 12 complete: total wall {total_wall:.1f}s")


if __name__ == "__main__":
    main()
