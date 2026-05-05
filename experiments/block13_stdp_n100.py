"""Block 13: step-16 STDP +14% headline reanchored at N=100.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Per the revalidation plan: ``step 16 STDP +14%, N=100 (partially
already confirmed via blocks 4/5 at rate=1.0/ivm=1.0)``.

The "step 16 STDP at rate=1.0 produces a +14% post-training fitness
improvement" claim from the original README dev log was measured at
N=5 (overnight_batch's block 4) and reconfirmed at N=20 in
overnight_batch's block 5 (the `baseline_stdp_only` config). Block
13 puts SEM on the +14% number at N=100 to nail down the headline.

Sweep:
    N=100 seeds (stride 37) at the headline config:
      plasticity_rate = 1.0
      init_v_mean     = 1.0    (matches step 16's INIT_V_MEAN default)
      init_v_std      = 0.3    (matches step 16's INIT_V_STD default)
    Single condition: closed-loop adrenaline gain=50, D008
    i_mult=8.0, T_train=20000, T_measure=2000.

Total runs: 100 (one per seed). Each runs step 16's three-phase
protocol (fit_before, train, fit_after). Per-run wall ~2-3 s on
the GTX 1050 Mobile -> total ~3-5 min.

Per-row metrics:
    fit_before     pre-training fitness, plasticity-frozen sim
    fit_after      post-training fitness, plasticity-frozen sim
    improvement_pct  100 * (fit_after - fit_before) / |fit_before|
    train_time     wall time for the 20k-step training scan
    wall_sec       total wall for the (fit_before, train, fit_after) trio

Block 13 imports `_step16_once` from `overnight_batch` to reuse the
exact same Phase A/B/C orchestration that produced the original
+14% claim - guarantees bit-exact comparability with blocks 4/5 at
the overlapping seeds.

Outputs:
    overnight_results/block13_stdp_n100.csv
    overnight_results/block13_stdp_n100.log

Resumable on `seed`. Standard MAX_CONSECUTIVE_FAILURES=5 abort.

Usage:
    .venv/bin/python experiments/block13_stdp_n100.py
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

# pylint: disable=import-error
import overnight_batch as ob
import step16_stdp_learning as s16


N_SEEDS: int = 100
SEED_BASE: int = 0
SEED_STRIDE: int = 37
PLASTICITY_RATE: float = 1.0
INIT_V_MEAN: float = s16.INIT_V_MEAN  # = 1.0
INIT_V_STD: float = s16.INIT_V_STD    # = 0.3

CSV_PATH: Path = Path("overnight_results/block13_stdp_n100.csv")
LOG_PATH: Path = Path("overnight_results/block13_stdp_n100.log")

MAX_CONSECUTIVE_FAILURES: int = 5


def _completed_seeds(csv_path: Path) -> set[int]:
    """Read CSV, return set of seeds already done."""
    if not csv_path.exists():
        return set()
    completed: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(int(row["seed"]))
    return completed


def _append_row(
    csv_path: Path,
    seed: int,
    fit_before: float,
    fit_after: float,
    improvement_pct: float,
    train_time: float,
    wall_sec: float,
) -> None:
    """Append one result row, writing header on first/empty file."""
    needs_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow([
                "seed", "fit_before", "fit_after", "improvement_pct",
                "train_time", "wall_sec",
            ])
        writer.writerow([
            seed,
            f"{fit_before:.6e}",
            f"{fit_after:.6e}",
            f"{improvement_pct:.2f}",
            f"{train_time:.2f}",
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
    """Total seeds in the full sweep grid."""
    return N_SEEDS


def _seeds_to_run(completed: set[int]) -> list[int]:
    """Seeds still to run, in deterministic order."""
    return [
        SEED_BASE + i * SEED_STRIDE
        for i in range(N_SEEDS)
        if (SEED_BASE + i * SEED_STRIDE) not in completed
    ]


def _improvement_pct(fit_before: float, fit_after: float) -> float:
    """Improvement percent: 100 * (fit_after - fit_before) / |fit_before|.

    Both fitnesses are negative (negative MSE); improvement means
    fit_after is *closer to zero* than fit_before. A positive
    improvement_pct means STDP made things better.
    """
    return 100.0 * (fit_after - fit_before) / abs(fit_before)


def _run_one(seed: int) -> tuple[dict[str, float], float]:
    """Run step 16's three-phase protocol for one seed.

    Delegates to `overnight_batch._step16_once` for the actual sim;
    Block 13 owns only the orchestration. Returns (metrics_dict,
    wall_sec).
    """
    t0 = time.monotonic()
    metrics = ob._step16_once(  # pylint: disable=protected-access
        seed, PLASTICITY_RATE, INIT_V_MEAN, INIT_V_STD,
    )
    wall = time.monotonic() - t0
    # Coerce explicitly so downstream f-string formatting doesn't
    # choke on JAX scalars in any future overnight_batch refactor.
    return (
        {
            "fit_before": float(metrics["fit_before"]),
            "fit_after": float(metrics["fit_after"]),
            "train_time": float(metrics["train_time"]),
        },
        wall,
    )


def main() -> None:
    """Drive the N=100 step-16 STDP sweep, resumable from a partial CSV."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_seeds(CSV_PATH)
    expected = _expected_total()
    _log(
        f"block 13 step-16 STDP sweep: rate={PLASTICITY_RATE}, "
        f"init_v={INIT_V_MEAN}+/-{INIT_V_STD}, N={N_SEEDS} seeds"
    )
    _log(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    _log(f"already complete: {len(completed)} / {expected}")
    overall_start = time.monotonic()
    todo = _seeds_to_run(completed)
    consecutive_failures = 0
    failed_seeds: list[int] = []
    for seed in todo:
        try:
            metrics, wall = _run_one(seed)
            consecutive_failures = 0
        except (RuntimeError, MemoryError) as exc:
            _log(
                f"FAILED seed={seed}: {type(exc).__name__}: {exc}"
            )
            failed_seeds.append(seed)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                _log(
                    f"ABORTING: {consecutive_failures} consecutive "
                    f"failures - configuration appears broken; "
                    f"raising last exception"
                )
                raise
            continue
        improvement = _improvement_pct(
            metrics["fit_before"], metrics["fit_after"],
        )
        _append_row(
            CSV_PATH, seed,
            metrics["fit_before"], metrics["fit_after"],
            improvement,
            metrics["train_time"], wall,
        )
        fb = metrics["fit_before"]
        fa = metrics["fit_after"]
        _log(
            f"  done seed={seed} fit_before={fb:.3e} "
            f"fit_after={fa:.3e} improvement={improvement:+.2f}% "
            f"wall={wall:.1f}s"
        )
    total_wall = time.monotonic() - overall_start
    if failed_seeds:
        _log(
            f"block 13 finished with {len(failed_seeds)} failed seeds: "
            f"{failed_seeds}"
        )
    _log(f"block 13 complete: total wall {total_wall:.1f}s")


if __name__ == "__main__":
    main()
