"""Overnight research batch: ~5-6 hours of autonomous experiments.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Runs a sequence of blocks, each writing per-run CSV rows to
`overnight_results/`. Every block is resumable: if a row for a
(config_name, seed) already exists, it's skipped. Progress and
errors go to `overnight_results/batch.log`.

Blocks (in order, earlier = higher priority):
    1. Step 17 factorial multi-seed       -- structural-growth sweep
    2. Acquisition-probability log-sweep  -- how much rebinding helps
    3. Long-training probe                -- does structural
                                             plasticity's advantage
                                             widen with more steps
    4. Step 16 multi-seed                 -- retrofit STDP-only at
                                             rate x init_v
    5. Best-config multi-seed confirmation-- tighten the headline
                                             number with 20 seeds

If a block fails, the exception is logged and the batch moves on
to the next block rather than aborting the whole night. Each block
commits its CSV on completion so partial results are safe.

Usage:
    .venv/bin/python experiments/overnight_batch.py
"""

from __future__ import annotations

import csv
import datetime as dt
import gc
import itertools
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import jax

sys.path.insert(0, str(Path(__file__).parent))

# pylint: disable=wrong-import-position
from step17_structural_growth import (  # noqa: E402
    Config as Step17Config,
    _run_config as step17_run_config,
)
from step16_stdp_learning import (  # noqa: E402
    N_NEURONS as S16_N,
    INHIBITORY_FRACTION as S16_IFRAC,
    N_MEASURE_STEPS as S16_MEASURE_STEPS,
    N_TRAIN_STEPS as S16_TRAIN_STEPS,
    _random_b_pool as s16_random_b_pool,
    _build_state as s16_build_state,
    _build_drives as s16_build_drives,
    _training_scan as s16_training_scan,
    _measure_fitness as s16_measure_fitness,
    _describe_pool as s16_describe_pool,
)
from silicritter.slotpool import assign_ei_identity  # noqa: E402


RESULTS_DIR = Path(__file__).parent.parent / "overnight_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_LOG = RESULTS_DIR / "batch.log"


def log(msg: str) -> None:
    """Append to batch log with UTC timestamp; also print."""
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with BATCH_LOG.open("a") as fh:
        fh.write(line + "\n")


# ----------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------


def _load_completed(csv_path: Path) -> set[tuple[str, int]]:
    """Return set of (config_name, seed) already present in the CSV."""
    if not csv_path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                done.add((row["config"], int(row["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append a single row to a CSV, writing header if new."""
    need_header = not csv_path.exists()
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def _git_commit_results(block_name: str) -> None:
    """Best-effort git add + commit + push for the results directory."""
    try:
        subprocess.run(
            ["git", "add", str(RESULTS_DIR)],
            check=True, capture_output=True,
        )
        msg = (
            f"Overnight batch: {block_name} complete\n\n"
            f"Auto-commit from experiments/overnight_batch.py.\n\n"
            f"Co-Authored-By: Claude Opus 4.7 (1M context) "
            f"<noreply@anthropic.com>"
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            check=True, capture_output=True,
        )
        push_result = subprocess.run(
            ["git", "push", "origin", "main"],
            check=False, capture_output=True,
        )
        if push_result.returncode == 0:
            log(f"[commit] {block_name} committed and pushed")
        else:
            stderr = push_result.stderr.decode().strip()[:300]
            log(
                f"[commit] {block_name} committed but push FAILED: "
                f"{stderr}"
            )
    except subprocess.CalledProcessError as exc:
        log(
            f"[commit] {block_name} commit skipped "
            f"(likely nothing to commit): "
            f"{exc.stderr.decode().strip()[:200]}"
        )
    except Exception as exc:  # pylint: disable=broad-except
        log(f"[commit] {block_name} commit FAILED: {exc!r}")


# ----------------------------------------------------------------------
# Block 1: Step 17 factorial multi-seed
# ----------------------------------------------------------------------


def _step17_factorial_configs() -> list[Step17Config]:
    configs: list[Step17Config] = [
        Step17Config(
            "baseline_stdp_only", "off", "uniform",
            0.2, 0.0, 10_000_000,
        ),
        Step17Config(
            "baseline_release_only", "off", "uniform",
            0.2, 0.05, 500,
        ),
    ]
    for mode, source, init_v, thr, dur in itertools.product(
        ("stochastic", "periodic", "valence_gated", "valence_inverted"),
        ("uniform", "hebbian"),
        (0.2, 1.0, 1.8),
        (0.05, 0.2),
        (200, 1000),
    ):
        name = f"{mode}_{source}_iv{init_v:.1f}_thr{thr:.2f}_dur{dur}"
        configs.append(Step17Config(
            name, mode, source, init_v, thr, dur,
        ))
    return configs


def block1_step17_factorial(seeds: tuple[int, ...]) -> None:
    csv_path = RESULTS_DIR / "block1_step17_factorial.csv"
    done = _load_completed(csv_path)
    configs = _step17_factorial_configs()
    total = len(configs) * len(seeds)
    log(
        f"[block1] step17 factorial: "
        f"{len(configs)} configs x {len(seeds)} seeds = {total} runs"
    )
    idx = 0
    for config in configs:
        for seed in seeds:
            idx += 1
            key = (config.name, seed)
            if key in done:
                log(f"[block1] skip {idx}/{total} {config.name} seed={seed}")
                continue
            try:
                t0 = time.perf_counter()
                metrics = step17_run_config(config, seed)
                dt_run = time.perf_counter() - t0
                row = {
                    "block": "step17_factorial",
                    "config": config.name,
                    "seed": seed,
                    "acq_mode": config.acq_mode,
                    "pre_id_source": config.pre_id_source,
                    "acq_initial_v": config.acq_initial_v,
                    "release_threshold": config.release_threshold,
                    "release_duration": config.release_duration,
                    "fit_before": metrics["fit_before"],
                    "fit_after": metrics["fit_after"],
                    "delta": metrics["fit_after"] - metrics["fit_before"],
                    "v_mean": metrics["v_mean"],
                    "v_std": metrics["v_std"],
                    "active_frac_end": metrics["active_frac_end"],
                    "cross_e_frac_end": metrics["cross_e_frac_end"],
                    "active_frac_min": metrics["active_frac_min"],
                    "valence_mean": metrics["valence_mean"],
                    "wall_sec": dt_run,
                }
                _append_row(csv_path, row)
                b_fit = metrics["fit_before"]
                a_fit = metrics["fit_after"]
                log(
                    f"[block1] {idx}/{total} {config.name} "
                    f"seed={seed}: "
                    f"{b_fit:+.3e} -> {a_fit:+.3e} "
                    f"(delta {a_fit - b_fit:+.2e}, {dt_run:.1f}s)"
                )
            except Exception as exc:  # pylint: disable=broad-except
                log(
                    f"[block1] ERROR {config.name} seed={seed}: {exc!r}\n"
                    f"{traceback.format_exc()}"
                )
            gc.collect()
    _git_commit_results("block1_step17_factorial")


# ----------------------------------------------------------------------
# Block 2: Acquisition-probability log-sweep
# ----------------------------------------------------------------------


def _acq_prob_sweep_runner(
    acq_prob: float,
    seed: int,
) -> dict[str, float]:
    """Run step 17's _run_config with overridden acquisition probability.

    We monkey-patch step17's module-level ACQ_PROB_STOCHASTIC to vary
    the probability; easier than re-plumbing it as a config axis.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    import step17_structural_growth as s17
    orig = s17.ACQ_PROB_STOCHASTIC
    s17.ACQ_PROB_STOCHASTIC = acq_prob
    try:
        # Stochastic uniform, init_v=0.2, thr=0.05, dur=500 is a
        # reasonable default operating point.
        config = Step17Config(
            f"acq_prob_sweep_p{acq_prob:.0e}",
            "stochastic", "uniform", 0.2, 0.05, 500,
        )
        return s17._run_config(config, seed)
    finally:
        s17.ACQ_PROB_STOCHASTIC = orig


def block2_acquisition_probability(seeds: tuple[int, ...]) -> None:
    csv_path = RESULTS_DIR / "block2_acquisition_probability.csv"
    done = _load_completed(csv_path)
    probs = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    total = len(probs) * len(seeds)
    log(
        f"[block2] acq-prob sweep: {len(probs)} probs x {len(seeds)} seeds = "
        f"{total} runs"
    )
    idx = 0
    for prob in probs:
        config_name = f"acq_prob_{prob:.0e}"
        for seed in seeds:
            idx += 1
            key = (config_name, seed)
            if key in done:
                log(f"[block2] skip {idx}/{total} {config_name} seed={seed}")
                continue
            try:
                t0 = time.perf_counter()
                metrics = _acq_prob_sweep_runner(prob, seed)
                dt_run = time.perf_counter() - t0
                row = {
                    "block": "acq_probability",
                    "config": config_name,
                    "seed": seed,
                    "acq_prob": prob,
                    "fit_before": metrics["fit_before"],
                    "fit_after": metrics["fit_after"],
                    "delta": metrics["fit_after"] - metrics["fit_before"],
                    "v_mean": metrics["v_mean"],
                    "v_std": metrics["v_std"],
                    "active_frac_end": metrics["active_frac_end"],
                    "cross_e_frac_end": metrics["cross_e_frac_end"],
                    "wall_sec": dt_run,
                }
                _append_row(csv_path, row)
                delta = metrics["fit_after"] - metrics["fit_before"]
                cxe = metrics["cross_e_frac_end"]
                log(
                    f"[block2] {idx}/{total} p={prob:.0e} "
                    f"seed={seed}: delta={delta:+.2e}, "
                    f"cross_e={cxe:.2%}"
                )
            except Exception as exc:  # pylint: disable=broad-except
                log(f"[block2] ERROR prob={prob} seed={seed}: {exc!r}")
            gc.collect()
    _git_commit_results("block2_acquisition_probability")


# ----------------------------------------------------------------------
# Block 3: Long-training probe
# ----------------------------------------------------------------------


def _long_training_run(
    n_train: int, with_structural: bool, seed: int,
) -> dict[str, float]:
    """Run step16-style training for `n_train` steps, optionally with
    structural plasticity (via step 17's scan)."""
    # pylint: disable=import-outside-toplevel,protected-access
    import step16_stdp_learning as s16
    import step17_structural_growth as s17
    orig_s17_n = s17.N_TRAIN_STEPS
    orig_s16_n = s16.N_TRAIN_STEPS
    s17.N_TRAIN_STEPS = n_train
    s16.N_TRAIN_STEPS = n_train
    try:
        if with_structural:
            config = Step17Config(
                f"long_structural_n{n_train}",
                "stochastic", "uniform", 0.2, 0.05, 500,
            )
            return s17._run_config(config, seed)
        # STDP-only: use the step 17 baseline_stdp_only config.
        config = Step17Config(
            f"long_stdp_only_n{n_train}",
            "off", "uniform", 0.2, 0.0, 10_000_000,
        )
        return s17._run_config(config, seed)
    finally:
        s17.N_TRAIN_STEPS = orig_s17_n
        s16.N_TRAIN_STEPS = orig_s16_n


def block3_long_training(seeds: tuple[int, ...]) -> None:
    csv_path = RESULTS_DIR / "block3_long_training.csv"
    done = _load_completed(csv_path)
    durations = (20_000, 50_000, 100_000, 200_000, 500_000)
    conditions = (("structural", True), ("stdp_only", False))
    total = len(durations) * len(conditions) * len(seeds)
    log(
        f"[block3] long-training: "
        f"{len(durations)} durations x {len(conditions)} conds x "
        f"{len(seeds)} seeds = {total} runs"
    )
    idx = 0
    for n_train, (cond_name, with_struct) in itertools.product(
        durations, conditions,
    ):
        config_name = f"{cond_name}_n{n_train}"
        for seed in seeds:
            idx += 1
            key = (config_name, seed)
            if key in done:
                log(f"[block3] skip {idx}/{total} {config_name} seed={seed}")
                continue
            try:
                t0 = time.perf_counter()
                metrics = _long_training_run(n_train, with_struct, seed)
                dt_run = time.perf_counter() - t0
                row = {
                    "block": "long_training",
                    "config": config_name,
                    "seed": seed,
                    "n_train_steps": n_train,
                    "condition": cond_name,
                    "fit_before": metrics["fit_before"],
                    "fit_after": metrics["fit_after"],
                    "delta": metrics["fit_after"] - metrics["fit_before"],
                    "v_mean": metrics["v_mean"],
                    "cross_e_frac_end": metrics["cross_e_frac_end"],
                    "wall_sec": dt_run,
                }
                _append_row(csv_path, row)
                delta = metrics["fit_after"] - metrics["fit_before"]
                log(
                    f"[block3] {idx}/{total} {config_name} "
                    f"seed={seed}: delta={delta:+.2e} ({dt_run:.1f}s)"
                )
            except Exception as exc:  # pylint: disable=broad-except
                log(
                    f"[block3] ERROR {config_name} seed={seed}: {exc!r}\n"
                    f"{traceback.format_exc()}"
                )
            gc.collect()
    _git_commit_results("block3_long_training")


# ----------------------------------------------------------------------
# Block 4: Step 16 multi-seed retrofit
# ----------------------------------------------------------------------


def _step16_once(
    seed: int,
    plasticity_rate: float,
    init_v_mean: float,
    init_v_std: float,
) -> dict[str, float]:
    """Run step 16's three-phase measurement; return metrics."""
    a_is_inh = assign_ei_identity(S16_N, S16_IFRAC)
    b_is_inh = assign_ei_identity(S16_N, S16_IFRAC)
    pool_b0 = s16_random_b_pool(
        seed + 1, plasticity_rate, init_v_mean, init_v_std,
    )
    fit_before, _, _ = s16_measure_fitness(
        pool_b0, a_is_inh, b_is_inh, S16_MEASURE_STEPS,
    )
    state0 = s16_build_state(pool_b0)
    i_ext_a, i_ext_b, adr_a = s16_build_drives(S16_TRAIN_STEPS)
    t0 = time.perf_counter()
    final_state, _, _, val_trace = s16_training_scan(
        state0, a_is_inh, b_is_inh, i_ext_a, i_ext_b, adr_a,
    )
    jax.block_until_ready(final_state.b.pool.v)
    train_time = time.perf_counter() - t0
    trained_pool = final_state.b.pool
    fit_after, _, _ = s16_measure_fitness(
        trained_pool, a_is_inh, b_is_inh, S16_MEASURE_STEPS,
    )
    stats = s16_describe_pool(trained_pool)
    return {
        "fit_before": fit_before,
        "fit_after": fit_after,
        "train_time": train_time,
        "v_mean": stats["v_mean"],
        "v_std": stats["v_std"],
        "cross_e_frac": stats["cross_e_frac"],
        "valence_mean": float(val_trace.mean()),
    }


def block4_step16_multiseed(seeds: tuple[int, ...]) -> None:
    csv_path = RESULTS_DIR / "block4_step16_multiseed.csv"
    done = _load_completed(csv_path)
    rates = (0.01, 0.1, 0.3, 1.0, 3.0)
    init_means = (0.2, 1.0, 1.8)
    total = len(rates) * len(init_means) * len(seeds)
    log(
        f"[block4] step16 multi-seed: "
        f"{len(rates)} rates x {len(init_means)} init_v x "
        f"{len(seeds)} seeds = {total} runs"
    )
    idx = 0
    for rate, init_v in itertools.product(rates, init_means):
        config_name = f"step16_rate{rate}_ivm{init_v}"
        for seed in seeds:
            idx += 1
            key = (config_name, seed)
            if key in done:
                log(f"[block4] skip {idx}/{total} {config_name} seed={seed}")
                continue
            try:
                t0 = time.perf_counter()
                metrics = _step16_once(seed, rate, init_v, 0.3)
                dt_run = time.perf_counter() - t0
                row = {
                    "block": "step16_multiseed",
                    "config": config_name,
                    "seed": seed,
                    "plasticity_rate": rate,
                    "init_v_mean": init_v,
                    "fit_before": metrics["fit_before"],
                    "fit_after": metrics["fit_after"],
                    "delta": metrics["fit_after"] - metrics["fit_before"],
                    "v_mean": metrics["v_mean"],
                    "cross_e_frac_end": metrics["cross_e_frac"],
                    "valence_mean": metrics["valence_mean"],
                    "wall_sec": dt_run,
                }
                _append_row(csv_path, row)
                delta = metrics["fit_after"] - metrics["fit_before"]
                log(
                    f"[block4] {idx}/{total} {config_name} "
                    f"seed={seed}: delta={delta:+.2e}"
                )
            except Exception as exc:  # pylint: disable=broad-except
                log(f"[block4] ERROR {config_name} seed={seed}: {exc!r}")
            gc.collect()
    _git_commit_results("block4_step16_multiseed")


# ----------------------------------------------------------------------
# Block 5: Best-config high-seed-count confirmation
# ----------------------------------------------------------------------


def block5_best_config_confirm(seeds: tuple[int, ...]) -> None:
    """Once Blocks 1-4 have identified a promising config, run it with
    20 seeds to tighten the headline number. Default target is
    (stochastic, uniform, init_v=0.2, thr=0.05, dur=500) -- the
    "baseline" operating point for structural plasticity.
    """
    csv_path = RESULTS_DIR / "block5_best_config_confirm.csv"
    done = _load_completed(csv_path)
    configs = [
        Step17Config(
            "confirm_baseline_stdp_only", "off", "uniform",
            0.2, 0.0, 10_000_000,
        ),
        Step17Config(
            "confirm_stochastic_uniform_iv0.2_thr0.05_dur500",
            "stochastic", "uniform", 0.2, 0.05, 500,
        ),
        Step17Config(
            "confirm_stochastic_hebbian_iv0.2_thr0.05_dur500",
            "stochastic", "hebbian", 0.2, 0.05, 500,
        ),
    ]
    total = len(configs) * len(seeds)
    log(
        f"[block5] best-config confirmation: "
        f"{len(configs)} configs x {len(seeds)} seeds = {total} runs"
    )
    idx = 0
    for config in configs:
        for seed in seeds:
            idx += 1
            key = (config.name, seed)
            if key in done:
                log(f"[block5] skip {idx}/{total} {config.name} seed={seed}")
                continue
            try:
                t0 = time.perf_counter()
                metrics = step17_run_config(config, seed)
                dt_run = time.perf_counter() - t0
                row = {
                    "block": "best_config_confirm",
                    "config": config.name,
                    "seed": seed,
                    "fit_before": metrics["fit_before"],
                    "fit_after": metrics["fit_after"],
                    "delta": metrics["fit_after"] - metrics["fit_before"],
                    "v_mean": metrics["v_mean"],
                    "active_frac_end": metrics["active_frac_end"],
                    "cross_e_frac_end": metrics["cross_e_frac_end"],
                    "wall_sec": dt_run,
                }
                _append_row(csv_path, row)
                delta = metrics["fit_after"] - metrics["fit_before"]
                log(
                    f"[block5] {idx}/{total} {config.name} "
                    f"seed={seed}: delta={delta:+.2e} ({dt_run:.1f}s)"
                )
            except Exception as exc:  # pylint: disable=broad-except
                log(f"[block5] ERROR {config.name} seed={seed}: {exc!r}")
            gc.collect()
    _git_commit_results("block5_best_config_confirm")


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


BLOCKS: list[tuple[str, Callable[..., None], dict[str, Any]]] = [
    ("block1_step17_factorial", block1_step17_factorial, {
        "seeds": (0, 1, 2, 3, 4),
    }),
    ("block2_acquisition_probability", block2_acquisition_probability, {
        "seeds": (0, 1, 2, 3, 4),
    }),
    ("block3_long_training", block3_long_training, {
        "seeds": (0, 1, 2, 3, 4),
    }),
    ("block4_step16_multiseed", block4_step16_multiseed, {
        "seeds": (0, 1, 2, 3, 4),
    }),
    ("block5_best_config_confirm", block5_best_config_confirm, {
        "seeds": tuple(range(20)),
    }),
]


def main() -> None:
    log(f"[batch] starting; device={jax.default_backend()} "
        f"{jax.devices()[0]}")
    log(f"[batch] results dir: {RESULTS_DIR}")
    t_batch_start = time.perf_counter()
    for block_name, block_fn, kwargs in BLOCKS:
        t_block_start = time.perf_counter()
        log(f"[batch] === {block_name} begin ===")
        try:
            block_fn(**kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            log(
                f"[batch] {block_name} FAILED at top level: {exc!r}\n"
                f"{traceback.format_exc()}"
            )
        dt_block = time.perf_counter() - t_block_start
        log(f"[batch] === {block_name} end ({dt_block/60:.1f} min) ===")
    dt_batch = time.perf_counter() - t_batch_start
    log(f"[batch] ALL BLOCKS COMPLETE: {dt_batch/3600:.2f} hours")


if __name__ == "__main__":
    main()
