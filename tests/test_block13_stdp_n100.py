"""Behavioral tests for Block 13 step-16 STDP N=100 headline reanchor.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

# pylint: disable=import-error
import block13_stdp_n100 as b13
import overnight_batch as ob
import step16_stdp_learning as s16

# pylint: disable=protected-access


# ----- Sweep-grid invariants -----------------------------------------------


def test_n_seeds_is_100() -> None:
    assert b13.N_SEEDS == 100


def test_plasticity_rate_is_headline() -> None:
    """rate=1.0 is the +14% headline config from the README dev log."""
    assert b13.PLASTICITY_RATE == 1.0


def test_init_v_mean_is_default() -> None:
    """init_v_mean=1.0 matches step 16's INIT_V_MEAN default."""
    assert b13.INIT_V_MEAN == 1.0
    assert b13.INIT_V_MEAN == s16.INIT_V_MEAN


def test_init_v_std_matches_step16() -> None:
    assert b13.INIT_V_STD == s16.INIT_V_STD


def test_seed_stride_matches_other_blocks() -> None:
    assert b13.SEED_STRIDE == 37
    assert b13.SEED_BASE == 0


def test_expected_total_is_n_seeds() -> None:
    """One seed = one (fit_before, fit_after) row. 100 total."""
    assert b13._expected_total() == 100


# ----- _completed_seeds ----------------------------------------------------


def test_completed_seeds_returns_empty_for_missing_file(
    tmp_path: Path,
) -> None:
    p = tmp_path / "no.csv"
    assert b13._completed_seeds(p) == set()


def test_completed_seeds_reads_existing_rows(tmp_path: Path) -> None:
    p = tmp_path / "rows.csv"
    p.write_text(
        "seed,fit_before,fit_after,improvement_pct,train_time,wall_sec\n"
        "0,-2.500000e-04,-2.200000e-04,12.00,1.50,2.00\n"
        "37,-2.510000e-04,-2.180000e-04,13.10,1.50,2.00\n"
    )
    completed = b13._completed_seeds(p)
    assert 0 in completed
    assert 37 in completed
    assert len(completed) == 2


# ----- _append_row + empty-file header fix ---------------------------------


def test_append_row_creates_file_with_header(tmp_path: Path) -> None:
    p = tmp_path / "new.csv"
    b13._append_row(p, 0, -2.5e-04, -2.2e-04, 12.0, 1.5, 2.0)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0] == [
        "seed", "fit_before", "fit_after", "improvement_pct",
        "train_time", "wall_sec",
    ]
    assert rows[1][0] == "0"


def test_append_row_writes_header_to_existing_empty_file(
    tmp_path: Path,
) -> None:
    p = tmp_path / "empty.csv"
    p.touch()
    assert p.stat().st_size == 0
    b13._append_row(p, 0, -2.5e-04, -2.2e-04, 12.0, 1.5, 2.0)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0][0] == "seed"


def test_append_row_then_completed_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "rt.csv"
    b13._append_row(p, 37, -2.5e-04, -2.2e-04, 12.0, 1.5, 2.0)
    assert 37 in b13._completed_seeds(p)


# ----- _seeds_to_run -------------------------------------------------------


def test_seeds_to_run_all_when_completed_empty() -> None:
    todo = b13._seeds_to_run(set())
    assert len(todo) == b13.N_SEEDS
    assert b13.SEED_BASE in todo
    assert b13.SEED_BASE + (b13.N_SEEDS - 1) * b13.SEED_STRIDE in todo


def test_seeds_to_run_subset_when_partially_completed() -> None:
    completed = {0, 37, 74}
    todo = b13._seeds_to_run(completed)
    assert 0 not in todo
    assert 37 not in todo
    assert 74 not in todo
    assert len(todo) == b13.N_SEEDS - 3


def test_seeds_to_run_empty_when_all_completed() -> None:
    completed = {
        b13.SEED_BASE + i * b13.SEED_STRIDE for i in range(b13.N_SEEDS)
    }
    assert not b13._seeds_to_run(completed)


# ----- _log ----------------------------------------------------------------


def test_log_writes_timestamped_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "block13.log"
    monkeypatch.setattr(b13, "LOG_PATH", log_path)
    b13._log("hello block 13")
    out = capsys.readouterr().out
    assert "hello block 13" in out
    assert out.startswith("[")


# ----- Cross-file coupling: ob._step16_once signature ----------------------


def test_run_one_calls_step16_once_with_expected_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the ob._step16_once call signature.

    Overnight_batch refactor that renames or reorders args breaks
    Block 13 at gate time, not at multi-minute run time.
    """
    captured: dict[str, Any] = {}

    def stub_step16_once(
        seed: int,
        plasticity_rate: float,
        init_v_mean: float,
        init_v_std: float,
    ) -> dict[str, float]:
        captured.update(
            seed=seed, plasticity_rate=plasticity_rate,
            init_v_mean=init_v_mean, init_v_std=init_v_std,
        )
        return {
            "fit_before": -2.5e-04,
            "fit_after": -2.2e-04,
            "train_time": 1.5,
            "v_mean": 0.5,
            "v_std": 0.1,
            "cross_e_frac": 0.4,
            "valence_mean": 0.0,
        }

    monkeypatch.setattr(ob, "_step16_once", stub_step16_once)
    metrics, wall = b13._run_one(7)
    assert captured["seed"] == 7
    assert captured["plasticity_rate"] == b13.PLASTICITY_RATE
    assert captured["init_v_mean"] == b13.INIT_V_MEAN
    assert captured["init_v_std"] == b13.INIT_V_STD
    assert isinstance(metrics["fit_before"], float)
    assert isinstance(metrics["fit_after"], float)
    assert wall >= 0.0


def test_improvement_pct_computed_correctly() -> None:
    """improvement_pct = 100 * (fit_after - fit_before) / |fit_before|."""
    pct = b13._improvement_pct(-2.5e-04, -2.2e-04)
    assert pct == pytest.approx(12.0, abs=0.01)
    # Negative fit_before convention: improvement is movement toward zero.
    pct2 = b13._improvement_pct(-1.0e-04, -1.1e-04)
    assert pct2 == pytest.approx(-10.0, abs=0.01)
