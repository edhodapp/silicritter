"""Behavioral tests for Block 12 fGn Hurst sweep at N=100.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

# pylint: disable=import-error
import block12_fgn_n100 as b12
import step14_fgn_stimulus as s14

# pylint: disable=protected-access


# ----- Sweep-grid invariants -----------------------------------------------


def test_n_seeds_is_100() -> None:
    assert b12.N_SEEDS == 100


def test_hurst_grid_matches_step14() -> None:
    """4 H values matching step 14's HURST_GRID."""
    assert b12.HURST_GRID == s14.HURST_GRID
    assert b12.HURST_GRID == (0.3, 0.5, 0.7, 0.9)


def test_conditions_are_open_and_closed_loop() -> None:
    assert b12.CONDITIONS == ("open_loop", "closed_loop")


def test_seed_stride_matches_other_blocks() -> None:
    """Stride 37 matches step 13 / Block 10 / Block 9 / Phase 2."""
    assert b12.SEED_STRIDE == 37
    assert b12.SEED_BASE == 0


def test_expected_total_matches_grid() -> None:
    """4 H x 100 seeds x 2 conditions = 800."""
    assert b12._expected_total() == 800
    assert b12._expected_total() == (
        len(b12.HURST_GRID) * b12.N_SEEDS * len(b12.CONDITIONS)
    )


# ----- _completed_pairs ----------------------------------------------------


def test_completed_pairs_returns_empty_for_missing_file(
    tmp_path: Path,
) -> None:
    p = tmp_path / "no.csv"
    assert b12._completed_pairs(p) == set()


def test_completed_pairs_reads_existing_rows(tmp_path: Path) -> None:
    p = tmp_path / "rows.csv"
    p.write_text(
        "seed,hurst,condition,track_fit,pred_fit,stim_lag1,"
        "a_rate_mean,b_rate_mean,wall_sec\n"
        "0,0.3,open_loop,-1.5e-04,-1.5e-04,0.05,42.0,38.0,0.5\n"
        "0,0.7,closed_loop,-3.0e-05,-3.5e-05,0.45,42.0,40.0,0.7\n"
    )
    completed = b12._completed_pairs(p)
    assert (0, 0.3, "open_loop") in completed
    assert (0, 0.7, "closed_loop") in completed
    assert len(completed) == 2


# ----- _append_row ---------------------------------------------------------


def test_append_row_creates_file_with_header(tmp_path: Path) -> None:
    p = tmp_path / "new.csv"
    b12._append_row(
        p, 0, 0.5, "open_loop",
        -1.5e-4, -1.5e-4, 0.1, 42.0, 38.0, 0.5,
    )
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0] == [
        "seed", "hurst", "condition", "track_fit", "pred_fit",
        "stim_lag1", "a_rate_mean", "b_rate_mean", "wall_sec",
    ]
    assert rows[1][0] == "0"
    assert rows[1][1] == "0.5"
    assert rows[1][2] == "open_loop"


def test_append_row_writes_header_to_existing_empty_file(
    tmp_path: Path,
) -> None:
    p = tmp_path / "empty.csv"
    p.touch()
    assert p.stat().st_size == 0
    b12._append_row(
        p, 0, 0.3, "open_loop",
        -1.5e-4, -1.5e-4, 0.05, 42.0, 38.0, 0.5,
    )
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0][0] == "seed"


def test_append_row_then_completed_pairs_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "rt.csv"
    b12._append_row(
        p, 37, 0.7, "closed_loop",
        -3.0e-5, -3.5e-5, 0.45, 42.0, 40.0, 0.7,
    )
    completed = b12._completed_pairs(p)
    assert (37, 0.7, "closed_loop") in completed


# ----- _conditions_to_run --------------------------------------------------


def test_conditions_to_run_all_when_completed_empty() -> None:
    todo = b12._conditions_to_run(0, 0.3, set())
    assert todo == list(b12.CONDITIONS)


def test_conditions_to_run_subset_when_partially_completed() -> None:
    completed = {(0, 0.3, "open_loop")}
    todo = b12._conditions_to_run(0, 0.3, completed)
    assert todo == ["closed_loop"]


def test_conditions_to_run_empty_when_all_completed() -> None:
    completed = {(0, 0.3, "open_loop"), (0, 0.3, "closed_loop")}
    assert not b12._conditions_to_run(0, 0.3, completed)


# ----- _log ----------------------------------------------------------------


def test_log_writes_timestamped_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "block12.log"
    monkeypatch.setattr(b12, "LOG_PATH", log_path)
    b12._log("hello block 12")
    out = capsys.readouterr().out
    assert "hello block 12" in out
    assert out.startswith("[")
    contents = log_path.read_text()
    assert "hello block 12" in contents


# ----- Cross-file coupling regression: s14._run_condition ------------------


def test_run_one_calls_s14_run_condition_with_expected_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the s14._run_condition call signature.

    Step 14 refactor that renames or reorders args breaks Block 12
    at gate time, not at multi-hour run time.
    """
    captured: dict[str, Any] = {}

    def stub_run_condition(
        hurst: float,
        closed_loop: bool,
        seed: int,
    ) -> tuple[float, float, float, float, float]:
        captured.update(
            hurst=hurst, closed_loop=closed_loop, seed=seed,
        )
        return (-1.5e-04, -1.6e-04, 0.45, 42.5, 38.5)

    monkeypatch.setattr(s14, "_run_condition", stub_run_condition)
    metrics, wall = b12._run_one(7, 0.7, "closed_loop")
    assert captured["hurst"] == 0.7
    assert captured["closed_loop"] is True
    assert captured["seed"] == 7
    assert isinstance(metrics, tuple)
    assert len(metrics) == 5
    # Float coercion: pin that all 5 metrics are Python floats so the
    # CSV-format string doesn't choke on JAX scalars.
    assert all(isinstance(m, float) for m in metrics)
    assert wall >= 0.0


def test_run_one_open_loop_passes_closed_loop_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def stub_run_condition(
        hurst: float, closed_loop: bool, seed: int,
    ) -> tuple[float, float, float, float, float]:
        del hurst, seed
        captured["closed_loop"] = closed_loop
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(s14, "_run_condition", stub_run_condition)
    b12._run_one(0, 0.3, "open_loop")
    assert captured["closed_loop"] is False


# ----- _evaluate_one (GPU-running smoke; uses smallest H, single seed) -----


def test_evaluate_one_returns_5_metric_floats() -> None:
    """End-to-end smoke at H=0.5, seed=0, open_loop: returns 5
    float metrics + wall time. T=2000 makes this fast."""
    import math  # pylint: disable=import-outside-toplevel
    metrics, wall = b12._run_one(0, 0.5, "open_loop")
    assert isinstance(metrics, tuple) and len(metrics) == 5
    track, pred, lag1, a_mean, b_mean = metrics
    for m in (track, pred):
        assert math.isfinite(m)
        assert m <= 0.0  # negative MSE
    for m in (lag1, a_mean, b_mean):
        assert math.isfinite(m)
    assert wall > 0.0
