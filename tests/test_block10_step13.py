"""Behavioral tests for Block 10 step-13 E/I N=100 multi-seed.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Block 10 produces an N=100 multi-seed comparison of D008's
(i_frac=0.2, i_mult=8.0) operating point against the prior canonical
(0.2, 4.0), at step 13's T=2000 with closed-loop gain=50. Tests pin
the sweep-grid invariants and the resume-CSV / fitness-math
contracts so a future refactor doesn't silently change Block 10's
target.

GPU-running tests use the script's actual T=2000 (fast); no
test-runtime concern.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import pytest

# pylint: disable=import-error
import block10_step13_ei_n100 as b10

# pylint: disable=protected-access


# ----- Sweep-grid invariants -----------------------------------------------


def test_n_seeds_is_100() -> None:
    """Block 10's whole reason is N=100. Renaming the script without
    updating this constant is a bug."""
    assert b10.N_SEEDS == 100


def test_seed_stride_matches_step13() -> None:
    """Stride 37 matches step 13's _multi_seed_stats so seeds 0..3663
    align with any step 13 anchor seeds."""
    assert b10.SEED_STRIDE == 37
    assert b10.SEED_BASE == 0


def test_i_frac_is_canonical_only() -> None:
    """Block 10 holds i_frac fixed at 0.2 (D008 + step 13 grid's
    canonical column) and varies i_mult only. A future change to
    i_frac would mean a different experiment."""
    assert b10.I_FRAC == 0.2


def test_i_mults_compares_canonical_and_d008() -> None:
    """Two i_mult points: prior canonical (4.0) vs D008 candidate
    (8.0). Order matters for CSV column readability but not science."""
    assert b10.I_MULTS == (4.0, 8.0)


def test_closed_loop_gain_matches_step10() -> None:
    """gain=50 matches step 10 / step 13 closed-loop default. A
    different gain is a different experiment."""
    assert b10.CLOSED_LOOP_GAIN == 50.0


def test_t_steps_matches_step13() -> None:
    """T=2000 keeps Block 10 directly comparable to step 13's existing
    grid measurements (which were single-seed at T=2000)."""
    assert b10.T_STEPS == 2_000


def test_expected_total_matches_grid() -> None:
    """2 i_mults x 100 seeds x 2 conditions = 400."""
    assert b10._expected_total() == 400
    assert b10._expected_total() == (
        len(b10.I_MULTS) * b10.N_SEEDS * len(b10.CONDITIONS)
    )


def test_conditions_are_open_and_closed_loop() -> None:
    """Resume key parsing needs these exact strings."""
    assert b10.CONDITIONS == ("open_loop", "closed_loop")


# ----- _completed_pairs ----------------------------------------------------


def test_completed_pairs_returns_empty_for_missing_file(
    tmp_path: Path,
) -> None:
    p = tmp_path / "no_such_file.csv"
    assert b10._completed_pairs(p) == set()


def test_completed_pairs_reads_existing_rows(tmp_path: Path) -> None:
    """Each row's (seed, i_mult, condition) tuple lands in the set."""
    p = tmp_path / "rows.csv"
    p.write_text(
        "seed,i_mult,condition,fitness,wall_sec\n"
        "0,4.0,open_loop,-1.500000e-04,5.10\n"
        "0,8.0,closed_loop,-2.700000e-05,7.20\n"
        "37,4.0,closed_loop,-2.800000e-05,7.30\n"
    )
    completed = b10._completed_pairs(p)
    assert (0, 4.0, "open_loop") in completed
    assert (0, 8.0, "closed_loop") in completed
    assert (37, 4.0, "closed_loop") in completed
    assert len(completed) == 3


# ----- _append_row ---------------------------------------------------------


def test_append_row_creates_file_with_header(tmp_path: Path) -> None:
    p = tmp_path / "new.csv"
    b10._append_row(p, 0, 4.0, "open_loop", -1.5e-4, 5.1)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0] == ["seed", "i_mult", "condition", "fitness", "wall_sec"]
    assert rows[1] == [
        "0", "4.0", "open_loop", "-1.500000e-04", "5.10",
    ]


def test_append_row_appends_without_duplicating_header(
    tmp_path: Path,
) -> None:
    p = tmp_path / "appended.csv"
    b10._append_row(p, 0, 4.0, "open_loop", -1.5e-4, 5.1)
    b10._append_row(p, 0, 8.0, "closed_loop", -2.7e-5, 7.2)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert len(rows) == 3
    assert rows[1][2] == "open_loop"
    assert rows[2][2] == "closed_loop"


def test_append_row_then_completed_pairs_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "rt.csv"
    b10._append_row(p, 37, 8.0, "closed_loop", -2.8e-5, 7.3)
    completed = b10._completed_pairs(p)
    assert (37, 8.0, "closed_loop") in completed


def test_append_row_writes_header_to_existing_empty_file(
    tmp_path: Path,
) -> None:
    """If the CSV file exists but is empty (e.g. a prior run created
    it then crashed before writing anything), _append_row writes the
    header before the first data row.

    Pin: this is the fix for the "header missing after interrupted
    write" failure mode (Gemini Block 10 review G10-iii). Without
    it, _completed_pairs would treat the first data row as the
    header on resume, silently corrupting the resume set.
    """
    p = tmp_path / "empty_then_append.csv"
    p.touch()  # zero-byte file
    assert p.exists() and p.stat().st_size == 0
    b10._append_row(p, 0, 4.0, "open_loop", -1.5e-4, 5.1)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0] == ["seed", "i_mult", "condition", "fitness", "wall_sec"]
    assert rows[1][0] == "0"
    # Round-trip: _completed_pairs should now correctly find the row.
    assert (0, 4.0, "open_loop") in b10._completed_pairs(p)


# ----- _log ----------------------------------------------------------------


def test_log_writes_timestamped_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_log prints to stdout and appends to LOG_PATH, with the
    bracketed UTC timestamp prefix that ``tail -f`` users expect."""
    log_path = tmp_path / "block10.log"
    monkeypatch.setattr(b10, "LOG_PATH", log_path)
    b10._log("hello block 10")
    out = capsys.readouterr().out
    # Test name promises "timestamped" - pin the bracket prefix so a
    # future change to _log's format doesn't silently drop it.
    assert "hello block 10" in out
    assert out.startswith("[")
    assert "+00:00]" in out  # UTC offset suffix from isoformat
    contents = log_path.read_text()
    assert "hello block 10" in contents
    assert contents.startswith("[")
    assert "+00:00]" in contents


# ----- _prediction_fitness_from_rates --------------------------------------


def test_prediction_fitness_zero_for_identical_uniform_rates() -> None:
    """Uniform identical rates give fitness == 0."""
    rate_a = jnp.full((b10.T_STEPS,), 0.05, dtype=jnp.float32)
    rate_b = jnp.full((b10.T_STEPS,), 0.05, dtype=jnp.float32)
    fit = b10._prediction_fitness_from_rates(rate_a, rate_b)
    assert abs(fit) < 1e-12


def test_prediction_fitness_negative_for_prediction_error() -> None:
    """Mismatched windowed rates produce strictly negative fitness."""
    n_windows = b10.T_STEPS // b10.WINDOW_STEPS
    a_per_window = jnp.linspace(0.01, 0.10, n_windows)
    b_per_window = jnp.full((n_windows,), 0.005, dtype=jnp.float32)
    rate_a = jnp.repeat(a_per_window, b10.WINDOW_STEPS).astype(jnp.float32)
    rate_b = jnp.repeat(b_per_window, b10.WINDOW_STEPS).astype(jnp.float32)
    fit = b10._prediction_fitness_from_rates(rate_a, rate_b)
    assert fit < 0.0
    assert bool(jnp.isfinite(jnp.array(fit)))


# ----- _conditions_for_seed_imult ------------------------------------------


def test_conditions_for_seed_imult_all_when_completed_empty() -> None:
    """No prior runs -> both conditions remain."""
    todo = b10._conditions_for_seed_imult(0, 4.0, set())
    assert todo == list(b10.CONDITIONS)


def test_conditions_for_seed_imult_subset_when_partially_completed() -> None:
    """open_loop done -> only closed_loop remains."""
    completed = {(0, 4.0, "open_loop")}
    todo = b10._conditions_for_seed_imult(0, 4.0, completed)
    assert todo == ["closed_loop"]


def test_conditions_for_seed_imult_empty_when_all_completed() -> None:
    completed = {(0, 4.0, "open_loop"), (0, 4.0, "closed_loop")}
    assert b10._conditions_for_seed_imult(0, 4.0, completed) == []


def test_conditions_for_seed_imult_does_not_match_other_seed_or_mult() -> None:
    """Completed entries for other (seed, i_mult) tuples don't mask todo."""
    completed = {(0, 4.0, "open_loop"), (37, 4.0, "open_loop")}
    todo = b10._conditions_for_seed_imult(0, 8.0, completed)
    assert todo == list(b10.CONDITIONS)


# ----- _evaluate_one (GPU-running; T=2000 is fast) -------------------------


def test_evaluate_one_runs_only_open_loop_when_requested() -> None:
    """When only open_loop is requested, only it is computed."""
    import math  # pylint: disable=import-outside-toplevel
    results = b10._evaluate_one(0, 4.0, ["open_loop"])
    assert set(results.keys()) == {"open_loop"}
    fitness, wall = results["open_loop"]
    assert isinstance(fitness, float)
    assert math.isfinite(fitness)
    assert fitness <= 0.0
    assert wall > 0.0


def test_evaluate_one_empty_todo_returns_empty_dict() -> None:
    results = b10._evaluate_one(0, 4.0, [])
    assert not results
