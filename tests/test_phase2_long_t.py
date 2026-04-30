"""Behavioral tests for Phase 2 long-T reproducer.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Pins the contract of every branch in
``experiments/phase2_step10_long_t.py``: CSV resume reads, append
rounds-trip, fitness math matches step 10's, setup builds traces of
the right shape, condition-filter computes the right todo list, and
``_evaluate_one`` runs only the requested subset of conditions.

Per the project HARD RULE on assertable contracts: every branch in the
script either has a unit test here or is exercised through an
integration call so the test fails meaningfully if the branch's
behavior drifts.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import pytest

# experiments/ is on sys.path via pyproject.toml's
# [tool.pytest.ini_options].pythonpath; pylint runs separately and does
# not read pytest config, so the import-error disable is still needed.
# pylint: disable=import-error
import phase2_step10_long_t as p2

# pylint: disable=protected-access


# ----- _completed_pairs ----------------------------------------------------


def test_completed_pairs_returns_empty_for_missing_file(
    tmp_path: Path,
) -> None:
    """Missing CSV is the first-launch case; must return empty, not raise."""
    p = tmp_path / "no_such_file.csv"
    assert p2._completed_pairs(p) == set()


def test_completed_pairs_reads_existing_rows(tmp_path: Path) -> None:
    """Each row's (seed, t_steps, condition) tuple lands in the set."""
    p = tmp_path / "rows.csv"
    p.write_text(
        "seed,t_steps,condition,fitness,wall_sec\n"
        "0,10000,open_loop,-1.500000e-04,2.50\n"
        "0,10000,gain=50,-5.600000e-05,2.70\n"
        "37,100000,gain=200,-5.500000e-05,28.10\n"
    )
    completed = p2._completed_pairs(p)
    assert (0, 10000, "open_loop") in completed
    assert (0, 10000, "gain=50") in completed
    assert (37, 100000, "gain=200") in completed
    assert len(completed) == 3


# ----- _append_row ---------------------------------------------------------


def test_append_row_creates_file_with_header(tmp_path: Path) -> None:
    """First write creates the file and emits the header row first."""
    p = tmp_path / "new.csv"
    p2._append_row(p, 0, 10000, "open_loop", -1.5e-4, 2.5)
    assert p.exists()
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0] == ["seed", "t_steps", "condition", "fitness", "wall_sec"]
    assert rows[1] == [
        "0", "10000", "open_loop", "-1.500000e-04", "2.50",
    ]


def test_append_row_appends_without_duplicating_header(
    tmp_path: Path,
) -> None:
    """Subsequent writes append data rows without re-emitting the header."""
    p = tmp_path / "appended.csv"
    p2._append_row(p, 0, 10000, "open_loop", -1.5e-4, 2.5)
    p2._append_row(p, 0, 10000, "gain=50", -5.6e-5, 2.7)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert len(rows) == 3  # header + two data rows
    assert rows[0][0] == "seed"
    assert rows[1][2] == "open_loop"
    assert rows[2][2] == "gain=50"


def test_append_row_then_completed_pairs_round_trip(tmp_path: Path) -> None:
    """A row written by _append_row is found by _completed_pairs."""
    p = tmp_path / "rt.csv"
    p2._append_row(p, 37, 100000, "gain=200", -5.5e-5, 28.1)
    completed = p2._completed_pairs(p)
    assert (37, 100000, "gain=200") in completed


# ----- _log ----------------------------------------------------------------


def test_log_writes_timestamped_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_log prints to stdout and appends the same line to LOG_PATH."""
    log_path = tmp_path / "phase2.log"
    monkeypatch.setattr(p2, "LOG_PATH", log_path)
    p2._log("hello world")
    out = capsys.readouterr().out
    assert "hello world" in out
    assert "[" in out and "]" in out  # timestamp brackets
    contents = log_path.read_text()
    assert "hello world" in contents
    assert contents.endswith("\n")


# ----- _prediction_fitness_from_rates --------------------------------------


def test_prediction_fitness_zero_for_identical_uniform_rates() -> None:
    """Uniform rates with no prediction error give fitness == 0."""
    t = 1000
    rate_a = jnp.full((t,), 0.05, dtype=jnp.float32)
    rate_b = jnp.full((t,), 0.05, dtype=jnp.float32)
    fit = p2._prediction_fitness_from_rates(rate_a, rate_b, t)
    assert abs(fit) < 1e-12


def test_prediction_fitness_negative_for_prediction_error() -> None:
    """Mismatched windowed rates produce strictly negative fitness."""
    t = 1000
    n_windows = t // p2.WINDOW_STEPS
    a_per_window = jnp.linspace(0.01, 0.10, n_windows)
    b_per_window = jnp.full((n_windows,), 0.005, dtype=jnp.float32)
    rate_a = jnp.repeat(a_per_window, p2.WINDOW_STEPS).astype(jnp.float32)
    rate_b = jnp.repeat(b_per_window, p2.WINDOW_STEPS).astype(jnp.float32)
    fit = p2._prediction_fitness_from_rates(rate_a, rate_b, t)
    assert fit < 0.0
    assert bool(jnp.isfinite(jnp.array(fit)))


def test_prediction_fitness_rejects_t_not_multiple_of_window() -> None:
    """T not divisible by WINDOW_STEPS cannot be reshaped → AssertionError."""
    rate_a = jnp.zeros((150,), dtype=jnp.float32)
    rate_b = jnp.zeros((150,), dtype=jnp.float32)
    with pytest.raises(AssertionError, match="WINDOW_STEPS"):
        p2._prediction_fitness_from_rates(rate_a, rate_b, 150)


def test_prediction_fitness_rejects_t_equal_to_window_steps() -> None:
    """``T == WINDOW_STEPS`` => n_windows == 1 => empty lead-lag diff =>
    NaN if not asserted. Finding E adds a guard; this test pins it."""
    t = p2.WINDOW_STEPS
    rate_a = jnp.zeros((t,), dtype=jnp.float32)
    rate_b = jnp.zeros((t,), dtype=jnp.float32)
    with pytest.raises(AssertionError, match="n_windows"):
        p2._prediction_fitness_from_rates(rate_a, rate_b, t)


# ----- _setup_for_seed -----------------------------------------------------


def test_setup_for_seed_returns_traces_of_right_shape() -> None:
    """Setup builds per-step scalar drive traces of length t_steps.

    i_ext_a / i_ext_b are ``(T,)`` not ``(T, N)``: the drive is uniform
    across neurons within a step, and ``simulate_paired`` broadcasts the
    scalar to ``(N,)`` inside its scan body. Avoids the ``T*N`` float32
    allocation that OOMs the GPU at T=10M (finding A from review).
    """
    t = 4000
    setup = p2._setup_for_seed(0, t)
    initial_state, a_is_inh, b_is_inh, i_ext_a, i_ext_b, valence, adr_a = (
        setup
    )
    assert i_ext_a.shape == (t,)
    assert i_ext_b.shape == (t,)
    assert valence.shape == (t,)
    assert adr_a.shape == (t,)
    assert a_is_inh.shape == (p2.N_NEURONS,)
    assert b_is_inh.shape == (p2.N_NEURONS,)
    # Initial state is a PairedState with both pools at full size.
    assert initial_state.a.pool.pre_ids.shape == (p2.N_NEURONS, p2.K_SLOTS)
    assert initial_state.b.pool.pre_ids.shape == (p2.N_NEURONS, p2.K_SLOTS)


def test_setup_for_seed_rejects_t_not_multiple_of_drive_profile() -> None:
    """A drive profile has 4 levels, so t must be divisible by 4."""
    with pytest.raises(AssertionError, match="A drive profile"):
        p2._setup_for_seed(0, 5)  # not divisible by 4


def test_setup_for_seed_drive_profile_segments_match_levels() -> None:
    """i_ext_a's per-segment mean reproduces A_DRIVE_PROFILE values."""
    t = 4000
    setup = p2._setup_for_seed(0, t)
    i_ext_a = setup[3]
    seg_len = t // len(p2.A_DRIVE_PROFILE)
    for i, expected_level in enumerate(p2.A_DRIVE_PROFILE):
        seg = i_ext_a[i * seg_len:(i + 1) * seg_len]
        assert float(seg.mean()) == pytest.approx(expected_level, rel=1e-6)


# ----- _conditions_for_seed_t ----------------------------------------------


def test_conditions_for_seed_t_all_when_completed_empty() -> None:
    """No prior runs → all 4 conditions remain."""
    todo = p2._conditions_for_seed_t(0, 10_000, set())
    assert todo == [
        p2.OPEN_LOOP_LABEL,
        "gain=10", "gain=50", "gain=200",
    ]


def test_conditions_for_seed_t_subset_when_partially_completed() -> None:
    """Only conditions absent from completed remain in the todo list."""
    completed = {
        (0, 10_000, p2.OPEN_LOOP_LABEL),
        (0, 10_000, "gain=50"),
    }
    todo = p2._conditions_for_seed_t(0, 10_000, completed)
    assert todo == ["gain=10", "gain=200"]


def test_conditions_for_seed_t_empty_when_all_completed() -> None:
    """All 4 conditions done → empty list (caller skips this (seed, T))."""
    completed = {
        (0, 10_000, p2.OPEN_LOOP_LABEL),
        (0, 10_000, "gain=10"),
        (0, 10_000, "gain=50"),
        (0, 10_000, "gain=200"),
    }
    assert p2._conditions_for_seed_t(0, 10_000, completed) == []


def test_conditions_for_seed_t_does_not_match_other_seed_t() -> None:
    """Completed entries for other (seed, T) tuples do NOT mask todo."""
    completed = {
        (0, 10_000, p2.OPEN_LOOP_LABEL),  # different seed/T
    }
    todo = p2._conditions_for_seed_t(37, 100_000, completed)
    assert todo == [
        p2.OPEN_LOOP_LABEL, "gain=10", "gain=50", "gain=200",
    ]


# ----- _expected_total -----------------------------------------------------


def test_expected_total_matches_sweep_grid() -> None:
    """Total grid size = N_DURATIONS x N_SEEDS x (1 + N_GAINS)."""
    expected = (
        len(p2.DURATIONS) * p2.N_SEEDS * (1 + len(p2.GAINS))
    )
    assert p2._expected_total() == expected


# ----- _evaluate_one (GPU-running tests; tagged for speed awareness) -------


def test_evaluate_one_runs_only_open_loop_when_requested() -> None:
    """When only OPEN_LOOP_LABEL is in the todo list, only it is computed."""
    # Use the smallest duration (T=10k) to keep the test under ~15s
    # including JIT compile.
    results = p2._evaluate_one(0, 10_000, [p2.OPEN_LOOP_LABEL])
    assert set(results.keys()) == {p2.OPEN_LOOP_LABEL}
    fitness, wall = results[p2.OPEN_LOOP_LABEL]
    assert isinstance(fitness, float)
    assert wall > 0.0


def test_evaluate_one_runs_only_specified_gains() -> None:
    """When only one gain is requested, no other condition is computed."""
    results = p2._evaluate_one(0, 10_000, ["gain=50"])
    assert set(results.keys()) == {"gain=50"}


def test_evaluate_one_empty_todo_returns_empty_dict() -> None:
    """Empty todo list → empty results (no sim run)."""
    results = p2._evaluate_one(0, 10_000, [])
    assert not results
