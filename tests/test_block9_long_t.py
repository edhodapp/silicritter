"""Behavioral tests for Block 9 N=500 long-T reproducer.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Block 9 is structurally identical to Phase 2's long-T reproducer
(experiments/phase2_step10_long_t.py), differing only in N_SEEDS and
in the conditions sweep (single gain=200 vs Phase 2's three gains).
Tests pin the sweep-grid invariants (expected total, conditions list,
seed stride, durations) so a future refactor that consolidates Phase
2 / Block 9 into a shared runner doesn't silently change Block 9's
sweep target.

Per the project HARD RULE on assertable contracts: every branch in
the script either has a unit test here or is exercised through an
integration call so the test fails meaningfully if the branch's
behavior drifts.

GPU-running tests use the smallest configured duration (T=10000) and
exercise only the open-loop path to keep the test wall-time bounded.
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import pytest

# experiments/ is on sys.path via pyproject pythonpath.
# pylint: disable=import-error
import block9_step10_n500 as b9

# pylint: disable=protected-access


# ----- Sweep-grid invariants -----------------------------------------------


def test_n_seeds_is_500() -> None:
    """Block 9's whole reason for existing is N=500. If anyone changes
    this without renaming the script, the test fails."""
    assert b9.N_SEEDS == 500


def test_gains_is_single_headline_value() -> None:
    """Block 9 measures only the headline gain=200, not the full
    Phase 2 gain sweep. This keeps wall-time / cost on AWS spot
    bounded; Phase 2 already established the gain ordering at N=5."""
    assert b9.GAINS == (200.0,)


def test_gain_condition_string_format() -> None:
    """The condition string is built via ``f"gain={g:g}"`` and used as
    the CSV resume key. Pin the exact string ``"gain=200"`` so a
    future change to GAINS or to the format spec doesn't silently
    break resume by mismatching prior-run CSV keys.
    """
    assert f"gain={b9.GAINS[0]:g}" == "gain=200"


def test_durations_match_phase2() -> None:
    """Same four durations as Phase 2 - the difference is only in N
    and gain count, not in the T-sweep grid."""
    assert b9.DURATIONS == (10_000, 100_000, 1_000_000, 10_000_000)


def test_seed_stride_matches_phase2() -> None:
    """Stride 37 matches Phase 2 / step 10 so seeds 0..148 align with
    the existing N=5 anchors. Block 9's seeds are 0, 37, 74, ... 18463."""
    assert b9.SEED_STRIDE == 37
    assert b9.SEED_BASE == 0


def test_expected_total_matches_grid() -> None:
    """4 durations x 500 seeds x (1 open_loop + 1 gain=200) = 4000."""
    assert b9._expected_total() == 4_000
    assert b9._expected_total() == (
        len(b9.DURATIONS) * b9.N_SEEDS * (1 + len(b9.GAINS))
    )


# ----- _completed_pairs ----------------------------------------------------


def test_completed_pairs_returns_empty_for_missing_file(
    tmp_path: Path,
) -> None:
    """Missing CSV is the first-launch case; must return empty, not raise."""
    p = tmp_path / "no_such_file.csv"
    assert b9._completed_pairs(p) == set()


def test_completed_pairs_reads_existing_rows(tmp_path: Path) -> None:
    """Each row's (seed, t_steps, condition) tuple lands in the set."""
    p = tmp_path / "rows.csv"
    p.write_text(
        "seed,t_steps,condition,fitness,wall_sec\n"
        "0,10000,open_loop,-1.500000e-04,0.50\n"
        "0,10000,gain=200,-3.668605e-05,0.70\n"
        "37,100000,gain=200,-2.770e-05,3.10\n"
    )
    completed = b9._completed_pairs(p)
    assert (0, 10000, "open_loop") in completed
    assert (0, 10000, "gain=200") in completed
    assert (37, 100000, "gain=200") in completed
    assert len(completed) == 3


# ----- _append_row ---------------------------------------------------------


def test_append_row_creates_file_with_header(tmp_path: Path) -> None:
    """First write creates the file and emits the header row first."""
    p = tmp_path / "new.csv"
    b9._append_row(p, 0, 10000, "open_loop", -1.5e-4, 0.5)
    assert p.exists()
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0] == ["seed", "t_steps", "condition", "fitness", "wall_sec"]
    assert rows[1] == [
        "0", "10000", "open_loop", "-1.500000e-04", "0.50",
    ]


def test_append_row_appends_without_duplicating_header(
    tmp_path: Path,
) -> None:
    """Subsequent writes append without re-emitting the header."""
    p = tmp_path / "appended.csv"
    b9._append_row(p, 0, 10000, "open_loop", -1.5e-4, 0.5)
    b9._append_row(p, 0, 10000, "gain=200", -3.669e-5, 0.7)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert len(rows) == 3
    assert rows[1][2] == "open_loop"
    assert rows[2][2] == "gain=200"


def test_append_row_then_completed_pairs_round_trip(tmp_path: Path) -> None:
    """A row written by _append_row is found by _completed_pairs."""
    p = tmp_path / "rt.csv"
    b9._append_row(p, 37, 100000, "gain=200", -2.77e-5, 3.1)
    completed = b9._completed_pairs(p)
    assert (37, 100000, "gain=200") in completed


# ----- _log ----------------------------------------------------------------


def test_log_writes_timestamped_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_log prints to stdout and appends to LOG_PATH."""
    log_path = tmp_path / "block9.log"
    monkeypatch.setattr(b9, "LOG_PATH", log_path)
    b9._log("hello block 9")
    out = capsys.readouterr().out
    assert "hello block 9" in out
    assert "[" in out and "]" in out
    contents = log_path.read_text()
    assert "hello block 9" in contents
    assert contents.endswith("\n")


# ----- _prediction_fitness_from_rates --------------------------------------


def test_prediction_fitness_zero_for_identical_uniform_rates() -> None:
    """Uniform identical rates give fitness == 0."""
    t = 1000
    rate_a = jnp.full((t,), 0.05, dtype=jnp.float32)
    rate_b = jnp.full((t,), 0.05, dtype=jnp.float32)
    fit = b9._prediction_fitness_from_rates(rate_a, rate_b, t)
    assert abs(fit) < 1e-12


def test_prediction_fitness_rejects_t_not_multiple_of_window() -> None:
    """T not divisible by WINDOW_STEPS cannot be reshaped → AssertionError."""
    rate_a = jnp.zeros((150,), dtype=jnp.float32)
    rate_b = jnp.zeros((150,), dtype=jnp.float32)
    with pytest.raises(AssertionError, match="WINDOW_STEPS"):
        b9._prediction_fitness_from_rates(rate_a, rate_b, 150)


def test_prediction_fitness_rejects_t_equal_to_window_steps() -> None:
    """T == WINDOW_STEPS => n_windows == 1 => empty diff. Guard fires."""
    t = b9.WINDOW_STEPS
    rate_a = jnp.zeros((t,), dtype=jnp.float32)
    rate_b = jnp.zeros((t,), dtype=jnp.float32)
    with pytest.raises(AssertionError, match="n_windows"):
        b9._prediction_fitness_from_rates(rate_a, rate_b, t)


# ----- _setup_for_seed -----------------------------------------------------


def test_setup_for_seed_returns_traces_of_right_shape() -> None:
    """Setup builds per-step scalar traces of length t_steps."""
    t = 4000
    setup = b9._setup_for_seed(0, t)
    initial_state, a_is_inh, b_is_inh, i_ext_a, i_ext_b, valence, adr_a = (
        setup
    )
    assert i_ext_a.shape == (t,)
    assert i_ext_b.shape == (t,)
    assert valence.shape == (t,)
    assert adr_a.shape == (t,)
    assert a_is_inh.shape == (b9.N_NEURONS,)
    assert b_is_inh.shape == (b9.N_NEURONS,)
    assert initial_state.a.pool.pre_ids.shape == (b9.N_NEURONS, b9.K_SLOTS)
    assert initial_state.b.pool.pre_ids.shape == (b9.N_NEURONS, b9.K_SLOTS)


def test_setup_for_seed_rejects_t_not_multiple_of_drive_profile() -> None:
    """A drive profile has 4 levels, so t must be divisible by 4."""
    with pytest.raises(AssertionError, match="A drive profile"):
        b9._setup_for_seed(0, 5)


# ----- _conditions_for_seed_t ----------------------------------------------


def test_conditions_for_seed_t_all_when_completed_empty() -> None:
    """No prior runs => all 2 conditions remain (open_loop + gain=200)."""
    todo = b9._conditions_for_seed_t(0, 10_000, set())
    assert todo == [b9.OPEN_LOOP_LABEL, "gain=200"]


def test_conditions_for_seed_t_subset_when_partially_completed() -> None:
    """Open_loop done => only gain=200 remains."""
    completed = {(0, 10_000, b9.OPEN_LOOP_LABEL)}
    todo = b9._conditions_for_seed_t(0, 10_000, completed)
    assert todo == ["gain=200"]


def test_conditions_for_seed_t_empty_when_all_completed() -> None:
    """Both conditions done => empty list."""
    completed = {
        (0, 10_000, b9.OPEN_LOOP_LABEL),
        (0, 10_000, "gain=200"),
    }
    assert b9._conditions_for_seed_t(0, 10_000, completed) == []


def test_conditions_for_seed_t_does_not_match_other_seed_t() -> None:
    """Completed entries for other (seed, T) tuples do NOT mask todo."""
    completed = {(0, 10_000, b9.OPEN_LOOP_LABEL)}
    todo = b9._conditions_for_seed_t(37, 100_000, completed)
    assert todo == [b9.OPEN_LOOP_LABEL, "gain=200"]


# ----- _evaluate_one (GPU-running; smallest T to bound test runtime) -------


def test_evaluate_one_runs_only_open_loop_when_requested() -> None:
    """When only OPEN_LOOP_LABEL is in todo, only it is computed.

    Pins the observable contract: open-loop fitness must be a finite
    non-positive float (it is ``-mean((b - a)**2)`` over windows; -inf
    or NaN signals a broken sim, positive signals a bug in the sign
    convention).
    """
    import math  # pylint: disable=import-outside-toplevel
    results = b9._evaluate_one(0, 10_000, [b9.OPEN_LOOP_LABEL])
    assert set(results.keys()) == {b9.OPEN_LOOP_LABEL}
    fitness, wall = results[b9.OPEN_LOOP_LABEL]
    assert isinstance(fitness, float)
    assert math.isfinite(fitness)
    assert fitness <= 0.0
    assert wall > 0.0


def test_evaluate_one_empty_todo_returns_empty_dict() -> None:
    """Empty todo => empty results (no sim run)."""
    results = b9._evaluate_one(0, 10_000, [])
    assert not results
