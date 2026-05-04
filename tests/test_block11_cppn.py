"""Behavioral tests for Block 11 CPPN-GA N=20 sweep.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Block 11 is a thin orchestration layer over step 11's existing
``_evolve`` GA: it runs N=20 independent GA seeds across two
conditions (open_loop, closed_loop), records each GA's final best
fitness to a resumable CSV, and produces 40 datapoints for the
distribution-of-GA-best comparison vs step 10's hand-wired
closed-loop result.

Tests pin sweep-grid invariants and the CSV resume contract. The
GA loop itself is exercised through step 11's tests (which are
not part of the gated tests/ suite, but step 11's ``_evolve`` has
been load-tested in production - step 11 itself has run cleanly
on multiple machines).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import jax
import pytest

# pylint: disable=import-error
import block11_cppn_n20 as b11
import step11_cppn_closedloop as s11

# pylint: disable=protected-access


# ----- Sweep-grid invariants -----------------------------------------------


def test_n_gas_is_20() -> None:
    """The whole reason for Block 11: N=20 independent GAs per condition."""
    assert b11.N_GAS == 20


def test_conditions_are_open_and_closed_loop() -> None:
    """Two conditions; resume keys depend on these exact strings."""
    assert b11.CONDITIONS == ("open_loop", "closed_loop")


def test_pop_size_matches_step11_default() -> None:
    """pop=32 matches step 11's default; changing means a different GA."""
    assert b11.POP_SIZE == 32


def test_n_generations_matches_step11_default() -> None:
    """gens=30 matches step 11's default."""
    assert b11.N_GENERATIONS == 30


def test_expected_total_matches_grid() -> None:
    """20 GAs x 2 conditions = 40 total rows."""
    assert b11._expected_total() == 40
    assert b11._expected_total() == b11.N_GAS * len(b11.CONDITIONS)


# ----- _completed_pairs ----------------------------------------------------


def test_completed_pairs_returns_empty_for_missing_file(
    tmp_path: Path,
) -> None:
    p = tmp_path / "no_such_file.csv"
    assert b11._completed_pairs(p) == set()


def test_completed_pairs_reads_existing_rows(tmp_path: Path) -> None:
    """Each row's (ga_seed, condition) tuple lands in the set."""
    p = tmp_path / "rows.csv"
    p.write_text(
        "ga_seed,condition,best_fitness,wall_sec\n"
        "0,open_loop,-1.500000e-04,55.10\n"
        "0,closed_loop,-5.200000e-05,67.20\n"
        "1,open_loop,-1.520000e-04,54.30\n"
    )
    completed = b11._completed_pairs(p)
    assert (0, "open_loop") in completed
    assert (0, "closed_loop") in completed
    assert (1, "open_loop") in completed
    assert len(completed) == 3


# ----- _append_row ---------------------------------------------------------


def test_append_row_creates_file_with_header(tmp_path: Path) -> None:
    p = tmp_path / "new.csv"
    b11._append_row(p, 0, "open_loop", -1.5e-4, 55.1)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0] == ["ga_seed", "condition", "best_fitness", "wall_sec"]
    assert rows[1] == ["0", "open_loop", "-1.500000e-04", "55.10"]


def test_append_row_appends_without_duplicating_header(
    tmp_path: Path,
) -> None:
    p = tmp_path / "appended.csv"
    b11._append_row(p, 0, "open_loop", -1.5e-4, 55.1)
    b11._append_row(p, 0, "closed_loop", -5.2e-5, 67.2)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert len(rows) == 3
    assert rows[1][1] == "open_loop"
    assert rows[2][1] == "closed_loop"


def test_append_row_writes_header_to_existing_empty_file(
    tmp_path: Path,
) -> None:
    """Same fix as Block 9/10 G10-iii: 0-byte file from a prior crashed
    run gets a header on the next append, not a silently-misinterpreted
    first data row on resume."""
    p = tmp_path / "empty_then_append.csv"
    p.touch()
    assert p.stat().st_size == 0
    b11._append_row(p, 0, "open_loop", -1.5e-4, 55.1)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0] == ["ga_seed", "condition", "best_fitness", "wall_sec"]
    assert rows[1][0] == "0"
    assert (0, "open_loop") in b11._completed_pairs(p)


def test_append_row_then_completed_pairs_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "rt.csv"
    b11._append_row(p, 7, "closed_loop", -3.5e-5, 68.0)
    completed = b11._completed_pairs(p)
    assert (7, "closed_loop") in completed


# ----- _log ----------------------------------------------------------------


def test_log_writes_timestamped_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "block11.log"
    monkeypatch.setattr(b11, "LOG_PATH", log_path)
    b11._log("hello block 11")
    out = capsys.readouterr().out
    assert "hello block 11" in out
    assert out.startswith("[")
    assert "+00:00]" in out
    contents = log_path.read_text()
    assert "hello block 11" in contents


# ----- _gas_to_run ---------------------------------------------------------


def test_gas_to_run_all_when_completed_empty() -> None:
    """No prior runs => all (ga_seed, condition) pairs remain."""
    todo = b11._gas_to_run(set())
    assert len(todo) == 40
    # Sanity: contains the corner cases.
    assert (0, "open_loop") in todo
    assert (19, "closed_loop") in todo


def test_gas_to_run_subset_when_partially_completed() -> None:
    completed = {(0, "open_loop"), (0, "closed_loop"), (1, "open_loop")}
    todo = b11._gas_to_run(completed)
    assert (0, "open_loop") not in todo
    assert (0, "closed_loop") not in todo
    assert (1, "open_loop") not in todo
    assert (1, "closed_loop") in todo
    assert len(todo) == 40 - 3


def test_gas_to_run_empty_when_all_completed() -> None:
    completed = {
        (g, c) for g in range(b11.N_GAS) for c in b11.CONDITIONS
    }
    assert not b11._gas_to_run(completed)


# ----- Cross-file coupling regression test (s11._evolve signature) ---------


def test_run_one_ga_calls_s11_evolve_with_expected_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the load-bearing kwargs Block 11 passes to s11._evolve.

    Block 11 imports step11_cppn_closedloop's private ``_evolve`` and
    calls it with named arguments (closed_loop, pop_size,
    n_generations, seed) and three positional structural arguments
    (scenario, a_is_inh, b_is_inh). If a step 11 refactor renames an
    arg or drops a default, this test fails at gate time rather than
    silently breaking the next 40-hour Block 11 run.

    Captures the call args via a stub, returns a deterministic
    (genome, fitness) pair so _run_one_ga's coercion is exercised
    too.
    """
    captured: dict[str, Any] = {}

    def stub_evolve(
        label: str,
        scenario: tuple[Any, ...],
        a_is_inh: jax.Array,
        b_is_inh: jax.Array,
        *,
        closed_loop: bool,
        pop_size: int,
        n_generations: int,
        seed: int,
    ) -> tuple[None, float]:
        captured["label"] = label
        captured["scenario_present"] = scenario is not None
        captured["a_is_inh_shape"] = a_is_inh.shape
        captured["b_is_inh_shape"] = b_is_inh.shape
        captured["closed_loop"] = closed_loop
        captured["pop_size"] = pop_size
        captured["n_generations"] = n_generations
        captured["seed"] = seed
        return None, -1.234e-04

    monkeypatch.setattr(s11, "_evolve", stub_evolve)
    best_fit, wall = b11._run_one_ga(7, "closed_loop")

    assert captured["label"] == "ga7/closed_loop"
    assert captured["scenario_present"] is True
    assert captured["a_is_inh_shape"] == (s11.N_NEURONS,)
    assert captured["b_is_inh_shape"] == (s11.N_NEURONS,)
    assert captured["closed_loop"] is True
    assert captured["pop_size"] == b11.POP_SIZE
    assert captured["n_generations"] == b11.N_GENERATIONS
    assert captured["seed"] == 7
    # Float coercion: even though stub returned a Python float, pin
    # the contract so a future stub returning a JAX scalar would still
    # round-trip through _append_row's "{:.6e}" formatter.
    assert isinstance(best_fit, float)
    assert best_fit == pytest.approx(-1.234e-04)
    assert wall >= 0.0


def test_run_one_ga_open_loop_passes_closed_loop_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The condition string "open_loop" must map to closed_loop=False."""
    captured: dict[str, Any] = {}

    def stub_evolve(
        label: str,
        scenario: tuple[Any, ...],
        a_is_inh: jax.Array,
        b_is_inh: jax.Array,
        *,
        closed_loop: bool,
        pop_size: int,
        n_generations: int,
        seed: int,
    ) -> tuple[None, float]:
        del label, scenario, a_is_inh, b_is_inh
        del pop_size, n_generations, seed
        captured["closed_loop"] = closed_loop
        return None, 0.0

    monkeypatch.setattr(s11, "_evolve", stub_evolve)
    b11._run_one_ga(0, "open_loop")
    assert captured["closed_loop"] is False
