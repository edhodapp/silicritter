"""Behavioral tests for Block 11b genome analysis + eval-seed re-eval.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Block 11b is a three-phase orchestration over Block 11's evolved
CPPN genomes:

  Phase 1: re-run Block 11's 40 GAs to capture evolved genomes
  Phase 2: decode each genome to a pool and write structural stats
  Phase 3: evaluate each genome at 100 different scenario eval_seeds

Tests pin the sweep-grid invariants, the resume contracts for both
the genome pickle (Phase 1) and the per-eval CSV (Phase 3), and the
cross-file coupling to step 11's _evolve and _evaluate_one (so a
step 11 refactor breaks Block 11b at gate time, not at run time).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

# pylint: disable=import-error
import block11b_genome_analysis as b11b
import step11_cppn_closedloop as s11

# pylint: disable=protected-access


# ----- Sweep-grid invariants -----------------------------------------------


def test_n_eval_seeds_is_100() -> None:
    """The whole reason for Phase 3: N=100 scenario eval_seeds per genome."""
    assert b11b.N_EVAL_SEEDS == 100


def test_eval_seeds_are_disjoint_from_training_scenario() -> None:
    """Training uses PRNGKey(777); eval seeds must NOT include 777,
    so every eval is on a novel scenario."""
    eval_seeds = [
        b11b.EVAL_SEED_BASE + i * b11b.EVAL_SEED_STRIDE
        for i in range(b11b.N_EVAL_SEEDS)
    ]
    assert 777 not in eval_seeds
    # Also: must be disjoint from Block 11's GA seeds 0..19 (the
    # population-init seed) so the variance characterization is
    # genuinely orthogonal to GA-init randomness.
    assert all(s >= 100 for s in eval_seeds)


def test_genomes_grid_matches_block11() -> None:
    """40 genomes = 20 ga_seeds x 2 conditions, matching Block 11."""
    pairs = b11b._genome_pairs()
    assert len(pairs) == 40
    assert (0, "open_loop") in pairs
    assert (19, "closed_loop") in pairs


def test_expected_total_evals() -> None:
    """40 genomes x 100 eval_seeds = 4000 evals total."""
    assert b11b._expected_total_evals() == 4_000


# ----- _genome_stats (pure-function topology characterization) -------------


def test_genome_stats_returns_expected_keys() -> None:
    """Stats dict has the 9 columns the CSV header expects."""
    rng = jax.random.PRNGKey(0)
    from silicritter.cppn import (  # pylint: disable=import-outside-toplevel
        random_cppn_genome,
    )
    genome = random_cppn_genome(s11.CPPN_HIDDEN_DIM, rng)
    stats = b11b._genome_stats(genome)
    assert set(stats.keys()) == {
        "cross_pct", "cross_e_pct", "cross_i_pct", "recurrent_pct",
        "v_mean", "v_std", "v_max", "v_mean_cross_e",
        "v_mean_recurrent",
    }


def test_genome_stats_percentages_sum_to_100() -> None:
    """cross_pct + recurrent_pct must sum to 100% (every slot is one
    or the other; pre_ids span [0, 2N) exhaustively)."""
    rng = jax.random.PRNGKey(7)
    from silicritter.cppn import (  # pylint: disable=import-outside-toplevel
        random_cppn_genome,
    )
    genome = random_cppn_genome(s11.CPPN_HIDDEN_DIM, rng)
    stats = b11b._genome_stats(genome)
    assert stats["cross_pct"] + stats["recurrent_pct"] == pytest.approx(
        100.0, abs=1e-6,
    )
    # cross_e + cross_i = cross (every cross slot is E or I).
    assert (
        stats["cross_e_pct"] + stats["cross_i_pct"]
    ) == pytest.approx(stats["cross_pct"], abs=1e-6)


def test_genome_stats_v_bounds() -> None:
    """v values are clamped to [0, V_MAX]; mean and max within range."""
    rng = jax.random.PRNGKey(1)
    from silicritter.cppn import (  # pylint: disable=import-outside-toplevel
        random_cppn_genome,
    )
    genome = random_cppn_genome(s11.CPPN_HIDDEN_DIM, rng)
    stats = b11b._genome_stats(genome)
    assert 0.0 <= stats["v_mean"] <= s11.V_MAX
    assert 0.0 <= stats["v_max"] <= s11.V_MAX
    assert 0.0 <= stats["v_std"]


# ----- _build_scenario_for_eval_seed ---------------------------------------


def test_build_scenario_for_eval_seed_returns_correct_shapes() -> None:
    """Same scenario shape as step 11's _build_scenario; only pool_a's
    RNGKey differs."""
    scenario = b11b._build_scenario_for_eval_seed(1000)
    pool_a, i_ext_a, i_ext_b, val_a, val_b, adr_a = scenario
    assert pool_a.pre_ids.shape == (s11.N_NEURONS, s11.K_SLOTS)
    assert i_ext_a.shape == (s11.N_TIMESTEPS, s11.N_NEURONS)
    assert i_ext_b.shape == (s11.N_TIMESTEPS, s11.N_NEURONS)
    assert val_a.shape == (s11.N_TIMESTEPS,)
    assert val_b.shape == (s11.N_TIMESTEPS,)
    assert adr_a.shape == (s11.N_TIMESTEPS,)


def test_build_scenario_for_eval_seed_varies_pool_a_with_seed() -> None:
    """Different eval_seeds produce different pool_a connectivity.

    Pin the contract that pool_a is the only seed-dependent part of
    the scenario (drives, valences, adrenaline_a are deterministic).
    """
    scenario_a = b11b._build_scenario_for_eval_seed(1000)
    scenario_b = b11b._build_scenario_for_eval_seed(2000)
    pool_a_a = scenario_a[0]
    pool_a_b = scenario_b[0]
    # pool_a's pre_ids are random draws → must differ between seeds.
    assert not bool(jnp.array_equal(pool_a_a.pre_ids, pool_a_b.pre_ids))
    # Drives match (deterministic).
    assert bool(jnp.array_equal(scenario_a[1], scenario_b[1]))


# ----- _completed_evals ----------------------------------------------------


def test_completed_evals_returns_empty_for_missing_file(
    tmp_path: Path,
) -> None:
    p = tmp_path / "no.csv"
    assert b11b._completed_evals(p) == set()


def test_completed_evals_reads_existing_rows(tmp_path: Path) -> None:
    p = tmp_path / "rows.csv"
    p.write_text(
        "ga_seed,condition,eval_seed,fitness,wall_sec\n"
        "0,open_loop,1000,-1.500000e-04,0.50\n"
        "0,open_loop,1037,-1.510000e-04,0.50\n"
        "5,closed_loop,2000,-5.500000e-05,0.70\n"
    )
    completed = b11b._completed_evals(p)
    assert (0, "open_loop", 1000) in completed
    assert (0, "open_loop", 1037) in completed
    assert (5, "closed_loop", 2000) in completed
    assert len(completed) == 3


# ----- _append_eval_row + _append_stats_row -------------------------------


def test_append_eval_row_creates_file_with_header(tmp_path: Path) -> None:
    p = tmp_path / "evals.csv"
    b11b._append_eval_row(p, 0, "open_loop", 1000, -1.5e-4, 0.5)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0] == [
        "ga_seed", "condition", "eval_seed", "fitness", "wall_sec",
    ]
    assert rows[1] == [
        "0", "open_loop", "1000", "-1.500000e-04", "0.50",
    ]


def test_append_eval_row_writes_header_to_existing_empty_file(
    tmp_path: Path,
) -> None:
    """G10-iii fix carry-over: 0-byte file from crashed prior run still
    gets a header on next append."""
    p = tmp_path / "empty.csv"
    p.touch()
    assert p.stat().st_size == 0
    b11b._append_eval_row(p, 0, "open_loop", 1000, -1.5e-4, 0.5)
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0][0] == "ga_seed"


def test_append_stats_row_creates_file_with_header(
    tmp_path: Path,
) -> None:
    p = tmp_path / "stats.csv"
    b11b._append_stats_row(p, 0, "open_loop", -1.5e-4, {
        "cross_pct": 50.0, "cross_e_pct": 40.0, "cross_i_pct": 10.0,
        "recurrent_pct": 50.0, "v_mean": 1.0, "v_std": 0.5,
        "v_max": 1.9, "v_mean_cross_e": 1.2, "v_mean_recurrent": 0.8,
    })
    rows = list(csv.reader(p.read_text().splitlines()))
    assert rows[0][0] == "ga_seed"
    assert rows[0][1] == "condition"
    assert "cross_pct" in rows[0]
    assert rows[1][0] == "0"


# ----- _log ----------------------------------------------------------------


def test_log_writes_timestamped_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "block11b.log"
    monkeypatch.setattr(b11b, "LOG_PATH", log_path)
    b11b._log("hello block 11b")
    out = capsys.readouterr().out
    assert "hello block 11b" in out
    assert out.startswith("[")
    contents = log_path.read_text()
    assert "hello block 11b" in contents


# ----- pickle round-trip for genome dict -----------------------------------


def test_save_load_genomes_round_trip(tmp_path: Path) -> None:
    """Genome dict pickled and re-loaded must compare equal element-wise."""
    rng = jax.random.PRNGKey(42)
    from silicritter.cppn import (  # pylint: disable=import-outside-toplevel
        random_cppn_genome,
    )
    g0 = random_cppn_genome(s11.CPPN_HIDDEN_DIM, rng)
    g1 = random_cppn_genome(s11.CPPN_HIDDEN_DIM, jax.random.PRNGKey(43))
    genomes = {(0, "open_loop"): g0, (1, "closed_loop"): g1}
    pkl_path = tmp_path / "genomes.pkl"
    b11b._save_genomes(pkl_path, genomes)
    assert pkl_path.exists()
    loaded = b11b._load_genomes(pkl_path)
    assert set(loaded.keys()) == set(genomes.keys())
    for k, expected in genomes.items():
        assert bool(jnp.array_equal(loaded[k].w1, expected.w1))
        assert bool(jnp.array_equal(loaded[k].w2, expected.w2))


# ----- Cross-file coupling regression tests --------------------------------


def test_capture_one_genome_calls_s11_evolve_with_expected_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same load-bearing kwargs as Block 11's regression test, plus
    Block 11b's slightly different responsibility (capture genome, not
    just fitness)."""
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
    ) -> tuple[Any, float]:
        del scenario  # captured shape via a_is_inh/b_is_inh below
        captured.update(
            label=label,
            closed_loop=closed_loop,
            pop_size=pop_size,
            n_generations=n_generations,
            seed=seed,
            a_is_inh_shape=a_is_inh.shape,
            b_is_inh_shape=b_is_inh.shape,
        )
        # Return a real CPPNGenome so decode paths don't NPE.
        # pylint: disable=import-outside-toplevel
        from silicritter.cppn import random_cppn_genome
        stub_genome = random_cppn_genome(
            s11.CPPN_HIDDEN_DIM, jax.random.PRNGKey(0),
        )
        return stub_genome, -1.234e-04

    monkeypatch.setattr(s11, "_evolve", stub_evolve)
    genome = b11b._capture_one_genome(7, "closed_loop")
    assert captured["closed_loop"] is True
    # Block 11b explicitly hardcodes 32/30 to match step 11's defaults.
    assert captured["pop_size"] == 32
    assert captured["n_generations"] == 30
    assert captured["seed"] == 7
    assert genome is not None
    assert hasattr(genome, "w1")
    assert hasattr(genome, "w2")


def test_evaluate_genome_at_calls_s11_evaluate_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the s11._evaluate_one call signature for Phase 3 evaluation."""
    captured: dict[str, Any] = {}

    def stub_evaluate_one(
        genome_b: Any,
        scenario: tuple[Any, ...],
        a_is_inh: jax.Array,
        b_is_inh: jax.Array,
        closed_loop: bool,
    ) -> jax.Array:
        captured.update(
            scenario_present=scenario is not None,
            a_is_inh_shape=a_is_inh.shape,
            b_is_inh_shape=b_is_inh.shape,
            closed_loop=closed_loop,
            genome_present=genome_b is not None,
        )
        result: jax.Array = jnp.float32(-2.5e-5)
        return result

    monkeypatch.setattr(s11, "_evaluate_one", stub_evaluate_one)
    from silicritter.cppn import (  # pylint: disable=import-outside-toplevel
        random_cppn_genome,
    )
    genome = random_cppn_genome(
        s11.CPPN_HIDDEN_DIM, jax.random.PRNGKey(0),
    )
    fitness, wall = b11b._evaluate_genome_at(
        genome, "closed_loop", 1000,
    )
    assert captured["closed_loop"] is True
    assert captured["genome_present"] is True
    assert isinstance(fitness, float)
    assert fitness == pytest.approx(-2.5e-5)
    assert wall >= 0.0


# ----- _eval_seeds_to_run --------------------------------------------------


def test_eval_seeds_to_run_all_when_completed_empty() -> None:
    """No prior runs => all 100 eval_seeds remain."""
    todo = b11b._eval_seeds_to_run(0, "open_loop", set())
    assert len(todo) == b11b.N_EVAL_SEEDS


def test_eval_seeds_to_run_subset_when_partially_completed() -> None:
    """Already-done eval_seeds are skipped."""
    completed = {
        (0, "open_loop", b11b.EVAL_SEED_BASE),
        (0, "open_loop", b11b.EVAL_SEED_BASE + b11b.EVAL_SEED_STRIDE),
    }
    todo = b11b._eval_seeds_to_run(0, "open_loop", completed)
    assert len(todo) == b11b.N_EVAL_SEEDS - 2
    assert b11b.EVAL_SEED_BASE not in todo


def test_eval_seeds_to_run_does_not_match_other_genomes() -> None:
    """Completed entries for other (ga_seed, condition) tuples don't
    mask todo for this one."""
    completed = {(5, "open_loop", b11b.EVAL_SEED_BASE)}
    todo = b11b._eval_seeds_to_run(0, "open_loop", completed)
    assert len(todo) == b11b.N_EVAL_SEEDS
