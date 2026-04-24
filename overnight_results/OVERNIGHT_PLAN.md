# Overnight batch plan — session handoff

This file captures the overnight-batch design so a fresh Claude Code
session (without the preceding conversation in context) can pick it
up and launch.

## TL;DR for the next Claude

**Your job:** launch `.venv/bin/python experiments/overnight_batch.py`,
ideally in the background so the session doesn't block, then monitor
progress via `tail -f overnight_results/batch.log` and
`tail overnight_results/block*.csv`. Every block commits + pushes its
results on completion, so if you crash mid-block the prior blocks are
already on GitHub.

**Permissions:** Ed is re-entering the session with
`--dangerously-skip-permissions`, so you should not need any manual
approvals. If you DO hit a prompt, STOP and log the issue in
`overnight_results/batch.log` rather than retrying forever. The most
likely trip points are novel Bash patterns we didn't anticipate.

## What's in this directory

- `batch.log` — time-stamped narrative of every run. Tail this for
  progress.
- `block1_step17_factorial.csv` — per-run metrics for block 1
- `block2_acquisition_probability.csv` — block 2
- `block3_long_training.csv` — block 3
- `block4_step16_multiseed.csv` — block 4
- `block5_best_config_confirm.csv` — block 5 (20-seed final
  confirmation at the best operating points)

Each CSV has a header row and one row per `(config_name, seed)`. The
runner is resumable: it skips rows already present before appending.
So if you restart the batch, it picks up where it left off.

## The blocks

### Block 1: Step 17 factorial multi-seed (~2 h)

72 configs + 2 baselines, 5 seeds each = 370 runs.

Factorial axes (3×2×3×2×2 = 72):
- `acq_mode` ∈ {stochastic, periodic, valence_gated}
- `pre_id_source` ∈ {uniform, hebbian}
- `acq_initial_v` ∈ {0.2, 1.0, 1.8}
- `release_threshold` ∈ {0.05, 0.2}
- `release_duration` ∈ {200, 1000}

Baselines: `baseline_stdp_only` (no structural plasticity) and
`baseline_release_only` (release on, acquisition off).

### Block 2: Acquisition-probability log-sweep (~0.5 h)

9 values of `ACQ_PROB_STOCHASTIC` from 1e-5 to 1e-1, 5 seeds each = 45
runs. Monkey-patches `step17_structural_growth.ACQ_PROB_STOCHASTIC`
at each iteration. All other params at the default operating point
(stochastic, uniform, init_v=0.2, thr=0.05, dur=500).

### Block 3: Long-training probe (~1.5 h)

5 durations × 2 conditions × 5 seeds = 50 runs.
- Durations: 20k, 50k, 100k, 200k, 500k
- Conditions: structural-on (step 17 default op point), STDP-only
  (step 17 baseline_stdp_only config)

### Block 4: Step 16 multi-seed retrofit (~0.5 h)

5 plasticity rates × 3 init_v_mean × 5 seeds = 75 runs.
- rates: 0.01, 0.1, 0.3, 1.0, 3.0
- init_v_mean: 0.2, 1.0, 1.8

Retrofits the single-seed step 16 findings to multi-seed.

### Block 5: Best-config 20-seed confirmation (~1 h)

3 configs × 20 seeds = 60 runs. Final tight bound on the headline
numbers for:
1. `baseline_stdp_only` (pure STDP, no structural plasticity)
2. `stochastic_uniform_iv0.2_thr0.05_dur500` (default structural op)
3. `stochastic_hebbian_iv0.2_thr0.05_dur500` (Hebbian acquisition)

## Expected total runtime

~5.5 hours if nothing fails. Slack budget: 2.5 hours.

## Smoke test already passed

- 1 config × 1 seed completed in 23 s and auto-committed.
- Result: `baseline_stdp_only` fit goes -2.586e-4 → -2.214e-4
  (matches step 16's seed=0 rate=1.0 run from yesterday — sanity).
- CSV + batch.log both written correctly.
- `git push origin main` succeeded from inside the runner.

## If things go wrong

1. **A run crashes mid-batch**: the per-run try/except catches it,
   logs the traceback to `batch.log`, and the next run proceeds.
2. **A whole block fails at top level**: same story — the outer
   try/except in `main()` catches it and moves to the next block.
3. **The process dies completely** (OOM, kernel panic, power hiccup):
   the CSVs are up-to-date through the last completed run because
   rows are flushed on every write. Re-launching the runner skips
   completed (config, seed) pairs and resumes.
4. **GPU thermal throttling**: each run is short (~10-30 s), and
   there's natural GC + `jax.block_until_ready` points. If
   throttling becomes severe you'll see `wall_sec` inflate in the
   logs.

## Context the runner was built on

- Step 16 (committed): STDP-only learning. Found fitness improves
  ~14% at plasticity_rate=1.0, but topology floor prevents reaching
  step 10's hand-wired -5.60e-5.
- Step 17 (local, uncommitted): adds slot acquisition in
  `src/silicritter/structural.py::apply_acquisition`. 10 new tests
  at 100% branch coverage. Experiment script and runner not yet
  committed at the time of handoff — they're in the working tree.

## Files uncommitted at handoff

- `experiments/step17_structural_growth.py` (new, linted clean)
- `experiments/overnight_batch.py` (new, linted clean)
- `src/silicritter/structural.py` (modified for acquisition)
- `tests/test_structural.py` (modified for new tests)

The next session should commit these as "Step 17: slot acquisition
and overnight sweep infrastructure" before launching the batch, so
the overnight commits have a clean starting point.

## Launch sequence (for the next Claude)

```bash
# 1. Verify tree is ahead of origin by the uncommitted step 17 work
git status

# 2. Commit step 17 + runner (see Files uncommitted above)
git add src/silicritter/structural.py tests/test_structural.py \
        experiments/step17_structural_growth.py \
        experiments/overnight_batch.py
git commit -m "Step 17: slot acquisition + overnight batch scaffolding"
git push origin main

# 3. Full test suite sanity check (should be 126 tests, 100% branch)
.venv/bin/pytest tests/ -q

# 4. Launch overnight runner in the background; monitor via tail
.venv/bin/python experiments/overnight_batch.py \
    > overnight_results/stdout.log 2>&1 &
echo "launched pid=$!"

# 5. Monitor
tail -f overnight_results/batch.log
```

## Morning review checklist

When Ed wakes up:
1. Check `overnight_results/batch.log` for the final
   "ALL BLOCKS COMPLETE" line. If absent, find where it stopped.
2. Review `git log` since handoff — each block commits its CSV,
   so the log tells the narrative.
3. Aggregate findings into `perf_history.md`. One entry per block
   is too much; one entry per informative finding is right.
4. Tally failure modes (runaway / dithering / collapse) from
   Block 1's factorial.
5. Find the Pareto frontier in Block 2's acq-prob curve.
6. Decide next step based on the data.
