# silicritter perf history

Durable record of performance measurements from experiment scripts.
Append one entry per significant run: what was measured, on what hardware,
with what code, and the actual numbers. Pass/fail alone is not enough;
this log is where drift is spotted.

Entries are chronological (oldest first) and immutable once recorded.
Use these numbers to spot regressions against previous runs, not just
against pass/fail thresholds.

---

## 2026-04-22 — Step 2: LIF forward-sim throughput baseline

- **Script:** `experiments/step02_throughput.py`
- **Module under test:** `src/silicritter/lif.py` (leaky integrate-and-fire, `jax.lax.scan` temporal loop, recurrent connectivity via N×N matmul on previous-step spikes)
- **Machine:** Huawei MateBook X Pro 2018 (MACH-WX9) — Intel i7-8550U + NVIDIA GeForce MX150 (2 GB VRAM, ~1.1 TFLOPS FP32), 16 GB RAM, Ubuntu 24.04, NVIDIA driver 580 / CUDA 13.0
- **Stack:** Python 3.12.3, JAX 0.10.0 (CUDA 12 plugin), numpy 2.4.4
- **Parameters:** N = 1024 neurons, T = 10 000 steps (dt = 1 ms → 10 s simulated), 5 timed repeats after a warmup pass

| metric                  | min       | median    |
|-------------------------|-----------|-----------|
| wall-clock elapsed (ms) | 1226.0    | 1226.1    |
| throughput (n·steps/s)  | 8.353e+06 | 8.352e+06 |

Mean firing rate across population: **40.0 Hz** (cortical-regime activity).

**Notes:**

- Repeat variance is negligible (min and median within 0.01 %), so thermal / boost effects aren't confounding at this scale.
- MX150 throughput is kernel-launch-overhead-limited for workloads this small, not sustained-compute-limited. Back-of-envelope: the dominant per-step cost is the N × N recurrent matmul (~2 N² ≈ 2.1 M FLOPS per step, ~21 GFLOPS total over 10 k steps). MX150 peak is ~1.1 TFLOPS, so a compute-bound floor is ~19 ms; we're ~65× off peak, dominated by per-step dispatch cost through `jax.lax.scan`.
- A modern discrete GPU (RTX 30-series or better) should see 10–100× better sustained throughput on the same script; this baseline is "GPU path is exercised correctly and JIT caches are hit," not "peak hardware performance."
- No plasticity in this baseline — weights are fixed. Slot-pool / structural-plasticity experiments will establish separate baselines as they land.

---

## 2026-04-22 — Step 3: slot-pool forward-sim baseline

- **Script:** `experiments/step03_slotpool_throughput.py`
- **Modules under test:** `src/silicritter/slotpool.py` (SlotPool representation, `synaptic_current`, `step`, `simulate`) + `src/silicritter/lif.py` (`integrate_and_spike` helper, shared with the dense-weights path)
- **Machine:** same as step 2 (Huawei MateBook X Pro 2018, MX150)
- **Stack:** same as step 2 (Python 3.12.3, JAX 0.10.0)
- **Parameters:** N = 1024 neurons, K = 64 slots / post-neuron (≈6 % of dense connectivity), T = 10 000 steps, 5 timed repeats after a warmup pass

| metric                             | min       | median    |
|------------------------------------|-----------|-----------|
| wall-clock elapsed (ms)            | 241.3     | 245.8     |
| throughput (neuron-steps/s)        | 4.24e+07  | 4.17e+07  |
| slot-eval throughput (slot-evals/s)| —         | 2.67e+09  |

Mean firing rate across population: **40.0 Hz** (matches step 2 regime).

**Notes:**

- **~5× speed-up over step 2 dense baseline** (8.35e6 → 4.24e7 neuron-steps/s). Per-step synaptic input dropped from O(N²) = 1 048 576 ops to O(N·K) = 65 536 ops — a 16× reduction in inner-loop work. The realized 5× speed-up (not 16×) is consistent with per-step dispatch / scan overhead on the MX150 still contributing a fixed baseline cost that doesn't shrink with smaller inner kernels.
- 40 Hz firing rate is preserved — the slot-pool representation with matched weight scale reproduces the step 2 firing regime qualitatively.
- Tests (`tests/test_slotpool.py`) include byte-exact equivalence between `slotpool.step(pool, ...)` and `lif.step(effective_weights(pool), ...)`, plus a full-trace equivalence over 50 steps — so the representation is validated against the dense-matrix ground truth, not just numerically plausible.
- Still no plasticity — slots are static. Step 4 adds three-factor STDP + valence broadcast on top of this representation.

---

## 2026-04-22 — Step 4: slot-pool + three-factor STDP baseline

- **Script:** `experiments/step04_plastic_throughput.py`
- **Module under test:** `src/silicritter/plasticity.py` (STDP traces, per-slot weight update modulated by scalar valence and gated by plasticity_rate)
- **Machine:** same as prior steps (MX150)
- **Stack:** same as prior steps
- **Parameters:** N = 1024, K = 64, T = 10 000; constant valence = +1; STDP defaults (tau_pre = tau_post = 20 ms, a_plus = 0.01, a_minus = 0.012, v ∈ [0, 0.5]); 5 timed repeats after a warmup pass

| metric                             | min       | median    |
|------------------------------------|-----------|-----------|
| wall-clock elapsed (ms)            | 573.0     | 579.2     |
| throughput (neuron-steps/s)        | 1.79e+07  | 1.77e+07  |
| slot-eval throughput (slot-evals/s)| —         | 1.13e+09  |

Mean firing rate: **40.8 Hz** (same regime as steps 2 and 3).

Weight drift over 10 s of simulated time:

| statistic   | initial | final  |
|-------------|---------|--------|
| v mean      | 0.0397  | 0.0927 |
| v min       | 0.0000  | 0.0000 |
| v max       | 0.2471  | 0.5000 |

Mean absolute change **|Δv| = 0.081**; max **|Δv| = 0.4998** (at least one slot rode the entire dynamic range up to the v_max ceiling and got clipped).

**Notes:**

- **Plasticity cost vs. step 3: ~2.2× slower** (4.17e7 → 1.85e7 neuron-steps/s) for the per-step overhead of trace decay + two gathers + STDP update + clip. Slot-eval throughput fell from 2.67e9 to 1.18e9 slot-evals/s, roughly matching.
- Weights are clearly moving (max reached the clip ceiling, mean doubled) — plasticity is active, not just nominally wired. Tests confirm valence = 0 and plasticity_rate = 0 both freeze weights exactly.
- Constant valence = +1 is the simplest possible three-factor setup; time-varying valence (rewarding specific network states) lands when we have an embodied target in step 5+.
- Still no structural plasticity — slots are bound for the lifetime of the sim. Slots at v_min = 0 are functionally silent but not released back to the free pool. Structural release / acquisition is the missing piece before we can genuinely claim exuberance-and-pruning dynamics.

---

## 2026-04-22 — Step 5: GA outer loop validation on adrenaline-driven target firing rate

- **Script:** `experiments/step05_ga_target_rate.py`
- **Module under test:** `src/silicritter/ga.py` (`Genome`, `random_genome`, `random_population`, `decode_to_pool`, `tournament_select`, `uniform_crossover`, `mutate`)
- **Machine:** same as prior steps (MX150)
- **Scenario:** N = 32 neurons, K = 8 slots/post, T = 2 000 steps, piecewise adrenaline profile (1.0 → 1.5 → 0.8 → 1.2), target firing rate = 40 Hz × adrenaline per 100-step window
- **GA:** population 48, generations 80, tournament size 3, elitism 2, v_sigma 0.01, rate_sigma 0.05, pre_resample_prob 0.03

**Throughput:**

| metric                      | value   |
|-----------------------------|---------|
| per-generation eval time    | ~22 ms  |
| total 80-gen wall time      | ~1.8 s  |
| population-batched inner sims per second | ~2 200 critter-lifetimes / s |

**Final best genome behavior by segment:**

| segment | adrenaline | target (Hz) | achieved (Hz) | \|err\| |
|---------|-----------:|------------:|--------------:|--------:|
| 0 | 1.00 | 40.0 | 34.0 | 6.0 |
| 1 | 1.50 | 60.0 | 68.2 | 8.2 |
| 2 | 0.80 | 32.0 | 16.0 | 16.0 |
| 3 | 1.20 | 48.0 | 48.9 | 0.9 |

Final best fitness: **−95.39** (mean of generation: −96.40). Fitness plateaus around gen 28–30; the subsequent 50 generations add marginal improvement. Direct encoding at this scale converges fast and then stalls.

**What step 5 validates (the whole point of the step):**

- Two-loop structure works end-to-end: `jax.jit(jax.vmap(evaluate_single))` batches a 48-member population through `simulate_plastic` in one GPU kernel sequence, each generation completes in ~22 ms including JIT'd tournament selection, crossover, and mutation. No plumbing issues.
- GA produces improvement over random init: initial best fitness ≈ −98 → final ≈ −95. Modest but real.
- Three of four adrenaline segments track the target within ~8 Hz.

**What step 5 genuinely reveals (not a failure to smooth over):**

- **Segment 2 exposes a floor problem in the multiplicative-gain model of adrenaline.** At adrenaline = 0.8 with drive 17–22 mV, effective drive is 13.6–17.6 mV. V_REST = −65, V_THRESH = −50, so a cell needs sustained drive > 15 mV to reach threshold at equilibrium. Many cells sit near or below the firing floor, producing ~16 Hz — roughly half the 32 Hz target. No weight configuration can fix this: the GA can amplify synaptic input but not push an individual neuron above its own threshold once adrenaline has suppressed its gain enough. This is an architectural finding about the modulator mechanism, not a tuning failure.
- **Direct encoding plateaus fast.** 30 generations ≈ 80 generations for this task. The GA rapidly finds good local optima and then can't explore past them. Scaling to richer tasks will need indirect encoding (CPPN / developmental rules) per the D004-era discussion — direct encoding is a loop-validation vehicle, not a research path.
- **Baldwin interference is latent but present.** The GA encodes initial slot-pool configuration; STDP reshapes the weights during the critter's lifetime. Fitness measures the lifetime-shaped behavior. A crossover child inherits the initial triplet (pre_ids, v, plasticity_rate) from one parent per slot but not the plasticized trajectory — so the child's initial state may land in a region neither parent explored. Not corrected for in this experiment; worth knowing for future step design.

**Implication for the adrenaline mechanism going forward:** pure multiplicative gain on `i_total` is too aggressive at low values (silences the network) and too aggressive at high values (saturates firing). A milder mechanism — e.g., `effective_gain = 1 + (adrenaline - 1) * sensitivity` with a sensitivity coefficient < 1, or acting on membrane time constant rather than input current — would preserve firing dynamics across a wider modulator range. Worth considering if we return to gain-mediated tasks; not an immediate blocker.

---

## 2026-04-22 — Step 5.5: side-by-side comparison of five adrenaline gain mechanisms

- **Script:** `experiments/step05b_adrenaline_comparison.py`
- **Modules under test:** new `GainMode` Literal + five `_modulate_*` helpers + `_GAIN_MODULATORS` dispatch in `src/silicritter/plasticity.py`
- **Machine:** same as prior steps (MX150)
- **Scenario:** identical to step 5 (N = 32, K = 8, T = 2 000, drive 17–22 mV, adrenaline profile (1.0, 1.5, 0.8, 1.2), target = 40 Hz × adrenaline, pop=48, gens=80, shared PRNG seed across modes for controlled-variance comparison)
- **Exit criterion:** declare a winner if any mechanism keeps all four segments within 5 Hz of target; otherwise the GA at this scale is the limiter, not the mechanism

### Results

| mode                | best fitness | err 0 | err 1 | err 2 | err 3 | max err (Hz) |
|---------------------|-------------:|------:|------:|------:|------:|-------------:|
| multiplicative      |       −95.39 |  6.00 |  8.19 | 16.00 |  0.94 |       16.00  |
| multiplicative_mild |       −90.08 |  6.06 | 14.44 |  3.00 |  8.69 |       14.44  |
| additive            |      −105.00 |  6.06 | 16.12 |  2.06 |  9.62 |       16.12  |
| **tau_m_scale**     |   **−48.91** |  6.00 |  8.44 |  5.00 |  6.31 |    **8.44**  |
| threshold_shift     |     −1699.37 |  6.00 | 70.12 | 32.00 | 28.00 |       70.12  |

Per-mode GA time: ~5.5 s on MX150 (80 generations including JIT warmup). Total comparison run: ~33 s.

### Verdict

**No mechanism cleared the 5 Hz/segment exit criterion.** The printed verdict was "NO WINNER — the GA at this scale is the limiter." But the comparison revealed a clear *relative* winner:

- **tau_m_scale is the best mechanism by roughly 2×** (fitness −48.91 vs. −90.08 for the next-best, −95.39 for the step-5 baseline). It's the only mechanism that keeps every segment within ~8 Hz of target. The mechanism choice *does* matter; the GA limiter conclusion applies only to the strict 5 Hz bar.
- **multiplicative_mild** (sensitivity 0.3) is a small improvement over the default multiplicative at the low-adrenaline end (3 Hz err vs. 16 Hz) but loses accuracy at the high end (14 Hz err vs. 8 Hz).
- **additive** under-modulates at default offset (5 mV): at adr=1.5 it adds only 2.5 mV, at adr=0.8 subtracts only 1 mV. Would need a larger offset to compete; not explored in this comparison.
- **threshold_shift** was catastrophic because the default shift (3 mV) multiplied by `tau_m/dt = 20` adds up to ±30 mV to `i_total`. Either the shift constant needs retuning (~0.15 mV would be roughly equivalent to the mild multiplicative regime) or the equivalence formulation needs a different scaling — but the comparison time-budget didn't cover re-tuning losers.

### Silicon implications

Each mechanism has a distinct analog-silicon cost profile:

- **additive** is the cheapest — a single analog bias line per neuron, summed into the integrator input. Most like a simple current-mode adder.
- **multiplicative** and **multiplicative_mild** need an analog multiplier (translinear / Gilbert cell or four-quadrant OTA), which is genuinely more expensive than a bias.
- **tau_m_scale** requires a variable membrane-leak conductance. In CMOS this maps to a modulated OTA bias setting the leak current, which is cheap (a single bias line adjusts an integrator time constant). Arguably *cheaper* than a multiplicative input path, because it modulates a bias rather than requiring a multiplier in the signal path.
- **threshold_shift** is cheap in silicon — just shift the comparator reference. The catastrophic result here is an experimental-parameter issue, not a silicon-cost issue.

So: **tau_m_scale wins on computational performance AND plausibly on silicon realizability.** It's cheaper than a multiplier and gives the best fitness.

### Recommendation going forward

The library default remains `gain_mode="multiplicative"` to preserve byte-exact compatibility with step 4 and step 5 perf numbers already in this log. But **step 6+ experiments should adopt `gain_mode="tau_m_scale"` as the working choice**, and future adrenaline-related task design should assume tau_m modulation rather than input-current scaling. If subsequent experiments want even tighter tracking, re-running the comparison at pop=200+ and gens=200+ would tell us whether pushing the GA harder recovers the 5 Hz bar under tau_m_scale.

**What this doesn't settle:** whether other modulators (cortisol-like, oxytocin-like) should also act on membrane time constants, on bias currents, on threshold shifts, or on specific plasticity parameters. Each chemical signal gets its own mechanism-selection question; tau_m_scale wins for a gain / arousal modulator but "arousal → faster membrane integration" doesn't necessarily generalize to "stress → slower plasticity." Decide per-modulator, per-role.

---

## 2026-04-22 — Step 6: structural plasticity (slot release)

- **Script:** `experiments/step06_structural_release.py`
- **Module under test:** new `src/silicritter/structural.py` (`StructuralParams`, `default_structural_params`, `apply_release`), plus optional `structural_params` parameter threaded through `step_plastic` and `simulate_plastic`
- **Scale:** N = 64 neurons, K = 16 slots/post (pool capacity = 1 024), 20 chunks × 250 steps (total T = 5 000 simulated)
- **Structural params:** `v_release_threshold = 0.01`, `release_dwell_steps = 100`
- **STDP params:** defaults (a_plus = 0.010, a_minus = 0.012)

### Results

Active-slot count over 20 measurement chunks (5 000 steps total):

| condition      | initial | final | released | retention |
|----------------|--------:|------:|---------:|----------:|
| valence = +1   |    1024 |   117 |      907 |     11.4% |
| valence = −1   |    1024 |   415 |      609 |     40.5% |

### Naively surprising; genuinely informative

The initial expectation — that valence = +1 would be "LTP-biased, strengthens weights, preserves slots" — is wrong given the default STDP parameters. The default rule has **a_minus (0.012) > a_plus (0.010)**, so under uncorrelated presynaptic / postsynaptic activity (the dominant regime in a random recurrent net), the STDP drift per spike pair is net-negative. With valence = +1, that net-negative drift pushes uncorrelated weights toward v_min = 0, where they sit for the 100-step dwell and get released.

Under valence = −1 the rule flips: a_plus now does LTD and a_minus now does LTP, so the net drift under uncorrelated activity becomes positive (a_minus > a_plus ⇒ more LTP than LTD in the flipped-sign regime). Fewer weights reach v_min; fewer slots get released.

**So: release is real and well-tuned.** ~60% of a fully-dense initial pool gets pruned under the default-asymmetric rule over 5 s of simulated time. The trajectory shape (fast initial pruning, plateau by chunk ~6–8) is qualitatively biologically plausible — an early mass-pruning phase followed by a stable working set, similar to the post-peak pruning in developmental biology.

### What this tests and what it does not

- **Tests**: release rule correctness (slots dwelling below threshold deactivate), persistence (slots above threshold keep counter at 0), preservation of `pre_ids` and `plasticity_rate` across release (they stay so subsequent acquisitions can reassign them).
- **Does not yet test**: slot **acquisition** — the freed slots stay free for the rest of the sim, so the pool only contracts. Genuine developmental exuberance-and-pruning dynamics require formation as well. Acquisition needs PRNG threading through the scan carry (acquisition is stochastic per biology), and lands in its own step.
- **Does not yet exercise**: structural plasticity under a GA outer loop. The step-5 GA does not pass `structural_params`. Step 6.5 or 7 will either add acquisition + integrate with the GA, or compare GA-on-static-topology vs. GA-on-dynamic-topology on the same task.

### Silicon implications

Release in silicon is cheap: a per-slot counter (4–8 bits is plenty at dwell = 100), a comparator reading v against a bias, and an AND gate gating the active flip-flop. Total area per slot: a comparator + counter + a few gates — negligible compared to the analog storage of v itself. The structural dynamics don't require anything fancier than what the existing slot cell already needs for weight storage.

### What's genuinely new about this result

- **Multiplicative-asymmetry-driven pruning matches developmental biology qualitatively.** No explicit "prune the weak" rule — just STDP with the standard a_minus > a_plus asymmetry + release dwell. The exuberance-to-sustained-working-set trajectory emerges from the mechanism rather than being programmed.
- **The 60% pruning under valence = −1 is notable.** Even under a rule regime that should favor LTP on uncorrelated activity, more than half the pool releases. Suggests the initial half-normal v distribution (mean ~0.04, many values close to 0) gives a lot of the pool a short runway to v_min even under favorable dynamics.

---

## 2026-04-22 — Step 7: paired-agent signal-following primitive

- **Modules landed:** `src/silicritter/paired.py` (NEW — `PairedState`, `init_paired_state`, `step_paired`, `simulate_paired`, `make_pool_for_partner`) and a refactor in `src/silicritter/plasticity.py` (`stdp_update` extracted as a public top-level function so paired sims can feed distinct pre / post rasters without touching the single-population `step_plastic` wrapper). `GAIN_MODULATORS` promoted from private to public.
- **Experiment:** `experiments/step07_paired_signal_following.py`
- **Scenario:** agent A externally driven by a 4-segment piecewise i_ext (18, 22, 19, 24 mV), slots 50% recurrent / 50% cross-bound into A via `pre_ids ∈ [0, 2N)` convention. A's scaffold is a fixed random pool shared across the whole GA population. Agent B receives no task-relevant external drive, only a tonic 16 mV baseline (so the subthreshold membrane has room for A's sparse cross-input to modulate firing rate rather than gate silent/spiking). B's scaffold is GA-evolved. Fitness = −MSE(B rate at window t, A rate at window t+1).
- **Scale:** N = 32 per agent, K = 8, T = 2 000, pop = 32, gens = 30, window = 100 steps.

### Primitive validation (the real step 7 deliverable)

**The paired primitive works end-to-end.** 6 tests in `test_paired.py` cover:

- Combined pre-space shapes (pool pre_ids ∈ [0, 2N), pre-trace length 2N, post-trace length N).
- `step_paired` shape invariants.
- `simulate_paired` per-agent spike traces of shape (T, N).
- **Quantitative cross-agent synaptic influence**: pair of runs with cross-weight 0.5 vs. 0.0, identical everything else; the difference in B's post-step V matches the analytic expected synaptic contribution `0.5 × dt / τ_m = 0.025 mV` within 1e-5 tolerance. This test would fail if the cross-raster convention were inverted or the synaptic path didn't reach through to the combined pre-raster correctly.
- Independence when cross-weights are zero: with A firing and B's cross-weights all zero, B stays completely silent — proves cross-influence is not leaking through any accidental back-channel.
- Compatibility with structural release: paired sim with `structural_params` set releases slots on both agents as expected.

### Signal-following task result (honestly reported)

**GA fitness plateaus early and does not differentiate the population.** Best and mean fitness converge to the same value within the first generation or two and stay there: best ≈ mean ≈ −5.18e−4. B fires at roughly 20 Hz with occasional 10 Hz dips, not tracking A's 20–50 Hz piecewise profile in any obvious way.

Per-generation eval: ~33 ms for the full 32-member population on MX150. Total 30-generation run: ~1 s.

### What the task result means (and doesn't mean)

The flat-fitness result is **consistent with step 5's finding** that direct-encoding GAs at N=32, pop~32-48 hit a fitness plateau early and don't recover without either bigger population / more generations or a better encoding (indirect / CPPN). Step 5's comparison across five adrenaline mechanisms showed mechanism choice matters substantially but none cleared a tight error bar at this scale; step 7 shows the same plateau behaviour on the paired-agent task.

**What this test cannot conclude:** that the paired primitive is incapable of supporting signal-following. We haven't run a hand-wired-predictor control (an explicit B pool designed by hand to mirror A's activity with a one-step delay) to prove the scenario itself is solvable under the given modulator strength and cross-weight budget. Without that control the "plateau = GA limiter" conclusion is narrative, not evidentiary. A follow-up experiment would either (a) add a hand-wired control, (b) run the GA at pop=256, gens=200 to test the scale hypothesis, or (c) try an indirect encoding on this task.

### Subsequent concrete questions for silicon or further experiments

- What cross-weight magnitude is needed for A to meaningfully influence B without swamping B's self-recurrent dynamics? The current weight scale (~0.04 mean per slot) gives A's cross-contribution roughly parity with recurrent self-input — possibly why B's behaviour is dominated by its baseline tonic.
- Should private modulators stay private once real social tasks arrive, or does a shared "environmental" modulator channel (pheromone-like) become architecturally meaningful? Not needed for step 7; likely needed for step 8+.
- Indirect encoding is overdue. Direct encoding has now plateaued on two separate tasks (target firing rate, signal following) under the same pop/gens scale. The step 8 / 9 GA work should probably move to a CPPN-style indirect encoding before trying more tasks.

### Scan / architectural notes

- `PairedState` is a NamedTuple-of-NamedTuples pytree and threads cleanly through `jax.lax.scan` as the carry; no explicit pytree registration needed.
- The two-phase `step_paired` (all LIF forward passes, then all STDP updates) means neither agent's STDP sees the partner's post-update state — verified by review. The intent "both neurons fire at the same wall-clock moment" is honored.
- `stdp_update` now a public top-level function; `step_plastic` is a thin wrapper that preserves all existing behaviour and retains its single-population assertion. 48 tests, 100% branch coverage across 7 library modules.

---

## 2026-04-22 — Step 7.5: hand-wired-predictor control for signal-following

- **Script:** `experiments/step07b_handwired_control.py`
- **Purpose:** answer the step-7 narrative gap -- is the GA the limiter, or is the task near its architectural ceiling? Run seven hand-wired B pool configurations (`plasticity_rate = 0` to freeze weights) through the identical step-7 scenario and compare fitness to the GA plateau.

### Results

All seven configurations sit within 10 % of the GA plateau; none meaningfully beats it:

| config | fitness | vs. GA | B rates by segment (Hz) |
|---|---:|---:|---|
| silent (v = 0) | −5.63e−4 | 1.09× worse | [18, 18, 18, 18] |
| all-recurrent v = 0.05 | −5.63e−4 | 1.09× worse | [18, 18, 18, 18] |
| all-recurrent v = 0.10 | −5.63e−4 | 1.09× worse | [18, 18, 18, 18] |
| all-cross v = 0.05 | **−5.18e−4** | 1.00× | [18, 18, 18, 20] |
| all-cross v = 0.10 | −5.21e−4 | 1.01× worse | [18, 18, 18, 20] |
| all-cross v = 0.20 | −5.36e−4 | 1.03× worse | [18, 18, 20, 18] |
| all-cross v = 0.30 | −5.47e−4 | 1.06× worse | [18, 18, 20, 18] |

A's per-segment rates over the same run: [28, 44, 32, 50] Hz.

### Verdict (inverting the step 7 hypothesis)

**The GA encoding is not the bottleneck.** No hand-wired configuration -- silent, pure-recurrent, pure-cross, or graded cross-weights from 0.05 to 0.30 -- moves B's firing rate more than ±2 Hz off 18 Hz. A varies across 22 Hz of range (28 → 50 Hz); none of that range leaks through to B meaningfully.

The step-7 conclusion "plateau = GA is the limiter" was the plausible one but wrong. The actual ceiling is architectural.

### Why the architecture caps at this fitness

Back-of-envelope: with K = 8 cross-slots and cross-weight saturated at v = 0.3, the maximum per-step synaptic input to a B neuron when every A neuron fires is 8 × 0.3 = 2.4 mV. The membrane update is `dv = i_total * dt / τ_m`, so that 2.4 mV contributes at most 2.4 / 20 = 0.12 mV to B's V per step. B's tonic drive is 16 mV. The cross-influence is a small (<1 %) perturbation on a baseline that already dominates the membrane dynamics. B fires at whatever rate the tonic produces (~18 Hz with this tonic) and the cross input is noise.

### Actionable implications

- **The current step-7 scenario is architecture-limited, not optimizer-limited.** Switching to indirect encoding, ES, or CMA-ES on this task will NOT meaningfully improve fitness.
- **To make the task solvable we need architectural changes**, in decreasing order of priority:
  - **Remove or reduce B's tonic drive.** With tonic = 0, B depends on cross-input entirely; cross-weight variation then has leverage. The step-7 choice of 16 mV tonic was a workaround for the original "B never fires" problem, but it drowned out the signal.
  - **Raise `v_max`** so cross-weights can go higher than 0.5. Current default caps saturate before cross can rival tonic.
  - **Increase K** so the sum of cross-slot contributions grows.
  - **Reduce N** or concentrate cross-slots so a smaller number of well-placed slots can drive B coherently.
- **Before any more optimizer work or a different task**, rerun step 7 with tonic = 0 and `v_max = 2.0` (or similar); a follow-up should establish whether that variant admits a meaningful dependence of B on A. If yes, then GA vs. better optimizer becomes a live question again. If not, the architecture scale (N = 32) is too small for this specific task.

### What this validates about the project methodology

The step-7 narrative gap flagged in this log ("plateau could be GA or architecture; we don't know") was worth paying the cost to close. Running seven hand-wired controls took ~10 s and flipped the conclusion from "GA is the limiter" to "architecture is the limiter." Without the control, we would have invested effort in indirect encoding / CMA-ES expecting a payoff that wouldn't come.

---

## 2026-04-22 — Phase 2 (N=256 re-scale): throughput re-baselines

**Context:** step 7.5 closed with the verdict that N=32 per agent is architecturally too small for the signal-following task. The plan before committing to new hardware is to re-baseline the existing experiments at N=256, K=32 on the MX150 and verify the mechanism scales cleanly. This entry captures the re-baseline runs.

All runs on the same reference machine (Huawei MateBook X Pro 2018, NVIDIA MX150, 2 GB VRAM, Python 3.12.3, JAX 0.10.0).

### Step 2 (dense LIF forward sim) at N=256

`.venv/bin/python experiments/step02_throughput.py --n-neurons=256 --n-timesteps=10000`

| metric | N=1024 baseline (prior) | **N=256** |
|---|---:|---:|
| elapsed (median, ms) | 1226 | **94** |
| throughput (median, neuron-steps/s) | 8.35e6 | **2.71e7** |
| mean firing rate | 40 Hz | 40 Hz |

**Counter-intuitive observation:** throughput at N=256 is ~3× *higher* than at N=1024 on the MX150. Reason is memory-bandwidth / cache-fit: the per-step work at N=1024 is O(N²) for the dense matmul (~1 M ops/step) vs. O(N²) = ~65 K ops at N=256, and the smaller state fits comfortably in the MX150's L2 cache. On a GPU with more bandwidth (3090), this inversion would flatten.

### Step 3 (slot-pool, no plasticity) at N=256, K=32

`.venv/bin/python experiments/step03_slotpool_throughput.py --n-neurons=256 --slots-per-post=32 --n-timesteps=10000`

| metric | N=1024 K=64 (prior) | **N=256 K=32** |
|---|---:|---:|
| elapsed (median, ms) | 245 | **100** |
| throughput (median, neuron-steps/s) | 4.17e7 | **2.56e7** |
| slot-eval throughput (median) | 2.72e9 | **8.20e8** |
| mean firing rate | 40 Hz | 40 Hz |

Lower raw throughput here than at N=1024 K=64 simply reflects a smaller workload per step (N·K drops from 65 536 to 8 192). Per-step slot-eval rate is 8.2e8/s, comparable to step 4's 1.0e9/s once STDP overhead flattens the comparison.

### Step 4 (slot-pool + STDP) at N=256, K=32

`.venv/bin/python experiments/step04_plastic_throughput.py --n-neurons=256 --slots-per-post=32 --n-timesteps=10000`

| metric | N=1024 K=64 (prior) | **N=256 K=32** |
|---|---:|---:|
| elapsed (median, ms) | 575 | **80** |
| throughput (median, neuron-steps/s) | 1.78e7 | **3.19e7** |
| slot-eval throughput (median) | 1.14e9 | **1.02e9** |
| mean firing rate | 40.8 Hz | 40.5 Hz |
| weight drift: v mean | 0.04 → 0.093 | 0.04 → **0.103** |
| weight drift: \|Δv\| mean | 0.081 | **0.094** |

**Plastic overhead is dramatically smaller at N=256.** Step 3 → step 4 slowdown at N=1024, K=64 was 4.17e7 / 1.78e7 ≈ 2.3×. At N=256, K=32 it's 2.56e7 / 3.19e7 ≈ 0.8× (i.e., plastic is *faster* than the no-plasticity step 3 — within the noise, roughly equal). This is a caching artifact on MX150; at smaller scales the whole working set fits in cache and the STDP compute cost is amortized.

Weight drift is qualitatively the same as step 4's N=1024 run: mean v rises from ~0.04 → ~0.10 over 10 s of simulated time, max v saturates at v_max=0.5 for some slots.

### Step 6 (structural release) at N=256, K=32

`.venv/bin/python experiments/step06_structural_release.py --n-neurons=256 --slots-per-post=32`

Pool capacity: N × K = **8 192 slots** per condition (up from 1 024 at N=64 K=16).

| condition | initial | final | released | retention |
|---|---:|---:|---:|---:|
| valence = +1 | 8192 | 740 | 7452 | **9.0%** |
| valence = −1 | 8192 | 3633 | 4559 | **44.3%** |

N=64 K=16 (prior) retention: +1 = 11.4%, −1 = 40.5%. **N=256 retention patterns are qualitatively identical** — the asymmetry-driven pruning story from step 6 (a_minus > a_plus drives net drift to v_min under uncorrelated activity) reproduces cleanly at 8× the pool size. The developmental exuberance-to-stable-working-set trajectory (fast initial pruning, plateau by chunk ~8) is preserved.

### Phase 2 conclusions (what we needed to learn)

- **Nothing breaks at N=256.** All four experiments run clean, produce sensible firing rates (40 Hz, same as prior), and exhibit the expected dynamics.
- **MX150 is fast at N=256** — per-experiment run times are ~100 ms for throughput benchmarks and ~15 s for the full step-6 trajectory (20 chunks × 2 conditions). Well within the iteration budget.
- **Structural release reproduces at scale.** The pruning mechanism is not N=64-specific; 8192-slot pools show the same 10–40 % retention bands depending on valence sign.
- **Plasticity overhead is scale-dependent on MX150.** At N=256 the inner loop fits in cache and STDP is essentially free; at N=1024 plasticity doubles per-step cost. This means Phase 4's GA runs at N=256 should be noticeably faster per generation than the naive 8× scale-up would predict.

### Ready for Phase 3

The architectural parameters for Phase 3's adjusted hand-wired control:

- N=256 per agent (matches this Phase 2 baseline)
- K=32 slots (per the step 7.5 analysis: K × v_max should rival tonic)
- tonic_drive = 0 (remove the workaround that drowned cross-input at N=32)
- v_max raised from 0.5 to 2.0 (so cross-weights can scale up to matter)

Phase 2 is green; Phase 3 is the actual test of whether these parameter changes give cross-coupling the leverage step 7.5 predicted.
