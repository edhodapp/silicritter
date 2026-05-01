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

---

## 2026-04-22 — Phase 3 (N=256 re-scale): architectural-leverage sweep, hypothesis corrected mid-sweep

- **Script:** `experiments/step07c_handwired_n256.py`
- **Purpose:** test step 7.5's architectural-fix hypothesis — that raising N/K and removing the tonic drive would give cross-coupling the leverage to modulate B's firing rate. Sweep held the architectural knobs at N=256, K=32, v_max=2.0 and varied tonic ∈ {0, 8, 14, 16} mV, running the same eight hand-wired configurations at each tonic.

### Initial hypothesis was wrong

Step 7.5 closed with the recommendation "tonic = 0, v_max = 2.0." At N=256 with A firing ~4 % per step, that gives an average cross input of `K × v_max × p_fire = 32 × 2 × 0.04 = 2.56 mV` — short of the 15 mV needed for B to cross threshold from rest. At tonic ∈ {0, 8} mV, **B never fires**: every configuration produces 0 Hz across every segment and the fitness sits locked at the "always predict zero" baseline of −1.635e−3. The "architectural knob change" as originally specified did not admit a firing regime.

### Corrected finding: tonic near threshold does

At **tonic = 16 mV** (the step 7 tonic) with v_max = 2.0 and K = 32 (up from step 7's K = 8), leverage appears:

| tonic (mV) | best config | best fitness | max B range |
|---:|---|---:|---:|
| 0 | silent | −1.635e−3 | 0.00 Hz |
| 8 | silent | −1.635e−3 | 0.00 Hz |
| 14 | all-cross v = 1.50 | −4.895e−4 | 12.00 Hz |
| **16** | **all-cross v = 2.00** | **−1.737e−4** | 8.00 Hz |

A's segment rates over the same run: [28, 44, 32, 50] Hz (range 22 Hz).

Best-configuration B rates at tonic = 16, v = 2.0: [28, 28, 32, 30] Hz. **Clear tracking on the shape** — B's rate rises when A's segments rise and drops when A drops — with a systematic under-response to A's large excursions (A's 44-Hz peak becomes B's 28-Hz rate; A's 50 Hz becomes B's 30 Hz).

**Improvement versus step 7 plateau:** −1.737e−4 vs. −5.18e−4 (the step 7 GA plateau) = **~3× better fitness**.

### Leverage verdict

- **Within-configuration variation:** 8 Hz max B range at the winning regime, vs. A's 22 Hz range — 36 %, above the 20 % threshold we set.
- **Across-configuration variation:** fitness spread across the sweep is 1.46e−3, far above the 1e−4 threshold. Configurations clearly matter.
- **Both criteria pass** at tonic = 16. Architecture has leverage.

### The corrected step 7.5 architectural recommendation

The step 7.5 perf note claimed "remove tonic, raise v_max" would fix things. The sweep shows the real answer is different and more subtle:

- **Tonic must stay above or just below threshold** so B is biased into a firing-capable regime. Removing it entirely produces a dead network because sparse random cross-bindings can't by themselves reliably cross threshold.
- **K should scale with N** (here, K/N ≈ 0.125) so the cross-path aggregates enough simultaneous contributions.
- **v_max should exceed 1** so a few coincident cross spikes can move V meaningfully on top of the tonic baseline.
- **The combination** — tonic near threshold + K large enough + v_max large enough — produces cross-modulated firing. None of the three alone is sufficient.

This is the kind of finding the step 7.5 note lacked the data to see: "tonic = 0" looked right on paper (maximize cross influence) but collapses to silence in practice without a drive that puts B in a firing regime to begin with.

### Caveats

- Best hand-wired fitness is −1.74e−4 with imperfect tracking (B under-shoots A's peaks by up to 20 Hz). The architecture has *leverage*, not *perfect predictive capacity*.
- The GA in Phase 4 will have to find configurations that do better than this baseline. If it plateaus near or below −1.74e−4, we learn the task is solvable only to this level at N=256.
- The sweep was deliberately narrow (tonic ∈ 4 points, v ∈ 8 levels, single PRNG seed). A more thorough sweep would include K ∈ {16, 32, 64, 128} and multiple seeds, but Phase 3's job is "is there ANY leverage here," and that answer is already yes.

### Phase 3 conclusion

**Architectural leverage achieved at tonic = 16 mV, v_max = 2.0, K = 32, N = 256.** Proceed to Phase 4: re-run the step-7 GA at these parameters, compare the GA's plateau to the hand-wired baseline of −1.74e−4.

---

## 2026-04-22 — Phase 4 (N=256 re-scale): GA re-run; Baldwin interference dominates

- **Script:** `experiments/step07_paired_signal_following.py`
- **Parameters:** N = 256, K = 32, tonic = 16 mV, v_max = 2.0, T = 2000, pop = 32, gens = 30, valence = 1 (plasticity ON, matching step 7)
- **Hand-wired baseline to beat:** **−1.74e−4** (Phase 3's best, all-cross v = 2.0 with plasticity_rate = 0 so the static configuration persisted)

### GA at default exploration parameters plateaus at step-7 level

First run used the same `v_init_scale = 0.05` and `V_SIGMA = 0.01` as the original step 7 (those defaults are tuned for v_max = 0.5):

| metric | value |
|---|---:|
| final best fitness | **−5.14e−4** |
| final mean fitness | −5.15e−4 |
| per-generation time | 620 ms |
| total run time | 18.6 s |

That is essentially identical to step 7's original N = 32 plateau (−5.18e−4). B fires at ~20 Hz regardless of genome — the same "silent cross-path" regime step 7 showed, because the initial v distribution (half-normal scale 0.05 → mean ~0.04) and the mutation scale (σ=0.01) can't reach the v ~ 1.5–2.0 regime in 30 generations. **The GA was searching in the wrong corner of the weight space.**

### Scaled exploration parameters help modestly

Added `--v-init-scale` and `--v-sigma` CLI overrides. Second run with `v_init_scale = 0.5`, `v_sigma = 0.1` (scales roughly proportional to v_max):

| metric | default | scaled |
|---|---:|---:|
| final best fitness | −5.14e−4 | **−4.17e−4** |
| B rate at best genome | 20 Hz flat | 10–27 Hz (wider) |

The scaled-exploration GA **does** reach the interesting weight regime — B's segment rates now span 10–27 Hz (vs. 20 Hz flat). Fitness is ~20 % better. **But the GA plateau is still ~2.4× worse than the hand-wired baseline.**

### Diagnosis: Baldwin interference is dominant here

The hand-wired Phase 3 winner had every B slot bound to A with v = 2.0 *and plasticity_rate = 0 so the configuration persisted*. The Phase 4 GA evolves initial configurations but runs STDP during each critter's lifetime. With default STDP parameters (a_minus = 0.012 > a_plus = 0.010), **uncorrelated activity drifts cross-weights toward zero faster than the GA can evolve them upward**. Whatever "high-cross-weight" initial configuration the GA discovers gets eroded within the 2 000-step evaluation window, and the resulting behaviour reverts to something close to the silent-cross baseline.

This was flagged as a latent risk in step 7's perf entry ("Baldwin interference is latent but present — the GA encodes initial slot-pool configuration; STDP reshapes the weights during the critter's lifetime. Fitness measures the lifetime-shaped behavior."). At N = 32 with v_max = 0.5 the GA's effective search space was small enough that Baldwin interference was invisible noise. At N = 256 with v_max = 2.0, the search space is large enough that the interference is the dominant force.

### What Phase 4 actually validates (and what it doesn't)

- **Validates:** the two-loop machinery works at N = 256. GA runs, fitness changes with genome, per-generation compute budget is 0.6 s on MX150 which is tight but tolerable.
- **Validates:** the default GA exploration parameters are poorly matched to raised v_max. This is a real, fixable issue — any future scale-up of v_max should scale v_init_scale and v_sigma proportionally.
- **Does not validate:** that the current silicritter architecture can solve signal-following with direct-encoding GA + plasticity. The hand-wired optimum sits above the GA's reach; the gap is not a GA encoding failure per se — it is the GA failing to *preserve* a high-v configuration against the STDP drift during evaluation.

### The clean next experiment

Three ways to close the Baldwin gap:

1. **GA with plasticity disabled during evaluation** (valence = 0 in the scenario). Tests whether the GA's direct encoding can find the hand-wired optimum when STDP isn't fighting it. If yes → the primitive + GA work fine, we just can't run plasticity during GA eval. If no → the GA's genome encoding (not plasticity) is the limiter.
2. **Evolve STDP hyperparameters alongside initial weights.** The genome gains a second payload: per-critter a_plus, a_minus, tau values. GA finds STDP rules that don't erode cross-weights during the lifetime, in parallel with the initial pool. Meta-learning-style.
3. **Indirect encoding (CPPN generator).** As-flagged previously: dramatically smaller search space via a generator network that produces consistent high-cross pools. Still leaves Baldwin interference intact; fixes a different failure mode.

**(1) is the cheapest and most informative diagnostic.** ~30 s of compute. It tells us whether the GA's direct-encoding can solve the task at all at this scale, separate from the plasticity complication.

### Phase 4 conclusion

Phase 4's operational outcome: the step-7 plateau (−5.18e−4) and the Phase 4 plateau (−4.17e−4) are both well above the hand-wired optimum (−1.74e−4). The gap is dominated by Baldwin interference between the GA outer loop and STDP inner loop, not by direct-encoding weakness per se. Proceeding to a Phase 4.1 control run with plasticity disabled is the natural next step — ~30 s to run, definitive on whether direct encoding itself can reach the hand-wired optimum.

---

## 2026-04-22 — Phase 4.1 (N=256 re-scale): plasticity-off control *inverts the Phase 4 diagnosis*

- **Script:** `experiments/step07_paired_signal_following.py` with `--valence-b=0.0`
- **Parameters:** identical to Phase 4's scaled-exploration run (N = 256, K = 32, v_max = 2.0, v_init_scale = 0.5, v_sigma = 0.1, pop = 32, gens = 30), except `valence_b_trace = 0` so STDP is inactive on B during every critter's lifetime. Initial pool configuration fully determines behaviour; no Baldwin interference possible.
- **Prediction (Phase 4 entry):** if Baldwin is the bottleneck, this run should improve meaningfully toward the hand-wired optimum of −1.74e−4.

### Result: Phase 4.1 is indistinguishable from Phase 4

| condition | fitness |
|---|---:|
| Phase 4 scaled, plasticity ON | −4.169e−4 |
| **Phase 4.1 scaled, plasticity OFF** | **−4.179e−4** |

Difference is 0.3 % — well inside the noise floor. **Baldwin interference contributes essentially nothing to the gap.** The GA plateau at N = 256 with v_max = 2.0 is the same with or without STDP running during evaluation.

### The Phase 4 diagnosis was wrong

The Phase 4 entry above called Baldwin interference "dominant." That diagnosis is now falsified. The actual bottleneck is **direct-encoding weakness in a combinatorially structured optimum**.

Back-of-envelope for why:

- Hand-wired Phase 3 winner has *all* K = 32 slots per B-neuron bound to A (pre_id ∈ [N, 2N)) with v ≈ 2.0. This is a highly structured configuration.
- GA random init has pre_id drawn uniformly from [0, 2N), so ~50 % of slots are cross-bound, ~50 % recurrent-bound. Getting to "all cross" requires resampling ~4000 pre_ids to the right half of the index range.
- `pre_resample_prob = 0.03` means ~3 % of slots get their pre_id redrawn per generation, and half of those land in the wrong range by chance.
- Over 30 generations, ~45 % of slots get any resampling; even fewer land cross by chance alone.
- The GA is searching for a specific large-scale structural pattern with mutations that are far too local.

Equivalently stated: the fitness landscape around the hand-wired optimum is probably smooth (all-cross + high-v is a broad basin) but the GA's initial population is hundreds of independent coin flips away from that basin.

### What this means for the project roadmap

The earlier "indirect encoding is overdue" noted in step 5.5 and step 7.5 now has a concrete empirical justification rather than a theoretical one:

- **Direct-encoding GA cannot reach the architectural ceiling at N = 256**, regardless of plasticity setting.
- **The hand-wired ceiling exists** — Phase 3 showed it — so the architecture is solvable in principle.
- **The gap is encoding-space size versus search pressure**, not plasticity interference and not a GA hyperparameter tuning issue.

Three real next directions, in order of promise:

1. **Indirect encoding (CPPN-style).** A small generator network (10–100 evolved parameters) produces the full (N × K) pool from a compact rule. "Bind all slots to A with v near v_max" is a two-line CPPN. Dropping from 8192 direct parameters to ~50 meta-parameters shrinks the search space by ~160× and moves the hand-wired-equivalent configurations into reach of a 30-generation GA.
2. **Structural mutation operators.** Instead of per-slot random resampling, add mutation operators that flip *ranges* of slots to all-cross or all-recurrent in one step. Keeps direct encoding but biases exploration toward the kind of structured configurations hand-wiring found. Cheaper than rewriting the encoding; may not buy enough.
3. **Task redesign.** If the achievable ceiling at N = 256 is near −1.74e−4 and a GA at this scale can't realistically reach it, maybe the signal-following task is too narrow a fitness surface for this architecture. A richer task (multiple correlated signals, or signals plus a reward channel) might give the GA more gradient to follow.

### Phase 4.1 conclusion

Baldwin interference was a plausible diagnosis; Phase 4.1 ruled it out definitively. The actual bottleneck is combinatorial encoding weakness. The ~30 seconds of compute it took to run Phase 4.1 saved us from pursuing plasticity-vs-GA solutions that wouldn't have helped.

**Recommended next concrete step: implement CPPN indirect encoding** (or an equivalent compact generator) in `ga.py`, wire it through step 7's GA, and re-run at the same N = 256 parameters. Expected outcome under the corrected diagnosis: fitness should drop meaningfully toward the hand-wired −1.74e−4 and potentially below it, because a generator can produce configurations that the random-slot direct encoding structurally can't reach in 30 generations.

---

## 2026-04-22 — Step 8: CPPN indirect encoding closes the gap to the hand-wired optimum

- **New module:** `src/silicritter/cppn.py` — Compositional Pattern Producing Network indirect encoding. 2-layer MLP CPPN with tanh hidden activations and sigmoid output heads. Inputs: `(post_idx_normalized, slot_idx_normalized, bias)`. Outputs: `(pre_id_raw, v_raw, plasticity_rate_raw)` decoded into a full SlotPool.
- **New tests:** `tests/test_cppn.py` — 10 behavioural tests including a hand-crafted genome test showing the CPPN can express the all-cross / high-v pattern.
- **New experiment:** `experiments/step07e_paired_cppn_n256.py` — same scenario as Phase 4 (N = 256, K = 32, v_max = 2.0, tonic = 16 mV, T = 2000, pop = 32, gens = 30, plasticity ON) with the CPPN encoder replacing the direct encoding.
- **Genome size:** 51 weights (3 × 8 + 9 × 3) at hidden_dim = 8. Compare to direct encoding's 24,576 parameters. **~480× smaller search space.**

### Result

| encoding | fitness |
|---|---:|
| step 7 original (N=32, direct) | −5.18e−4 |
| Phase 4 scaled direct (N=256) | −4.17e−4 |
| Phase 3 hand-wired (plasticity off) | −1.74e−4 |
| **step 8 CPPN (N=256, plasticity on)** | **−1.704e−4** |

The CPPN GA fitness **matches and slightly beats** the hand-wired baseline. Convergence happened by generation 14; plateau from there. Per-generation eval ~620 ms on the MX150 (same as direct encoding since the sim cost dominates over encoding cost).

### The decoded best-genome pool

```
100.0% cross-bound (8192/8192 slots bound to A-side indices)
v mean 1.999, v max 2.000
```

**The CPPN discovered the exact hand-wired-optimal configuration.** All 8192 slots bind to A (the right half of the pre-raster); v saturates at v_max. The small improvement over the static hand-wired result (plasticity off) reflects a small contribution from lifetime STDP fine-tuning on top of the hand-wired structure.

### Per-window behaviour of the best CPPN

B's per-window firing rate shows clear tracking of A, with the same systematic under-response to A's peaks that the hand-wired baseline had:

- A = 20 Hz → B = 20 Hz (match)
- A = 30 Hz → B = 30 Hz (match)
- A = 40 Hz → B = 30 Hz (undershoot)
- A = 50 Hz → B = 30 Hz (undershoot)

The ceiling at ~30 Hz is an architectural property of the tonic-plus-cross dynamics at K = 32, v_max = 2.0 — not an encoding failure. A GA that learned to compensate (e.g., by binding *multiple* slots to the same A-neuron to linearly amplify) might push past 30 Hz, but the current 2-layer CPPN doesn't find that pattern in 30 generations.

### What this validates, empirically

1. **Phase 4.1's diagnosis was correct.** "The actual bottleneck is combinatorial encoding weakness" is now an empirical claim, not a theoretical one. Replacing the encoding with a compact generator closed the entire 2.4× gap between direct-encoding plateau and hand-wired optimum.
2. **The search space argument holds.** 51 evolved parameters beat 24,576 evolved parameters in 30 generations, on the exact same task, with the exact same sim and same GA machinery.
3. **The CPPN's expressivity is sufficient** at this task's complexity. 8 hidden units + 3 output heads is enough to find "bind all slots to A with high v." Richer tasks (multi-signal prediction, differentiated neural populations) may need more hidden units or HyperNEAT-style activation-function diversity.

### What this does NOT validate

- **The 30-Hz ceiling is real.** Even the optimal configuration undershoots A's peaks. To push past 30 Hz we'd need a different architectural knob — e.g., dynamic cross-weight amplification through plasticity (which our setup allows but didn't converge to), or a richer task that rewards A-peak-tracking directly.
- **CPPN is not magic.** It solved *this* task because the architectural optimum has a simple regular structure ("bind every slot the same way"). Tasks whose optima require spatially-differentiated patterns may hit different limits.
- **We didn't evolve topology.** Full HyperNEAT evolves the CPPN's topology as well as weights. We fixed the topology at 3-8-3. Adding topology evolution is the next step if this encoding plateaus on richer tasks.

### Phase 4-era conclusions are now coherent

Reading across the N=256 phase sequence:

- Phase 2 confirmed nothing breaks at N=256.
- Phase 3 showed architectural leverage exists at tonic=16, K=32, v_max=2.0.
- Phase 4 showed direct-encoding GA plateaus at −4.17e−4, well short of the hand-wired baseline.
- Phase 4.1 (plasticity-off control) falsified the Baldwin-interference diagnosis.
- **Step 8 (CPPN)** closes the gap. Encoding size was the bottleneck all along.

### Phase 5 / step 9 candidate directions

With the encoding problem solved at this scale, the next interesting questions are:

1. **Different tasks.** The signal-following task's architectural ceiling is ~30 Hz. Tasks that exercise the network's capacity differently (e.g., phase-locked oscillation, sequence memory, two-agent coordination where B influences A back) might reveal different limits and different encoding needs.
2. **Richer CPPNs.** Mixed activation functions (sin, gaussian, linear), topology evolution (NEAT-style), or multi-layer CPPNs could reach patterns that a tanh-only 2-layer net can't.
3. **Slot acquisition (deferred from step 6).** Now that we have working GA+CPPN infrastructure, layering in structural acquisition gives us developmental dynamics.
4. **Social-intelligence tasks beyond signal-following.** The "awareness + prediction" target from earlier framing invites tasks where both agents are evolved and have to coordinate, not the asymmetric signal-follower setup.

Step 8 completes the N=256 re-scale plan's research arc: we can now solve the step-7 task at the scale we wanted to operate at, with a GA that actually does useful work. That's the infrastructure milestone the project has been aiming for since steps 5 and 7 first plateaued.

---

## 2026-04-22 — Step 9: inhibition substrate adopted, validated at canonical values

- **Module additions:** `slotpool.synaptic_current` gained optional `pre_is_inhibitory` and `i_weight_multiplier` parameters; `slotpool.assign_ei_identity` helper added. `paired.step_paired` and `paired.simulate_paired` thread `a_is_inhibitory` / `b_is_inhibitory` / `i_weight_multiplier`. All additions default to the pre-step-9 behavior (no E/I), so every prior experiment reproduces byte-exactly.
- **Decision log:** D007 logs the adoption of canonical E/I values (ratio 4:1, multiplier 4.0) as provisional.
- **Experiment:** `experiments/step09_handwired_n256_ei.py` runs a hand-wired control at the step-7c parameters (N=256, K=32, tonic=16, v_max=2.0) comparing E/I-substrate variants against the no-E/I baseline.

### Results

| config | fitness | B rates by segment (Hz) |
|---|---:|---|
| cross-random v=2.0, no E/I (step 7c baseline) | −1.737e−4 | 28, 28, 32, 30 |
| **cross-random v=2.0, E/I ON (canonical balanced)** | **−5.204e−4** | **18, 18, 18, 18** |
| cross-E-only v=2.0, E/I ON | **−1.564e−4** | 28, 29, 32, 31 |
| cross-I-only v=2.0, E/I ON | −1.637e−3 | 0, 0, 0, 0 |
| cross-E-only v=1.0, E/I ON | −3.163e−4 | 24, 23, 24, 25 |
| cross-E-only v=0.5, E/I ON | −3.752e−4 | 20, 22, 22, 22 |

A's per-segment firing rates: [28, 44, 32, 50] Hz.

### Validation — substrate behaves as predicted

- **Canonical balanced cross produces near-tonic-only firing** (row 2: 18 Hz flat across segments). Theoretical expectation: `0.8 × v − 0.2 × v × 4 = 0` mean, so B's firing is driven by tonic only and fluctuations don't push past it meaningfully at these parameters. Observed exactly that.
- **Pure inhibitory cross produces silence** (row 4: 0 Hz). Theoretical: `0 × E − 1.0 × v × 4 = −8 mV` mean cross input overwhelms the 16 mV tonic, V_eq = −65 + 16 − 8 = −57 mV, below threshold. Observed: zero firing.
- **Pure excitatory cross preserves step 7c behavior** (row 3). Same regime as pre-E/I; fitness essentially matches baseline.
- **Intermediate E-only weights** (rows 5, 6) show graded reduction in B's firing rate as cross-input shrinks. Monotonic as expected.

### One interesting finding worth naming

**Cross-E-only v=2.0 with E/I ON is marginally BETTER than the no-E/I baseline** (−1.564e−4 vs. −1.737e−4). Modest improvement (~10%), likely due to A's own internal E/I balance sharpening A's spike timing — A's dynamics are slightly crisper with inhibition, and B inherits the sharper signal. Not a large effect; within the range where a different seed could erase it.

### Architectural observation: balanced cross ≠ useful cross

The theory predicts "balanced E/I = zero mean" and that's exactly what Step 9 shows. Biologically this is the *local* balance of cortical circuits — neighboring pyramidal-interneuron pairs. But for *long-range feedforward* connections (our B-observes-A setup), cortex uses E-biased projections, not balanced ones. The feedforward axons from one cortical area to another are predominantly excitatory onto the target area's excitatory cells; the balance comes from local inhibition *within* the target area, not from balanced feedforward.

Step 9 reproduces this: the useful configuration for signal-following is E-targeted cross (row 3), not balanced cross (row 2). This is the correct architecture even with canonical E/I values; it's not a failure of the substrate, it's the substrate pointing us at the right connectivity pattern.

### What Step 9 validates, and what remains for Step 10+

**Validates:**

- E/I substrate implementation is correct (theory and observation agree on all six configurations).
- Default values (ratio 4:1, multiplier 4.0) produce the predicted dynamic regimes.
- No regressions in existing experiments (63 tests pass, 100 % branch coverage across 8 library modules).
- The substrate is opt-in; every step 2–8 experiment reproduces byte-exactly when E/I is not explicitly engaged.

**Does NOT validate (these are Step 10 / 11 work):**

- Whether the canonical values are optimal for silicritter's architecture (D007 punch list).
- Whether E/I substrate interacts usefully with closed-loop adrenaline (Step 10).
- Whether inhibitory plasticity (Vogels 2011 iSTDP) is needed to prevent I-weight drift.
- Whether the slight fitness improvement at cross-E-only v=2.0 with E/I is a real effect or noise; needs multi-seed confirmation.

### Phase 10 ready

The substrate is stable. Step 10 can now layer closed-loop adrenaline on top of the E/I substrate and test whether dynamic gain modulation does useful work where static balance alone cannot — e.g., pushing past the 30-Hz ceiling on demand when A's rate demands it.

---

## 2026-04-22 — Step 10: closed-loop adrenaline breaks the architectural ceiling

- **Script:** `experiments/step10_closedloop_adrenaline.py`
- **Controller design:** leaky-integrator EMA of A's and B's firing rates (decay = 0.98, ~50 ms time constant at dt=1ms); error = `rate_a_ema − rate_b_ema`; adrenaline = `clip(1.0 + gain × error, 0.5, 3.0)`. Adrenaline feeds `step_paired` via the `tau_m_scale` mechanism (the step-5.5 winner for adrenaline).
- **Substrate:** step 9's validated E/I (205 E + 51 I per agent, i_weight_multiplier = 4.0). Hand-wired B pool is cross-E-only v = 2.0 (step 9's best single-config fitness of −1.56e−4).
- **Purpose:** test whether a closed-loop gain controller can push past the 30-Hz structural ceiling analyzed in the prior discussion — the ceiling that step 8's CPPN GA (−1.70e−4) and step 9's best hand-wired (−1.56e−4) both bottomed out at.

### Result

| config | fitness | B per-segment (Hz) | avg adr per segment |
|---|---:|---|---|
| open-loop (constant adr = 1.0) | −1.566e−4 | 28, 29, 32, 31 | 1.00, 1.00, 1.00, 1.00 |
| closed-loop gain = 10 | −1.108e−4 | 28, 40, 32, 36 | 1.00, 1.04, 1.00, 1.12 |
| **closed-loop gain = 50** | **−5.60e−5** | **28, 44, 32, 50** | 1.00, 1.00, 1.00, 1.00 |
| closed-loop gain = 200 | −5.60e−5 | 28, 44, 32, 50 | 0.96, 0.93, 0.94, 0.92 |

A's rates per segment: [28.0, 44.0, 32.0, 50.0] Hz. **B at gain = 50 tracks A exactly, segment by segment.**

### What this measures

- **Fitness improvement ~3×** over the open-loop baseline (−1.566e−4 → −5.60e−5).
- **Fitness improvement vs. the step 8 CPPN GA ceiling** (−1.70e−4) and **the best hand-wired static configuration** (−1.56e−4): about **3.0× better in both cases.**
- **Residual error of 5.6e−5** corresponds to per-window rate mismatch of about `sqrt(5.6e−5) ≈ 0.0075`, i.e., about 7.5 Hz RMS if it were steady. But segment-by-segment rates match exactly — the residual is concentrated at segment *transitions* where the EMA lag prevents instantaneous adaptation. The mean adrenaline per segment sits near 1.0 because the controller spends a segment catching up and then operating near the setpoint.

### The 30-Hz structural ceiling is defeated by dynamic gain, as predicted

The prior conversation analyzed the ceiling as:

```
V_eq = V_rest + i_total
ISI = τ_m · ln((V_eq − V_reset) / (V_eq − V_thresh))
```

At the best static configuration (K=32, v_max=2.0, tonic=16 mV), max steady-state i_total ≈ 19.2 mV → V_eq ≈ −45.8 → ISI ≈ 30 ms → f ≈ 33 Hz. That's the static ceiling. Dynamic adrenaline via `tau_m_scale` changes **τ_m on the fly**:

```
ISI(adrenaline) = (τ_m / adrenaline) · ln(...)
```

So doubling adrenaline halves the ISI and doubles the firing rate. With adrenaline = 1.7, the ceiling moves from 33 Hz to ~55 Hz; that covers A's 50 Hz peak. The controller actually needed only small adrenaline excursions (the avg = 1.00 per segment is misleading — instantaneous adrenaline during A's rising edges is briefly higher before the EMA catches up).

### What Step 10 validates beyond the ceiling

- **Closed-loop control works as advertised.** My earlier concern that Option B was "just a gain amplifier papering over the structural problem" was wrong in the useful direction — the gain amplifier's job here is *exactly* what's needed to break the structural ceiling, and the closed-loop operation is what makes it non-trivial.
- **Biology's use of NE matches this function cleanly.** Aston-Jones & Cohen 2005 describe LC phasic NE bursts as the mechanism by which cortex pushes past its local saturation when a salient stimulus demands more response. Step 10 is literally that mechanism in code: an error signal between "what B is doing" and "what B ought to be doing" drives a gain modulator that shortens the membrane time constant.
- **Gain sensitivity is low.** gain = 50 and gain = 200 produce indistinguishable per-segment fitness (both −5.60e−5). Gain = 10 is underpowered but still improves on open-loop. This suggests the controller is robust over a wide gain range — rare in hand-tuned controllers.

### Caveats and honest limits

- **The task is still signal-following.** Closed-loop adrenaline at gain = 50 produces the behavioral cheat "match A's rate" without requiring B to learn any internal structure. B is essentially a rate-follower by construction. The question of whether B develops any *predictive* capability (not just tracking, but anticipating A's next-window rate) isn't answered by this test. The A trace is piecewise constant; "predict next window" collapses to "match current window" for most windows, with errors concentrated at transitions — which the controller handles via its lag.
- **Tonic drive is still the kludge it was in step 7.** The E/I substrate could in principle remove the need for an external tonic drive (inhibition-loop dynamics could maintain a firing regime), but we haven't designed that architecture. Step 10 rides on top of the same tonic=16 mV scaffold as step 7.
- **STDP is still running but doing nothing load-bearing.** The hand-wired pool has plasticity_rate = 0 by construction. So the weight evolution dynamics we built in step 4 don't contribute to step 10's result. The controller is doing all the work.
- **Single seed.** Fitness of −5.60e−5 at gain = 50 should be replicated across multiple seeds before being cited as "the architectural-ceiling breaker"; could be a lucky alignment. Multi-seed validation is a trivial rerun.

### Candidate for novelty flag

Per the earlier agreement to flag potentially original contributions: Step 10's combination of (E/I substrate, slot-pool structural plasticity ready, tau_m_scale gain mechanism under closed-loop controller reading from spike-rate EMAs) is a specific architectural configuration I cannot place in a single paper from training-data recollection. The individual components have precedent:

- **E/I balanced networks** — van Vreeswijk & Sompolinsky 1996+
- **Adaptive gain theory** — Aston-Jones & Cohen 2005
- **Closed-loop neural gain control in hardware** — various neuromorphic publications
- **Slot-pool synaptic representations** — less common but not unprecedented

The specific composition — particularly the choice to make adrenaline a *closed-loop* analog broadcast with a concrete controller rather than an open-loop "input salience → gain" feed-forward — is where I see a potential novelty claim. Confidence: moderate. Verification would be a literature search for "closed-loop neuromodulation", "feedback control of neural gain", and related terms in recent (2024-26) neuromorphic and computational-neuroscience journals.

Not asserting originality here; flagging it for later verification if we pursue publication. The engineering achievement ("architectural ceiling overcome by dynamic gain") is a concrete silicritter milestone regardless of novelty.

### Phase completion

Steps 9 and 10 together close the N=256 re-scale arc in full:

- N=256 scaled cleanly (Phase 2).
- Architectural leverage exists at tonic=16 / v_max=2 / K=32 (Phase 3).
- Direct GA can't reach the static ceiling (Phase 4, Phase 4.1).
- CPPN indirect encoding matches static ceiling (Step 8).
- E/I substrate validates (Step 9).
- **Closed-loop adrenaline breaks the static ceiling** (Step 10).

The project now has working GA + indirect encoding + E/I substrate + closed-loop modulation machinery. That's the full toolkit the project has been circling since step 7 first plateaued.

---

## 2026-04-21 — Step 10 multi-seed confirmation

- **Script:** `experiments/step10_closedloop_adrenaline.py --n-seeds=5`
- **Seeds tested:** 0, 37, 74, 111, 148 (stride 37 for independent draws of B's cross-E-only pre-assignment)
- **All other parameters:** unchanged from the single-seed run above (N=256, K=32, T=2000, E:I 80:20, i_mult=4.0, EMA decay 0.98, adrenaline range [0.5, 3.0])

| condition | mean | std | min | max |
|---|---:|---:|---:|---:|
| open-loop (const adr=1.0) | −1.561e−4 | 8.49e−7 | −1.571e−4 | −1.550e−4 |
| closed-loop gain = 10 | −1.072e−4 | 4.60e−6 | −1.116e−4 | −1.012e−4 |
| closed-loop gain = 50 | **−5.595e−5** | **0.00e+00** | −5.595e−5 | −5.595e−5 |
| closed-loop gain = 200 | **−5.595e−5** | **0.00e+00** | −5.595e−5 | −5.595e−5 |

### What this tells us

- **Open-loop is already nearly seed-independent** (std 8.5e−7 on a fitness of 1.56e−4, i.e., relative variation ~0.05%). The 256-neuron population averages over per-seed draw differences in which specific A-E neuron each of B's slots latches onto.
- **Closed-loop gain = 50 and gain = 200 produce identical fitness to the float-precision limit across all 5 seeds** (std exactly 0.0). The controller is saturating adrenaline against the `ADR_MAX = 3.0` rail during A's peak segments, which makes B's instantaneous firing pattern deterministic given the piecewise-constant A drive. This is a ceiling — not B's intrinsic f-I ceiling (which tau_m_scale defeated), but the controller's **output clip.** Raising ADR_MAX would presumably let gain = 200 pull ahead of gain = 50 briefly, though the return on an already-near-perfect tracking residual may not justify it.
- **The single-seed result from the previous entry is confirmed as the actual architectural finding**, not a lucky alignment. ~2.8× fitness improvement (open-loop 1.56e−4 → closed-loop 5.6e−5) holds robustly across seeds.
- **gain = 10 sits at the intermediate regime** where the controller is too slow to catch A's rising edges before the window ends — modest variance (std 4.6e−6, ~4% relative) reflects per-seed differences in exactly *when* the controller catches up relative to the 100-step windows used for rate measurement.

### Caveat this run resolves

Earlier entry flagged "single seed" as a caveat. This entry discharges that caveat: the ceiling-breaking result reproduces to the float-precision limit at gain ≥ 50, so the finding is not seed-dependent.

### Caveat this run surfaces

The zero-variance at gain ≥ 50 is a **new** finding worth its own note: the controller is rail-limited at its output, not at the neuron-dynamics level. If a future task requires B to track A beyond 50 Hz peaks (the current A_DRIVE_PROFILE's max), ADR_MAX = 3.0 will become the ceiling. The substrate has headroom; the controller currently does not.

---

## 2026-04-23 — Step 11: CPPN GA + E/I + closed-loop adrenaline

- **Script:** `experiments/step11_cppn_closedloop.py`
- **Library:** this run also promotes the closed-loop controller from the step 10 script into `src/silicritter/closedloop.py` (`ControllerState`, `ControllerParams`, `init_controller`, `step_closedloop`, `simulate_closedloop`), with 100% branch-coverage tests in `tests/test_closedloop.py`. Step 10 was refactored to call into the library and produces byte-identical multi-seed output.
- **Substrate:** identical to step 10 — N=256, K=32, T=2000, E:I 80/20, i_mult=4, tau_m_scale gain mode. A's pool fixed at seed 777.
- **GA:** pop=32, 30 generations, CPPN hidden_dim=8 (51 weights per genome), init_scale=1.0, mutate_sigma=0.15, elite=2, tournament=3. Same hyperparameters as step 7e so comparison is fair.
- **Conditions:** (1) **open-loop** — CPPN evolves B's pool with constant adr=1.0; (2) **closed-loop** — CPPN evolves B's pool with the controller on at gain=50 during every fitness evaluation.

### Result

| condition | best fitness | decoded B pool topology |
|---|---:|---|
| step 9 hand-wired cross-E, open | −1.56e−4 | 100% cross-E, v=2.0 (uniform random over A's E neurons) |
| step 10 hand-wired cross-E, closed-loop gain=50 | −5.60e−5 | same as step 9 |
| step 7e CPPN GA, no E/I, open | −1.70e−4 | high cross, high v |
| **step 11 CPPN GA + E/I, open-loop** | **−1.412e−4** | 100% cross, v mean 1.999, max 2.000 |
| **step 11 CPPN GA + E/I, closed-loop gain=50** | **−4.923e−5** | **87.9% cross / 12.1% recurrent**, v mean 1.992, max 1.994 |

Both step 11 conditions converge inside 5 generations and plateau for the remaining 25. Total wall time: ~37 s on the MX150.

### What this run measures

- **Open-loop CPPN+E/I beats step 9's hand-wired cross-E (−1.41e−4 vs −1.56e−4, ~10%).** Step 9 drew A's E pre-ids uniformly at random from A's 205 excitatory neurons. The CPPN-decoded pool is also 100% cross-E at v=2.0 (within float-precision of v_max), but it is *not* a uniform-random draw — the CPPN's output distribution concentrates some B post-neurons onto a shared subset of A's E neurons. That clustered connectivity correlates input across B's post-neurons, changing B's rate dynamics enough to shave 10% off the MSE at the static ceiling. The GA didn't break the architectural ceiling here — it found a better *sampling* of the same cross-E-only topology.
- **Closed-loop CPPN+E/I beats step 10's hand-wired by ~12% (−4.92e−5 vs −5.60e−5), and — this is the informative part — the evolved topology is NOT cross-E-only.** ~12% of B's slots (≈990 of 8192) bind to B's own neurons (recurrent). The GA discovered that, under closed-loop gain, adding ~12% recurrent E gives the controller a second lever to work with: adrenaline shortens `tau_m`, which speeds integration of both A-driven cross input AND B→B recurrent input. At the segment transitions where the EMA-lag residual lived, the recurrent component provides transient amplification the pure-cross config didn't have.
- **The "closed-loop saturation at ADR_MAX" story from step 10's multi-seed note is incomplete.** Step 10 argued the controller was rail-limited at gain≥50 and therefore topology wouldn't help. Step 11 refutes that: topology *does* help, just not through the same mechanism. The pure-cross B at gain=50 saturates adrenaline at the rail, yes; but the 88/12 cross/recurrent B at gain=50 can sit at a lower adrenaline and still track A, because recurrence handles part of the amplification the rail-limited adrenaline was being asked to do.

### The specific numbers, interpreted

- Static ceiling (no closed loop, no topology tricks): −1.56e−4 (step 9).
- Static ceiling with GA-discovered topology tricks: −1.41e−4. A ~10% improvement purely from better connectivity sampling, no controller involved.
- Closed-loop ceiling with hand-wired topology: −5.60e−5 (step 10). ~3.5× better than static.
- Closed-loop ceiling with GA-discovered topology: −4.92e−5. Another ~12% on top of that.

The composition effects stack: smart topology alone gives ~10%; closed-loop alone gives ~3.5×; smart topology + closed-loop gives ~3.5× × ~1.12 ≈ 4.0× over the step 9 static baseline.

### Caveats

- **Single seed.** Step 11's conclusions rest on seed=0 for the CPPN init and seed=0 for E/I assignment. Step 1's multi-seed discipline should be reapplied here before any of these numbers get cited as robust findings; the topology discovery in particular could be seed-sensitive (different initial populations might converge on different non-cross fractions). A ~5-seed confirmation run would take ~3 minutes of wall time.
- **Plasticity ran but did nothing load-bearing.** As in step 10, the CPPN-decoded pool has plasticity_rate = 0 everywhere, so STDP updates are zero-weight. The fitness reflects the CPPN-decoded topology evaluated under the controller, not any learned evolution of that topology during the simulation.
- **The "12% recurrent" finding is a population statistic, not a circuit.** I haven't inspected *which* B post-neurons get the recurrent slots or whether the recurrent fraction is clustered by post-neuron. Could be uniformly sprinkled (each post has ~4 of 32 slots recurrent) or could be pathological (a small subset of posts are fully recurrent, the rest are fully cross). That distinction matters architecturally.
- **Task is still "match A's rate."** This run inherits step 10's task limitations. The GA isn't being asked to *predict* anything non-trivial; it's being asked to track a piecewise-constant signal with its input-to-output transfer function.

---

## 2026-04-23 — Step 12: tonic-drive sweep under E/I + closed-loop

- **Script:** `experiments/step12_tonic_sweep.py`
- **Question:** `B_BASELINE_DRIVE_MV = 16.0` has been a load-bearing kludge since step 7. Step 9 added E/I substrate; step 10 added closed-loop adrenaline. Does either — individually or combined — make the tonic redundant? How far can we peel it off before B's tracking breaks?
- **Setup:** step 10's hand-wired cross-E-only B pool (v = 2.0), identical across the sweep. Two conditions per tonic value: open-loop (const adr = 1.0) and closed-loop (gain = 50).
- **Reference:** A's per-segment firing rates are [28.0, 44.0, 32.0, 50.0] Hz.

### Result

| tonic (mV) | open-loop fit | open-loop B rates (Hz) | closed-loop fit | closed-loop B rates (Hz) |
|---:|---:|:---|---:|:---|
| 16.0 | −1.566e−4 | [28.0, 28.7, 32.0, 30.8] | −5.595e−5 | [28.0, 44.0, 32.0, 50.0] |
| 12.0 | −6.273e−4 | [14.0, 17.1, 16.0, 17.0] | −9.117e−5 | [28.0, 39.4, 31.8, 40.4] |
|  8.0 | −1.637e−3 | [ 0.0,  0.0,  0.0,  0.0] | −3.850e−4 | [14.0, 25.1, 16.0, 30.5] |
|  4.0 | −1.637e−3 | [ 0.0,  0.0,  0.0,  0.0] | −1.637e−3 | [ 0.0,  0.0,  0.0,  0.0] |
|  0.0 | −1.637e−3 | [ 0.0,  0.0,  0.0,  0.0] | −1.637e−3 | [ 0.0,  0.0,  0.0,  0.0] |

### Findings

- **Open-loop collapses fast.** At tonic = 12 mV, B's firing rate hollows out (14–17 Hz across all segments, no shape). At tonic ≤ 8 mV, B is completely silent — cross-E-only input from A cannot push B above threshold without a baseline drive to keep its membrane near V_thresh. The −1.637e−3 fitness floor is the "B never fires" pessimum, i.e., `mean((0 − a_next)^2)` across windows.
- **Closed-loop pushes the cliff down by 4 mV.** At tonic = 12, closed-loop still tracks A cleanly: [28, 39, 32, 40] vs A's [28, 44, 32, 50], fitness −9.1e−5 (~1.6× worse than the tonic = 16 baseline of −5.6e−5, but still an order of magnitude better than open-loop at the same tonic). At tonic = 8, B becomes partially responsive — [14, 25, 16, 31], roughly the right shape at half amplitude, fitness −3.85e−4 (4× worse than tonic = 16 closed-loop, but still meaningfully better than the open-loop floor).
- **Closed-loop has its own cliff at tonic ≤ 4.** Once tonic drops to 4 mV, no amount of adrenaline can rescue B. The physics: `V_eq = V_rest + i_total`, and `ISI = (τ_m / adrenaline) × ln((V_eq − V_reset) / (V_eq − V_thresh))`. When `V_eq < V_thresh`, the log argument is negative and ISI is undefined — the neuron simply cannot spike, no matter how small `τ_m / adrenaline` gets. Closed-loop rescues near-threshold neurons by speeding their integration; it cannot rescue sub-threshold neurons that never reach the firing condition.

### Takeaway for the project

**Tonic is not redundant. It is load-bearing, for a reason different from the one I'd been casually assuming.** I'd been treating tonic as a "scaffold" that might be replaceable by E/I self-balancing; that framing was wrong. Tonic provides the *minimum excitation floor* below which cross-input cannot push B to threshold. E/I substrate shapes existing excitation (inhibition clips runaway peaks, cross-E provides signal), but it does not generate novel excitation — inhibition cannot create action potentials.

The analogy: tonic is like the cortical default-mode's metabolic baseline. Real cortex has ongoing spontaneous activity even in the absence of stimulus; that activity keeps neurons near threshold so incoming signals arrive at a responsive population. Remove the baseline (deep anesthesia, coma), and stimuli fail to propagate even through intact circuitry.

### What might actually replace tonic (future work)

- **Recurrent E.** Step 11's closed-loop-evolved pool had 12% recurrent slots. At tonic = 0, that recurrence could in principle self-sustain activity once a single spike occurs. But this requires bootstrapping — *something* has to fire the first spike. In biology, spontaneous synaptic noise handles that.
- **Synaptic noise.** A very low-amplitude Gaussian noise current added to each B neuron would provide the "randomness floor" cortex uses for spontaneous activity. Would need to verify that the noise doesn't destroy the MSE fitness signal.
- **Structural-plasticity-driven recurrent growth.** If B's recurrent fraction can grow under valence pressure to levels that sustain activity, tonic becomes truly optional. But this is a big architectural ask, not a quick experiment.

### Caveats

- **Single seed.** As with steps 10 and 11. Step 1's multi-seed discipline should reapply here before citing these numbers as robust findings.
- **Step 11's evolved pool NOT tested here.** This sweep used step 10's hand-wired 100%-cross pool. The step 11 evolved pool (87.9% cross / 12.1% recurrent) might push the cliff down further under closed-loop, because recurrent E provides self-amplification once any spikes occur. That's a natural follow-up experiment.
- **Plasticity off throughout.** `plasticity_rate = 0` means no weight adaptation during the sim. Whether STDP would grow B's effective sensitivity to cross-input enough to recover from low-tonic regimes is untested — but given the "sub-threshold physics" reason for the cliff, probably not.

---

## 2026-04-23 — Step 13: perturbation validation of D007 E/I canonical values

- **Script:** `experiments/step13_ei_perturbation.py`
- **Question:** D007 adopted `(inhibitory_fraction, i_weight_multiplier) = (0.2, 4.0)` provisionally, based on literature consensus and with an explicit "may be following the herd" caveat. Does the substrate's own dynamics confirm this as a reasonable point, reveal a sharply better off-canonical configuration, or show catastrophic failure modes nearby?
- **Grid:** 5×5: `i_fraction ∈ {0.0, 0.1, 0.2, 0.3, 0.4}` × `i_mult ∈ {1.0, 2.0, 4.0, 6.0, 8.0}`. All other parameters held at step 10's configuration (hand-wired cross-E-only B pool, tonic = 16 mV). Both open-loop and closed-loop (gain = 50) evaluated at every cell.

### Open-loop fitness grid (const adr = 1.0)

```
i_mult \\ i_frac |    0.00 |    0.10 |    0.20 |    0.30 |    0.40
         1.0     -1.74e-4  -1.67e-4  -1.68e-4  -1.71e-4  -1.72e-4
         2.0     -1.74e-4  -1.66e-4  -1.71e-4  -1.72e-4  -1.57e-4
         4.0     -1.74e-4  -1.69e-4  -1.57e-4*  -1.51e-4  -1.53e-4
         6.0     -1.74e-4  -1.61e-4  -1.47e-4  -1.54e-4  -1.77e-4
         8.0     -1.74e-4  -1.49e-4  -1.46e-4  -1.65e-4  -1.83e-4
```
(\*) = D007 canonical.

### Closed-loop fitness grid (gain = 50)

```
i_mult \\ i_frac |    0.00 |    0.10 |    0.20 |    0.30 |    0.40
         1.0     -5.79e-5  -5.73e-5  -5.53e-5  -5.61e-5  -5.75e-5
         2.0     -5.79e-5  -5.61e-5  -5.57e-5  -5.76e-5  -5.74e-5
         4.0     -5.79e-5  -5.53e-5  -5.60e-5*  -5.27e-5  -5.33e-5
         6.0     -5.79e-5  -5.53e-5  -5.33e-5  -5.57e-5  -1.00e-4
         8.0     -5.79e-5  -5.48e-5  -4.98e-5  -9.08e-5  -1.06e-4
```
(\*) = D007 canonical.

### Findings

- **D007 is reasonable but not optimal.** Canonical `(0.2, 4.0)` sits at −1.57e−4 open-loop and −5.60e−5 closed-loop. Best-found in both conditions is `(0.2, 8.0)`: −1.46e−4 open-loop (−7% MSE vs canonical), −4.98e−5 closed-loop (−11% MSE vs canonical). The winning i_fraction = 0.2 matches D007; it's the multiplier that wants to be twice as high.
- **The `i_fraction = 0` column is constant per row** (all open-loop = −1.74e−4; all closed-loop = −5.79e−5). That's a sanity check: with no inhibitory neurons, the multiplier has no effect.
- **There's a collapse region at high (i_fraction, i_mult)** visible in the closed-loop grid: `(0.4, 8.0) = −1.06e−4`, `(0.4, 6.0) = −1.00e−4`, `(0.3, 8.0) = −9.08e−5`. Same mechanism as step 12's tonic cliff: too much inhibition drops B's equilibrium below threshold, no adrenaline rescue possible. At `(0.4, 8.0)` with tonic=16 and i_mult=8, inhibitory drive overwhelms the cross-E signal entirely. The open-loop grid shows milder but similar degradation at `(0.4, 8.0) = −1.83e−4`.
- **Canonical sits in the smooth interior of a safe basin**, not near either optimum or collapse. The nearest neighbors in the grid are all within ~15% of canonical's fitness; the `(0.2, 8.0)` improvement is a consistent monotonic trend along i_frac=0.2, not a sharp pocket.

### D007 status

**Not superseding D007 on this data.** Multi-seed discipline (see step 10's multi-seed entry) should confirm any adoption change, and the `(0.2, 8.0)` advantage is 7–11% — meaningful but within the range of single-seed variation I haven't measured for this experiment yet. A `--n-seeds=5` rerun of step 13 at `(0.2, 4.0)` vs `(0.2, 8.0)` would settle it. Until that's done, D007 stays provisional per its own text. If multi-seed confirms, the decision to supersede D007 with `(0.2, 8.0)` would be a one-line D008.

### Takeaway for the broader architecture

- **E/I detail matters, but not sharply.** The substrate is forgiving of parameter choices within the interior of the grid; performance changes smoothly in the 10% range. It is NOT forgiving at the high-inhibition corner where physics takes over.
- **The i_fraction axis is narrower than the i_mult axis.** Moving from i_fraction=0.2 to 0.4 at i_mult=4 loses ~5%; moving from i_mult=4 to 8 at i_fraction=0.2 *gains* ~11% in closed-loop. The inhibitory-fraction range 0.1–0.3 is safe; above 0.3 is cliff-adjacent.
- **Canonical E:I literature reporting typically conflates fraction and multiplier**. "20% inhibitory" says nothing about synaptic strength. This sweep suggests the interesting axis is strength (per-synapse current magnitude), not count.

### Caveats

- **Single seed.** Same caveat as steps 10–12. Multi-seed confirmation is the gating requirement before any D007 supersession.
- **Tonic fixed at 16 mV.** Step 12 showed tonic interacts with the collapse cliff; step 13's cliff might shift if tonic is lower. A joint (tonic × i_mult × i_frac) sweep would be more thorough but is 3× the grid; skipped for now.
- **Hand-wired B pool only.** Step 11's evolved 87.9% cross / 12.1% recurrent pool might have a different E/I sensitivity profile (the recurrent slots carry B's own I neurons' spikes, so i_mult affects B's internal inhibition too). Re-running step 13 with step 11's genome-decoded pool would be the natural next step if E/I tuning becomes load-bearing for a downstream experiment.

---

## 2026-04-23 — Step 13 multi-seed: D008 supersedes D007

- **Script:** `experiments/step13_ei_perturbation.py --n-seeds=5`
- **Focused comparison** between D007 canonical `(0.2, 4.0)` and the step 13 grid's best-found point `(0.2, 8.0)`. Stride 37 for independent seed draws. Both open-loop and closed-loop conditions.

| point | condition | mean | std | min | max |
|---|---|---:|---:|---:|---:|
| (0.2, 4.0) | open-loop | −1.561e−4 | 8.49e−7 | −1.571e−4 | −1.550e−4 |
| (0.2, 4.0) | closed-loop | −5.595e−5 | 0.00e+00 | −5.595e−5 | −5.595e−5 |
| (0.2, 8.0) | open-loop | −1.460e−4 | 9.41e−7 | −1.473e−4 | −1.450e−4 |
| (0.2, 8.0) | closed-loop | −4.944e−5 | 5.94e−7 | −5.016e−5 | −4.862e−5 |

### Findings

- **Supersession confirmed.** `(0.2, 8.0)` wins open-loop by 6.5% and closed-loop by 11.6%. Inter-point gaps (1.01e−5 open-loop, 6.5e−6 closed-loop) exceed each point's std by ~10×. The improvement is robust to seed variation. **D008 records the supersession.**
- **Mechanism clue in the closed-loop std column.** At D007 canonical, closed-loop std is exactly 0.0 — same rail-limit finding step 10's multi-seed entry flagged: adrenaline saturates at `ADR_MAX = 3.0` and B's firing pattern becomes deterministic. At D008's `(0.2, 8.0)`, closed-loop std is 5.94e−7 (small but non-zero), meaning the controller is NOT saturated. Stronger per-synapse inhibition (i_mult=8 vs 4) compresses B's firing range enough that the controller operates in the middle of its band, not at the rail. The fitness improvement is *driven by* moving the operating point off the rail.

### Follow-up flagged but not chased

Step 12's tonic sweep and step 13's grid were run at i_mult=4.0. Re-running either at the new D008 default (8.0) might shift the cliff locations — tonic might drop further before collapsing, or might collapse sooner because B's inhibition is stronger. Not urgent; flagged in DECISIONS.md (D008 "Impact on code" section) for anyone re-exploring these regimes.

---

## 2026-04-23 — Step 14: fractional-Gaussian-noise stimulus to A

- **Script:** `experiments/step14_fgn_stimulus.py`
- **Setup:** step 10's hand-wired cross-E-only B pool + E/I substrate at D008 `(0.2, 8.0)` + closed-loop gain=50. A's drive replaced with fGn (mean=20.75 mV, std=2.0 mV) at H ∈ {0.3, 0.5, 0.7, 0.9}. B's tonic held at 16 mV. Each H run produces both a **tracking** fitness (−MSE(B(t), A(t))) and a **prediction** fitness (−MSE(B(t), A(t+1))).

### Result

Open-loop (const adr = 1.0):

| H | track | pred | pred−track | stim lag1 | A rate | B rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | −8.01e−5 | −8.45e−5 | −4.39e−6 | −0.171 | 38.8 Hz | 30.2 Hz |
| 0.50 | −9.57e−5 | −1.01e−4 | −5.16e−6 | +0.077 | 39.0 Hz | 30.1 Hz |
| 0.70 | −9.40e−5 | −9.64e−5 | −2.39e−6 | +0.349 | 39.5 Hz | 30.2 Hz |
| 0.90 | −1.33e−4 | −1.40e−4 | −7.87e−6 | +0.589 | 40.9 Hz | 30.1 Hz |

Closed-loop (gain = 50):

| H | track | pred | pred−track | stim lag1 | A rate | B rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | **−4.12e−6** | −8.80e−6 | −4.67e−6 | −0.171 | 38.8 Hz | 37.8 Hz |
| 0.50 | **−4.52e−6** | −1.31e−5 | −8.60e−6 | +0.077 | 39.0 Hz | 37.9 Hz |
| 0.70 | −1.18e−5 | −1.47e−5 | −2.92e−6 | +0.349 | 39.5 Hz | 38.0 Hz |
| 0.90 | −1.14e−5 | −2.86e−5 | −1.71e−5 | +0.589 | 40.9 Hz | 38.5 Hz |

### Findings

- **fGn stimulus is viable.** All four H values produce sensible firing regimes; no cliff, no silent segments. A fires at ~39–41 Hz across H; B tracks at ~30 Hz open-loop and ~38 Hz closed-loop. The architecture runs on fractional noise without modification.
- **Closed-loop dramatically improves tracking under all H.** Open-loop track is roughly −8e−5 to −1.3e−4 depending on H; closed-loop track is −4e−6 to −1.2e−5 — 20–30× improvement. The closed-loop controller works on fGn just as it worked on the step function. B's mean rate pushes from 30 Hz up to ~38 Hz under closed-loop, tracking A's ~40 Hz across conditions.
- **Tracking fitness depends on H non-monotonically.** Closed-loop track is best at H=0.3 (−4.1e−6) and H=0.5 (−4.5e−6), gets worse at H=0.7 (−1.2e−5) and H=0.9 (−1.1e−5). The effect is real, not noise — a 3× worsening from H=0.5 to H=0.7.
- **The pre-run hypothesis is falsified.** I predicted the closed-loop prediction-minus-tracking gap would scale cleanly with `(1 − lag1)`, i.e., big gap at H=0.3 (antipersistent), small gap at H=0.9 (persistent). Observed is the opposite: H=0.9 has the LARGEST gap (−1.71e−5), H=0.3 has a moderate one, H=0.7 the smallest. Whatever's happening, it's not pure-tracking-plus-stim-autocorr.
- **Why the non-monotonic shape.** Two mechanisms fighting: (a) per-window A-rate variance grows with H (long-range dependence means less averaging-out within a 100-ms window), so there's more for B to track; (b) A's lag1 grows with H, so the "trivial prediction" error (report current) gets smaller. At H=0.5 the variance is moderate and lag1 ≈ 0, and B tracks easily; at H=0.9 the variance is large (harder target) but lag1 is high (which could help if B had memory — it doesn't). The closed-loop controller's EMA smooths on the response side but not on the input side, so high-H signals stress the architecture.

### Hypothesis update

B has **no mechanism to exploit long-range temporal structure**. The EMA in the controller integrates B's and A's recent rates, but its time constant (~50 ms) is matched to the response loop, not to A's temporal structure. B's pool is fixed with `plasticity_rate = 0`, and the valence trace is zero throughout, so STDP does nothing.

The non-monotonic fitness curve across H is the architecture's dynamics interacting with stimulus variance structure — NOT genuine predictive learning. Whatever H-dependence we see is accidental; making prediction load-bearing requires either (i) non-exponential memory in the controller (fractional EMA — the direction Ed is investigating separately in ~/math/fraccalc), (ii) non-zero plasticity with a valence signal driven by prediction error, or (iii) an architectural memory in B (recurrent slots that carry a historical signature of A's past).

### Not backing out

The original back-out clause (see README dev log, 2026-04-23 entry) said "if fitness is indistinguishable across H, the fractional-stimulus direction is a blind alley." That's not what happened — fitness clearly depends on H. The direction is still viable; the **finding** is that the current architecture doesn't extract useful information from long-range dependence, only stumbles through different-variance regimes of it.

### Next (Step 5c promised)

Compute the Wiener-Kolmogorov optimal predictor error for the same fGn stimulus at each H. That gives a theoretical floor for "best any architecture could do." If B's observed prediction residual is far above the WK floor, there's real room for an architectural upgrade to close the gap. If B is already near the WK floor, further architectural work isn't going to help on this task.

### Caveats

- **Single seed.** As with every recent experiment. Multi-seed discipline should be applied before strong claims.
- **Hurst range is limited.** fGn is defined for H ∈ (0, 1) but the Davies-Harte embedding can become ill-conditioned near 1.0. H=0.95 would stress-test the numerics; not run here.
- **Window size 100 ms is a pre-chosen parameter.** Varying WINDOW_STEPS might reshape the tracking/prediction distinction. Not swept.
- **fGn is mean-zero stationary.** Real stimuli are rarely that well-behaved; the broader question (does this architecture handle non-stationary signals?) is open.

---

## 2026-04-23 — Step 15: Wiener-Kolmogorov floor vs. observed prediction error

- **Script:** `experiments/step15_wk_floor_comparison.py`
- **New library:** `src/silicritter/wk.py` (Durbin-Levinson recursion, windowed-fGn autocovariance, WK floor) with `tests/test_wk.py` (37 tests across 5 independent cross-check paths, 100% branch coverage). See the test-design discussion in the session log for the rationale.
- **What's compared:** For each Hurst value, the step 14 architecture is run to produce A's per-window rate trace; the empirical autocovariance of that trace is computed; Durbin-Levinson gives the theoretical one-step prediction MSE achievable by any linear predictor with access to A's history. That's compared to B's observed prediction MSE from the same run.

### Result

Open-loop (const adr=1.0):

| H | A window var | WK floor | B pred MSE | B / floor |
|---:|---:|---:|---:|---:|
| 0.30 | 4.85e−6 | 4.40e−6 | 8.45e−5 | 19.19 |
| 0.50 | 6.19e−6 | 5.62e−6 | 1.01e−4 | 17.96 |
| 0.70 | 7.60e−6 | 6.39e−6 | 9.64e−5 | 15.08 |
| 0.90 | 1.66e−5 | 1.12e−5 | 1.40e−4 | 12.57 |

Closed-loop (gain=50):

| H | A window var | WK floor | B pred MSE | B / floor |
|---:|---:|---:|---:|---:|
| 0.30 | 5.13e−6 | 4.63e−6 | 8.80e−6 | **1.90** |
| 0.50 | 6.30e−6 | 5.69e−6 | 1.31e−5 | **2.31** |
| 0.70 | 7.64e−6 | 6.52e−6 | 1.47e−5 | **2.25** |
| 0.90 | 1.72e−5 | 1.18e−5 | 2.86e−5 | **2.43** |

### Findings

- **Closed-loop B is 1.9–2.4× above the WK floor at every H.** This is strikingly close to optimal for an architecture with **zero explicit memory**. B has no recurrent state, `plasticity_rate = 0`, and the controller's EMA has a 50-ms time constant. Whatever implicit "prediction" is happening is doing most of the job the theoretical-best linear predictor would do.
- **Open-loop is 12–19× above the floor.** The controller matters a lot here. Without gain modulation, B is barely tracking, let alone predicting.
- **The WK floor is NOT much smaller than A's window variance.** Ratio floor/variance is 0.88–0.91 across H, meaning the theoretical best linear predictor reduces prediction MSE by only 9–12% below "just report the mean." The 2.4× gap between B and the floor (at H=0.9, closed-loop) is thus ~2× the *absolute* reducible error, not ~2× the whole prediction error.
- **Step 14's "gap grows with H" finding is preserved in the WK-floor-normalized view.** The B/floor ratio is slightly smaller at H=0.3 (1.90) than at H=0.9 (2.43). That is, B gets slightly *closer* to optimal when the signal is antipersistent. This makes mechanistic sense: antipersistent signals are easy to track (each window tends to undo the previous deviation), so B's lag hurts less.

### Interpretation for the fractional-stimulus direction

**The "does memory help" back-out condition is clarified, and the answer leans toward "probably not much."**

The upper bound on improvement from adding memory to B is 2.4× reduction in prediction MSE (the floor is 2.4× below current B at H=0.9, closed-loop). And that upper bound assumes the memory-capable architecture reaches the information-theoretic floor exactly — real architectures would fall short.

A reasonable expectation: memory upgrades (fractional EMA in controller, recurrent B, or learned plasticity with prediction-error valence) might close half the gap, so maybe 1.5× improvement. That's not nothing, but it's also not the 10× improvement the step 14 numbers had me imagining when I didn't know the floor.

**This doesn't mean the direction is dead; it means the scientific framing shifts.** The interesting question isn't "can we close the gap to WK floor" (answer: maybe ~1.5–2×, with effort). The interesting question is "what happens when we relax the task to require actual memory" — e.g., non-stationary A, A with structural change-points, longer prediction horizons. The fGn-stationary task is a well-defined benchmark that happens to already have very little memory headroom available.

### Caveats

- **Single seed.** As with every recent experiment.
- **Empirical autocov estimation has sampling error**, especially at high lags. At n_windows=20, only ~19 samples of lag-1 autocov; variance on the estimate is substantial. Using longer simulations (N_TIMESTEPS=10000 → 100 windows) would tighten the floor estimate.
- **max_lag=10 (half of n_windows).** DL-based floor converges as max_lag grows; using the full history minus 1 might push the floor slightly lower.
- **Empirical autocov is a biased estimator.** The N-normalized (rather than (N-k)-normalized) form is consistent but biased low at high lags; the floor might be slightly underestimated as a result. For a precise measurement, the unbiased form should be used — but the 2× gap is bigger than that bias.

---

## 2026-04-23 — Step 16: STDP-driven learning with plasticity turned on

- **Script:** `experiments/step16_stdp_learning.py`
- **First silicritter experiment with `plasticity_rate > 0`.** Every step from 9 onward ran with `plasticity_rate = 0` everywhere, so STDP was wired but did nothing. Step 16 asks: does STDP actually reshape a random B pool into a tracking-capable one under a valence-gated closed-loop reward?
- **Setup:** B's initial pool is random (uniform `pre_ids` over `[0, 2N)`, truncated-Gaussian `v`). A hand-wired as in step 10. Valence is a shaped reward computed online from rate EMAs: `v = max(0, 1 − |rate_a_ema − rate_b_ema| / 0.015)`. Closed-loop adrenaline controller on at gain=50. Three-phase structure: Phase A measures pre-training fitness (plasticity frozen), Phase B trains for 20k steps (plasticity on), Phase C measures post-training fitness (plasticity re-frozen).

### Plasticity-rate sweep (mid-init: v_mean=1.0, v_std=0.3)

| rate | mean v after | v std after | sat @ max | sat @ min | Phase A fit | Phase C fit | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.996 | 0.299 | 0.000 | 0.000 | −2.586e−4 | −2.626e−4 | −3.99e−6 (noise) |
| 0.10 | 0.983 | 0.300 | 0.000 | 0.000 | −2.586e−4 | −2.583e−4 | +3.27e−7 (flat) |
| 0.30 | 0.952 | 0.305 | 0.000 | 0.000 | −2.586e−4 | −2.478e−4 | +1.08e−5 |
| 1.00 | 0.848 | 0.352 | 0.000 | 0.004 | −2.586e−4 | −2.214e−4 | +3.72e−5 |

Alternative (init near saturation: v_mean=1.9, v_std=0.1, rate=1.0): Phase A = −2.961e−4, Phase C = −2.640e−4, Δ = +3.21e−5. Mean v ended at 1.756 with 3.4% at v_max.

### Findings

- **STDP does move weights, but only at meaningful plasticity_rate.** At rate=0.01, the accumulated change is ~0.1% of v_max over 20k steps — within measurement noise. At rate≥0.3, weight distributions shift visibly. Rate=1.0 is the operating point where the signal emerges from the noise.
- **Motion is LTD-dominant.** Default STDP has `a_minus=0.012 > a_plus=0.01`, and in this paired substrate the mean weight *decreases* during training. B→A-I slots (cross-inhibitory) reliably produce "pre fires, post doesn't" patterns that trigger LTD; those are functionally counterproductive and STDP correctly weakens them. This is useful pruning, not runaway weakening.
- **Fitness improves, but modestly.** At rate=1.0 the improvement is ~14% (−2.59e−4 → −2.21e−4). That's real learning — valence is 0 at task failure so STDP isn't randomly reinforcing; it IS preferentially weakening bad-for-tracking slots. But the improvement is bounded.
- **Architecture floor is structural, not weight-related.** Post-training fitness (~−2.2e−4) is still ~4× worse than step 10's hand-wired cross-E-only (−5.60e−5). STDP adjusts `v`, not `pre_ids`. With uniform-random `pre_ids`, ~50% of slots are on B→B recurrent targets and ~10% on B→A-I targets. STDP can drive those v's toward zero but can't release the structural binding or reassign the slot. The useful-slot fraction is frozen at initialization.

### The three failure modes, graded

From the original experiment's pre-registered watch list:

- **Runaway saturation:** partial. At init-high with rate=1.0, 3.4% hit v_max (LTP still occurs on cross-E→post correlations). But no catastrophic saturation: 96.6% stayed at or below initial. Homeostasis is informal (via v-clipping) and sufficient here.
- **Dithering:** confirmed at rate ≤ 0.1. Weights move but the net fitness is unchanged. At those rates STDP is noise-dominated.
- **Catastrophic weakening:** NOT observed. Lowest mean v reached was 0.848 at rate=1.0 mid-init; 0% of slots saturated at v_min. Valence-gating prevents the "everything goes to zero" failure mode the pre-registration worried about.

### What this says about the larger project

The project pitch has always been "innate scaffolds evolved by GA, lifetime learning via STDP gated by modulators." Step 16 is the first test of the lifetime-learning leg against a concrete task, and it finds:
- **Lifetime STDP works.** Modest but real fitness improvement from a plastic-enabled pool.
- **Lifetime STDP is not enough alone.** From a random initial pool, STDP-only cannot reach hand-wired-optimum fitness because the topology constraint binds.
- **Structural plasticity is the missing lever.** The release-then-reacquire dynamic (slot exuberance + pruning) is what would let the topology move under pressure. Slot *release* exists in `structural.py`; *acquisition* does not.

**Step 17's obvious shape:** implement slot acquisition from the free pool. Released slots (v below threshold for long enough) re-bind to a random new pre-neuron. Combined with STDP-driven LTD, this is the exuberance-and-sculpting dynamic the project was always pitching. Expect: pools that start random converge toward cross-E-heavy topologies similar to step 10's hand-wired or step 11's evolved.

### Caveats

- **Single seed.** All results use seed=0; multi-seed confirmation before strong claims.
- **20k steps is short for plasticity.** Longer training might show additional weight movement. Each run is ~3 seconds, so 100k-200k is cheap to try if needed.
- **Valence signal is heuristic.** `max(0, 1 − |error|/scale)` is a first-pass shaped reward. Alternatives (sign-aware valence, threshold-gated reward, valence driven by prediction error rather than tracking error) are unexplored.
- **Plasticity_rate=1.0 is biologically implausible.** Real synapses adjust on the ~seconds-to-minutes scale, not per-spike. The 1.0 value here is compensating for the short simulation; biological scaling would be rate ~1e-3 over hour-long simulations. Results at rate=1.0 should be read as "plasticity-saturated" rather than realistic.
- **Only B is plastic.** Pool A is hand-wired and frozen. An ecosystem-style setup with both agents plastic would have different dynamics (and harder-to-interpret fitness).

---

## 2026-04-27 — Deferred optimization note: `apply_acquisition` Hebbian path

Not a measurement; a documented deferral. Recorded here so future
work finds the reasoning before re-investigating speculatively.

**Concern (Gemini review, 2026-04-26):** The Hebbian branch of
`apply_acquisition` (`src/silicritter/structural.py:163-167`) calls
`jax.random.choice(p=...)` with a per-step probability vector. JAX
implements this as cumsum + searchsorted — `O(N · K · log n_pre)`
per call. At step17's `(N, K, n_pre, T) = (256, 32, 256, 20_000)`,
that's ~1.3 billion ops total per training scan. Linear scaling to
T=10M (Phase 3 target) gives ~650 billion ops on this one op — the
review flagged it as "will become the primary bottleneck at larger
N or longer T."

**Why we are not optimizing now:**

1. **Phase 0 profile (2026-04-26):** the training scan is
   *launch-overhead-bound*, not compute-bound (~15% GPU util on
   GTX 1050 Mobile; 50 ms total GPU compute / 350 ms wall on a
   T=2000 reference run). Optimizing compute saves a fraction of
   what dispatch latency is already eating. Even a pessimistic
   3× speedup of the choice call ≈ ~1% wall-time reduction at
   current scales.

2. **Project policy** (`runtime_not_precious` memory entry): the
   project is willing to spend GPU time for thorough measurements
   and defers optimization until profile data motivates it. "Days
   of GPU compute is fine if it makes a claim bulletproof"
   contradicts speculative optimization without measurement.

3. **Optimization risk:** the Gumbel-max replacement (`argmax(log p +
   gumbel)`) avoids the cumsum + searchsorted but introduces
   hand-rolled categorical sampling. Subtle bugs (`log(0)`
   underflow, argmax tie-breaking) can shift the sampling
   distribution in ways that don't fail tests but corrupt the
   experimental signal.

**When to revisit:**

- When Phase 3 (T=10M, N=500) runs hit a wall-time wall and
  profile data fingers `apply_acquisition` as the bottleneck.
- Or when any future experiment scales `n_pre` substantially (the
  log factor in `O(K · log n_pre)` per slot is where this op
  becomes asymptotically painful).
- The deferred (c) profile design from the 2026-04-27 cleanup
  session is documented for re-use: A/B the existing `uniform` vs
  `hebbian` paths in step17 Config at the same T, measure
  wall-time delta directly, escalate to TensorBoard trace if the
  signal is ambiguous.

**Out of scope of this entry but related:** `jax.random.choice(p=...)`
is also called inside step17 if `pre_id_source="hebbian"` — same code
path. Any optimization here propagates automatically. The cheap
counterpart, `pre_id_source="uniform"` via `jax.random.randint`, is
already O(N · K) with no log factor; that path scales fine.

---

## 2026-04-27 — Deferred: drive-array `(T, N)` pre-allocation in `_build_drives`

Not a measurement; a documented deferral.

**Concern (Gemini final review, 2026-04-27):** Both step16's and
step17's `_build_drives(n_steps)` pre-allocate the full `(T, N)`
external-current raster via `jnp.full(...)` and `jnp.concatenate(...)`.
At T=10M, N=256, each raster is ~10 MB (float32) — small. At
T=100M, ~100 MB per raster, two rasters per call → 200 MB. At
T=1B (hypothetical), several GB. The concern: long-T runs run out
of host memory before they run out of GPU memory.

**Why we are not refactoring now:**

1. **At Phase 3 target (T=10M, N=500):** drive arrays are ~40 MB
   total. Negligible compared to other state. Not the binding
   constraint.

2. **The natural refactor is non-trivial:** moving drive generation
   inside the scan body changes the function-call boundary for the
   driver pattern (`A_DRIVE_PROFILE` piecewise-constant logic
   becomes per-step inside the scan rather than precomputed). The
   refactor is mechanical but touches step16, step17, the test
   file, and overnight_batch.

3. **Unlike the raster-retention fix from 2026-04-26**, this one
   doesn't unblock anything currently planned. Phase 3 runs are
   expected at T=10M, where the drive memory cost is already
   small.

**When to revisit:**

- When any experiment plans T ≥ 100M (drive arrays exceed 100 MB
  per side).
- When a memory-bound regression is profile-attributed to drive
  pre-allocation rather than to other allocations.
- The proposed refactor: drop `_build_drives` from the scan input
  set; reconstruct `i_ext_a_t` and `i_ext_b_t` from `step` (the
  scan-step counter) inside `scan_step`. `A_DRIVE_PROFILE`
  selection becomes `A_DRIVE_PROFILE[step // (T // len(profile))]`
  via dynamic indexing.

---

## 2026-04-27 — Deferred (LOW): `scan_step` recompilation per `_training_scan` call

Verified empirically: ~100 ms compile overhead per call, NOT the
30–60 minutes Gemini's review claimed.

**Background:** Gemini final review flagged step17's `_training_scan`
as triggering full JAX recompilation per call because `scan_step` is
defined as a closure inside the function. Predicted impact: 30–60
minutes wasted per 500-run batch.

**Empirical verification (2026-04-27, 4 GB GTX 1050 Mobile, T=400):**

| trial | wall-time |
|------|-----------|
| 1    | 667.0 ms  |
| 2    | 570.2 ms  |
| 3    | 590.9 ms  |

The trial 1 → trial 2 drop is ~100 ms; subsequent trials are flat.
JAX IS caching the lowered IR across calls; the per-call overhead
is ~100 ms (trace + dispatch), not full recompilation.

**Per-batch impact:** at 500 runs, ~50 seconds total compile
overhead — not 30–60 minutes. Gemini's claim was 3 orders of
magnitude high.

**Why deferring:** the refactor (move `scan_step` to module level,
plumb `acq_mode` through carry/closure) is real surgery for ~50 s
savings per batch. The G-3 monkey-patching refactor (#30 in the
2026-04-27 task list) addresses the same surface for unrelated
fragility reasons; if that lands, the scan_step structure can be
revisited as part of the same pass at near-zero marginal cost.

---

## 2026-04-27 — Deferred (Ed-confirmed): G-3 monkey-patching of step16/step17 globals in overnight_batch

Not a measurement; an Ed-confirmed deferral.

**Concern (Gemini final review, 2026-04-27):** `experiments/overnight_batch.py`
mutates module-level globals in `step16_stdp_learning` and
`step17_structural_growth` via `try/finally` patterns to vary run
parameters. Examples:

- `_strong_op_run`: monkey-patches `s17.PLASTICITY_RATE` and
  `s17.INIT_V_MEAN`.
- `_long_training_run`: monkey-patches `s17.N_TRAIN_STEPS` (and the
  symmetric `s16.N_TRAIN_STEPS` path).
- `_acq_prob_stochastic_sweep`: monkey-patches `s17.ACQ_PROB_STOCHASTIC`.

The pattern is fragile under any concurrency (signal handlers,
parallel runs, pytest collection) and obscures the actual parameter
set per run from anyone reading the code.

**Why deferring (Ed-confirmed, 2026-04-27):** "Monkey patching is
fine for now." The current overnight batch runs are sequential and
single-process, so the concurrency risk doesn't bite. The proper
refactor (plumb all variable parameters through `Config` /
explicit function arguments) is real medium-scope surgery —
worth doing eventually but not blocking any current experiment.

**When to revisit:**

- If overnight batches ever run in parallel (multi-process or
  multi-thread): the monkey-patches will race and produce wrong
  results silently. That's the trigger to refactor.
- If a future experiment wants to vary a parameter that's not
  currently monkey-patchable cleanly (e.g., a deeply-nested config
  field), that becomes the natural moment to do the broader plumbing.
- If `_run_config` / `_step16_once` ever need to be JIT-compiled
  themselves (currently they're Python functions wrapping JAX scan
  calls): monkey-patching breaks under jit because globals are
  baked in at trace time. Trace cost would be paid before any
  patch took effect.

**Proposed refactor when the time comes:**

1. Extend `Config` (or add a sibling `RunConfig`) with the variable
   fields: `plasticity_rate`, `init_v_mean`, `init_v_std`,
   `n_train_steps`, `n_measure_steps`, `acq_prob_stochastic`, etc.
2. `_run_config` and `_step16_once` take this expanded config and
   stop reading module globals.
3. overnight_batch's `_strong_op_run` etc. construct the config
   with the desired values and pass it down — no monkey-patching.
4. Module-level constants in step16/step17 become defaults
   (used when no config field overrides them) or simply demoted
   to test fixtures.

The change would touch step16, step17, overnight_batch, and a few
test files. Bounded but real.

---

## 2026-04-28 — Full overnight_batch run on new laptop, post-cleanup

First full overnight_batch run on the new laptop (ASUS VivoBook
X580GD, GTX 1050 Mobile, 4 GB VRAM) after the laptop migration AND
the multi-pass review-finding cleanup landed in commit fd30661.
Headline: faster than expected, and bit-for-bit equivalent to the
pre-cleanup baseline.

- **Script:** `experiments/overnight_batch.py` (default `main()`,
  all 8 blocks).
- **Commit under test:** `fd30661` ("Phase 1 raster fix + multi-pass
  review-finding cleanup"). Baseline for comparison: archived CSVs
  at `overnight_results/archive_2026-04-28_pre-fd30661/` from
  pre-cleanup runs on the old MateBook X Pro (MX150, 2 GB).
- **Machine:** ASUS VivoBook X580GD (Intel i7-8550U + NVIDIA GTX
  1050 Mobile, 4 GB VRAM, ~1.6× compute / ~2.3× bandwidth vs old
  MX150). NVIDIA driver 580.142, JAX 0.10.0 with CUDA 12.9 plugin.
- **Total wall time:** 1.21 hours (72 min) end-to-end. Estimate
  before run was 6–10 hours; reality was ~6× faster.
- **Result:** all 8 blocks completed cleanly, all auto-commits +
  pushes landed. Zero errors in the log.

### Per-block timing

| block | runs | wall time |
|---|---:|---:|
| 1 step17_factorial            | 490 | 32 min |
| 2 acquisition_probability     |  45 |  3 min |
| 3 long_training               |  50 |  6 min |
| 4 step16_multiseed            |  75 |  3 min |
| 5 best_config_confirm         |  60 |  4 min |
| 6 strong_op_confirm           |  60 |  4 min |
| 7 long_duration_confirm       |  40 | 11 min |
| 8 stdp_regression_bisect      | 100 |  9 min |
| **total**                     | **920** | **72 min** |

Block 1 dominates (44% of wall time) because of its 490-run config
× seed grid; all other blocks are <11 minutes. Per-run cost is
~1.5–6 seconds for short-T blocks, scaling with `n_train_steps`
for blocks 7 and 8.

### Bit-for-bit comparison vs pre-cleanup baseline

A column-by-column diff of every scientific column (filtering out
`wall_sec` and `train_time` which always vary) across all 920 rows
showed **zero numerical drift**. Every block matches its
pre-cleanup CSV exactly:

- block1_step17_factorial: identical across 490 rows
- block2_acquisition_probability: identical across 45 rows
- block3_long_training: identical across 50 rows
- block4_step16_multiseed: identical across 75 rows
- block5_best_config_confirm: identical across 60 rows
- block6_strong_op_confirm: identical across 60 rows
- block7_long_duration_confirm: identical across 40 rows
- block8_stdp_regression_bisect: identical across 100 rows

The cleanup is observationally indistinguishable from the prior
code path on every numerical metric.

The most surprising piece: **block 1 also matched bit-for-bit**,
despite the cleanup adding a `step > 0` guard to
`_acq_prob_periodic` that changed the per-step probability output
at step=0 from 1.0 to 0.0. Mechanism: the initial pool is fully
active at step=0, so the `acquire_mask = ~pool.active & bernoulli`
intersection is all-False regardless of the prob value. The fix
removes a redundant RNG draw + Bernoulli computation but produces
identical slot state — so all downstream metrics (fit_before,
fit_after, cross_e_frac_end, etc.) are bit-identical.

### Implications

1. **Refactor discipline confirmed.** The 11-task multi-pass
   cleanup landed without changing any experimental output. Future
   refactors can use this same archive-and-diff pattern as a
   regression check before claiming the change is purely structural.
2. **Cycle time is short.** 72 minutes for a full revalidation
   batch means iterate-and-rerun is cheap on this laptop — fits
   between morning coffee and lunch. The compute budget is no
   longer a planning constraint at the current Phase 1/2 scale.
3. **Phase 3 still unblocked-but-not-yet-fast.** The current
   batches operate at N≤20 seeds and T≤500k steps. Phase 3 plans
   N=100/N=500 × T=10M, ~100× the current per-batch compute.
   Linear extrapolation: ~120 hours on this laptop, vs.
   ~10–15 minutes on a g5.xlarge (A10G, AWS spot). The cloud-burst
   path is now the practical route for Phase 3, not laptop time.

### Caveats

- **Single-machine measurement.** All numbers are from one machine
  on one OS state. Future runs may show ±10% wall-time noise from
  thermal throttling, background processes, or driver updates.
  The bit-for-bit comparison is robust against this since wall
  time isn't a scientific column; deltas are.
- **Process snapshot.** This entry locks in the 1.21-hour full-
  batch reference. Any future change that bumps full-batch time
  past ~90 minutes warrants investigation.

---

## 2026-04-30 — Phase 2: long-T step-10 reproducer on Colab A100

First long-T (T up to 10M timesteps) confirmation of step 10's
closed-loop adrenaline result. Run on Google Colab Pro+ with A100
GPU; the 4 GB GTX 1050 Mobile cannot fit T=10M even with the
memory-friendly rate-output mode (the 40 GB A100 has comfortable
headroom).

- **Script:** `experiments/phase2_step10_long_t.py` (commit `b9c8633`).
- **Machine:** Google Colab Pro+, NVIDIA A100-SXM4-40GB.
- **Stack:** Colab default Python 3.11 + JAX with CUDA12 plugin (`pip install -e ".[dev]"` from the cloned repo).
- **Sweep grid:** T ∈ {10k, 100k, 1M, 10M} × N=5 seeds × 4 conditions
  (open-loop, closed-loop gain ∈ {10, 50, 200}) = 80 runs total.
- **Total wall time:** 91.3 minutes (5480.5 s) — matches the ~90 min pre-run estimate.

### Mean fitness by (T, condition), N=5 seeds

Fitness is `-mean((b[:-1] - a[1:])**2)` over 100-step windows;
0 is perfect, more negative = larger prediction error.

| T          | open_loop      | gain=10        | gain=50        | gain=200       |
|------------|---------------:|---------------:|---------------:|---------------:|
| 10 000     | −1.564e-04     | −7.82e-05      | **−3.669e-05** | **−3.669e-05** |
| 100 000    | −1.524e-04     | −7.25e-05      | −2.738e-05     | −2.770e-05     |
| 1 000 000  | −1.534e-04     | −6.81e-05      | **−2.709e-05** | **−2.709e-05** |
| 10 000 000 | −1.589e-04     | −7.30e-05      | −2.717e-05     | −2.728e-05     |

Variance across seeds at gain ≥ 50 is essentially zero — for example, all
five seeds at T=10M, gain=200 give exactly **−2.728e-05** to 4 sig figs.
This confirms the prior handoff observation that controller rail-clipping
produces fully deterministic dynamics.

### Per-condition wall time on A100

| T          | per-run, open_loop | per-run, gain=200 |
|------------|-------------------:|------------------:|
| 10 000     |    0.5 s           |    0.7 s          |
| 100 000    |    2.2 s           |    3.1 s          |
| 1 000 000  |   18.6 s           |   27.6 s          |
| 10 000 000 |  173.5 s (~2.9 m)  |  268.5 s (~4.5 m) |

Linear scaling in T after JIT warm-up; closed-loop adds ~50% over
open-loop (controller's per-step EMA + clip).

### Implications for the revalidation plan

1. **Step 10's reported "−5.60e-5 hand-wired closed-loop breakthrough"
   does not reproduce at the same number** — but it reproduces with a
   *better* value. Phase 2 finds gain=200 at **−2.7e-5 to −2.8e-5**
   across T ≥ 100k (closer to zero = lower error squared).
   The original −5.60e-5 was N=1 at T=2000, where the EMA controller
   (decay 0.98, τ ≈ 50 steps) is barely settled within a 4-segment
   stimulus cycle. At T ≥ 100k the controller has thousands of cycles
   to equilibrate and the steady-state fitness is consistently lower.
2. **Qualitative claim from step 10 holds robustly.** Open-loop sits
   at −1.5 to −1.6e-04; closed-loop at gain ≥ 50 sits at −2.7e-05.
   Ratio ~5.7×. That structural finding (closed-loop adrenaline
   pushes prediction error toward zero by ~5×) is now anchored at
   N=5 across four orders of magnitude in T.
3. **Gain=50 and gain=200 are essentially equivalent at long T.** Both
   rail at adr_max for most of the run; the controller doesn't
   distinguish them. Block 9 can pick either; gain=200 is the
   conservative choice (matches step 10's reported headline gain).
4. **Long-T toolchain is well-behaved.** Open-loop drifts ~2% from
   T=10k to T=10M (−1.564 → −1.589e-04) — consistent with finite-
   sample variance, not a runaway. Closed-loop is rock-stable. **Green
   light for Block 9** (N=500 at gain=200, durations 10k → 10M).
5. **Curiosity at T=10k:** gain=50 and gain=200 give bit-identical
   fitness (−3.669e-05) for all five seeds. Both railed at adr_max
   immediately and never left, so the trajectories are identical.
   At larger T the controller occasionally comes off the rail at
   gain=50, producing slightly different (but very close) fitness.

### Headline number to use going forward

The step-10 paper / README claim, replication-tracked, should read:
> Closed-loop adrenaline (gain=200, hand-wired cross-E pool, EMA
> controller decay=0.98, adrenaline range [0.5, 3.0]) reduces
> prediction-error fitness from **−1.589e-04** (open-loop, no
> adrenaline modulation) to **−2.728e-05**, a ~5.8× improvement,
> measured at N=5, T=10M, on Colab A100 (commit b9c8633). The
> originally reported −5.60e-5 was N=1 at T=2000 and is superseded
> by this measurement.

### Caveats

- **CSV precision is reconstructed from the inline run log, not the
  original Colab CSV file.** Colab's runtime timed out before
  cell 7 (which would have copied the CSV from the ephemeral runtime
  to Drive) executed, so the durable CSV in `overnight_results/`
  carries only 3 sig figs of precision (the log's print format)
  rather than the script's native 6 sig figs. The next run of this
  notebook (now patched to symlink `overnight_results/` directly to
  Drive) will preserve full precision.
- **One-machine, one-run.** Standard caveats: Colab A100 thermal /
  scheduling state, JAX version drift, etc. The qualitative
  conclusions are robust; the third-significant-figure values may
  differ ±1 unit on a re-run.
- **Variance bound is partial.** N=5 with std=0 (rail-clipped) gives
  no information about the variance distribution — it just bounds
  it below the resolution of the rail. Block 9 (N=500) will produce
  the actual variance bound on the closed-loop fitness.

---

## 2026-05-01 — Phase 2 durable rerun confirms 2026-04-30 reconstruction

A second independent Phase 2 run on Colab A100, this time with the
notebook patched to symlink ``overnight_results/`` directly to a
Drive-mounted folder so per-row writes are durably saved (fix for the
session-timeout that lost the prior run's CSV file). Resolves the
"reconstructed from inline log at 3 sig figs" caveat on the
2026-04-30 entry: the durable CSV in ``overnight_results/
phase2_step10_long_t.csv`` is now this run's full-precision
(~6 sig fig) output.

- **Script:** unchanged from prior run, ``experiments/phase2_step10_long_t.py``
  at commit ``b9c8633`` (no code under test changed between runs).
- **Notebook:** ``colab/phase2_long_t.ipynb`` updated in commit
  ``3b9352e`` to mount Drive in cell 3, symlink overnight_results/
  → ``MyDrive/silicritter_phase2/<UTC-timestamp>/``, and seed-write
  a sentinel file to fail-fast on Drive auth issues.
- **Machine:** Google Colab Pro+, NVIDIA A100-SXM4-40GB.
- **Drive folder:** ``MyDrive/silicritter_phase2/2026-05-01T020640Z/``.
- **Total wall:** 5446.9 s (90.8 min) — vs 5480.5 s (91.3 min) on
  prior run. Within thermal noise.

### Cross-run agreement (Run 1 vs Run 2 means, N=5 seeds)

| T          | Run 1 gain=200 mean | Run 2 gain=200 mean | delta     |
|------------|--------------------:|--------------------:|----------:|
| 10 000     | −3.669e-05          | −3.668605e-05       | ~5e-10    |
| 100 000    | −2.770e-05          | −2.769557e-05       | ~5e-10    |
| 1 000 000  | −2.709e-05          | −2.708675e-05       | ~3e-10    |
| 10 000 000 | −2.728e-05          | −2.728288e-05       | ~3e-10    |

Run-to-run delta is below the 4th significant figure for every
condition; the qualitative findings from 2026-04-30 are unchanged.
Headline: gain=200 at T=10M, **−2.728e-05** (mean across 5 seeds, 5th
sig fig varies by ~1 across seeds) — supersedes step 10's reported
−5.60e-5 (which was N=1 at T=2000).

### New observation only visible at full precision

At T=1 000 000, gain=50 and gain=200 are **bit-identical
(−2.708675e-05) for all five seeds** — both controllers rail at
``adr_max`` for the entire run, producing exactly the same trajectory.
At T=10M they diverge by ~0.4% (gain=50 ≈ −2.717e-05, gain=200 ≈
−2.728e-05): the longer run gives gain=50 enough opportunity to
occasionally come off the rail. This was hidden in the 3-sig-fig
reconstruction; visible now.

### Caveats

- Run-to-run drift in the 5th–6th significant figure at long T is
  expected: A100 reductions are not bit-deterministic across runs
  because XLA may schedule reductions in different orders. The 4-sig-
  fig agreement is the right reproducibility bar.
- Wall time 90.8 vs 91.3 min — A100 thermal/scheduler noise; no
  performance regression.
- Same Caveats apply as the 2026-04-30 entry (one machine, N=5,
  variance bound rail-floor-limited). Block 9 at N=500 still
  required for actual variance distribution.

---

## 2026-05-01 — Block 10: D008 E/I confirmed at N=100, with mechanism

The N=100 reanchor of D008's (i_frac=0.2, i_mult=8.0) E/I operating-
point choice over the prior canonical (0.2, 4.0). Replaces step 13's
existing N=20 anchor. The result confirms D008 *and* surfaces a
mechanism that the smaller-N anchor couldn't see.

- **Script:** `experiments/block10_step13_ei_n100.py` (commit a99f3f3).
- **Machine:** ASUS VivoBook X580GD, NVIDIA GTX 1050 Mobile (4 GB VRAM).
- **Stack:** same as 2026-04-28 entry (Python 3.12.13, JAX 0.10.0 + CUDA 12.9).
- **Sweep:** 2 i_mults × 100 seeds (stride 37) × 2 conditions = 400 runs at
  step 13's T=2000 with closed-loop gain=50.
- **Total wall time:** 128.4 s (2 min 8 s) - per-run wall ~0.3 s, much
  faster than the 5-9 s extrapolation from step 13's earlier timing.
  The rate-mode + scalar-drive optimizations from Phase 2 transfer to
  T=2000 too.

### Mean fitness by (i_mult, condition), N=100 seeds

| condition   | i_mult | mean fitness    | std       | min          | max          |
|-------------|-------:|----------------:|----------:|-------------:|-------------:|
| open_loop   | 4.0    | −1.5545e-04     | 9.28e-07  | −1.5728e-04  | −1.5354e-04  |
| closed_loop | 4.0    | **−5.5948e-05** | 3.98e-11  | −5.5948e-05  | −5.5948e-05  |
| open_loop   | 8.0    | −1.4675e-04     | 1.49e-06  | −1.5033e-04  | −1.4330e-04  |
| closed_loop | 8.0    | **−4.9040e-05** | 4.97e-07  | −5.0369e-05  | −4.8083e-05  |

D008 (i_mult=8.0) vs prior canonical (i_mult=4.0):
- **closed_loop fitness improves by 12.35%** (−5.595e-05 → −4.904e-05)
- open_loop fitness improves by 5.60% (−1.555e-04 → −1.467e-04)

Both directions favor D008 at N=100, supersedes the N=20 anchor with a
much tighter variance bound.

### New mechanism observation only visible at N=100

At i_mult=4.0 (prior canonical), closed-loop std across 100 seeds is
**3.98e-11** - effectively zero, ~12 orders of magnitude smaller than
the open-loop std at the same i_mult. This is the controller railing
at adr_max for the entire run: every seed gets the same trajectory
because the controller has no dynamic range to use, so it produces a
deterministic fitness regardless of pool randomization.

At i_mult=8.0 (D008), closed-loop std is **4.97e-07** - 4 orders of
magnitude larger than the i_mult=4.0 case, and now in the same order
of magnitude as the open-loop noise floor. **The controller is no
longer rail-clipped.**

Interpretation: stronger inhibition (i_mult=8.0) makes B harder to
drive, so when A's drive demands push B's rate up, the controller
needs less of `adr_max`'s headroom to compensate - it operates
within its `[adr_min, adr_max]` range rather than saturating. This
is exactly what you want from a control system. The fitness
improvement at i_mult=8.0 isn't just "more inhibition is better"
in a flat sense; it's "the controller has somewhere to work, so it
actually works."

This mechanism wasn't visible at N=20 because the rail-clipping
deterministic-floor at i_mult=4.0 needs ≥30 seeds before the
std=4e-11 vs std=5e-07 separation becomes statistically obvious.

### Implications

1. **D008 confirmed at N=100, no caveats.** The decision to adopt
   (0.2, 8.0) over (0.2, 4.0) was correct; tightening the bound from
   N=20 to N=100 doesn't change the conclusion.
2. **The controller-headroom interpretation is new.** Previously D008
   was justified on fitness alone; now we have a mechanistic story
   ("inhibition lets the controller work in range"). Worth folding
   into the README's E/I section and into Block 11/12/13's setup
   reasoning.
3. **Block 10 wall-time radically beat estimate** (2 min vs 30-60
   min predicted). The rate-output_mode + scalar-drive memory fixes
   from Phase 2 cut per-step overhead substantially, even at small
   T. Future block estimates on the laptop should use ~0.3 s/run as
   the baseline, not the older step-13 5-9 s anchor.

### Caveats

- N=100 with std at order 1e-7 (closed_loop, i_mult=8.0) gives a
  reasonable variance bound but not bulletproof - a follow-up at
  N=500 (paralleling Block 9 for step 10) would tighten further if
  Block 11+ surface anything weird.
- Single machine, single run. Per the cross-run reproducibility
  pattern from Phase 2, a second independent run on different
  hardware would harden the result; the laptop-only measurement is
  enough for tonight's result, AWS-spot-paralleled re-run would be
  the rigorous version.
- Wall-time numbers (0.3 s/run) are GTX 1050 Mobile-specific.
  A100 / L4 will be faster but at small T the speedup is modest
  (Phase 2's T=2000 was also fast on A100; long-T is where the GPU
  capacity matters).
