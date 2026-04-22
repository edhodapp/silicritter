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
