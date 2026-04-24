# silicritter

An analog-neural-silicon exploration that aims to emulate **living-creature neural networks** — specifically the *developmental* arc of plasticity (synaptic exuberance in early life, followed by lifelong sculpting) rather than the adult steady state that most neuromorphic work targets.

Exploratory / blue-sky research. Single author. No chip yet.

---

## Goals

### Long-term aim

A silicon substrate, in standard CMOS (SkyWater 130nm open PDK), that runs biologically-grounded neural dynamics with:

- **Structural plasticity** via a per-postsynaptic-neuron slot pool — synapses are not a fixed N×N matrix but a pool of discrete slots that can be formed, strengthened, weakened, and eventually released. Most analog-neuromorphic chips approximate pruning by driving weights toward zero; silicritter aims at genuine reclamation and rebinding.
- **Multi-modulator broadcast chemistry**. Not a single scalar "reward" line. A palette of broadcast signals — valence (dopamine-like, phasic), adrenaline (norepinephrine-like, neural-gain modulator), later cortisol / oxytocin / vasopressin / BDNF analogs — each with its own timescale, source, and plasticity targets. Added one at a time as specific experiments require them; not pre-committed as a full suite.
- **Innate scaffolds optimised by genetic algorithms**, lifetime scaffolds learned by spike-timing-dependent plasticity gated by the modulator palette. Two-timescale architecture: outer loop is evolution (offline, tunes the scaffold), inner loop is lifetime learning (on-chip / in-sim, tunes weights in the scaffold).
- **Long-term directional pull toward social intelligence.** Social-like dynamics appear at surprisingly simple substrates (slime-mold aggregation, bacterial quorum sensing), so silicon-scale social behaviour need not wait for brain-scale networks. Dogs are the aspirational model per Pat Shipman's *The Invaders* (2015); the near-term tractable targets are minimal paired-agent feedback, gaze-following, and pair-bonding circuits.

### Build methodology

Four separable phases of increasing concreteness; architecture is finalized at each phase before lowering to the next. Iteration cost scales roughly hours → days → weeks per phase, so premature lowering spends the budget in the wrong place. This principle is inherited from a Verilog instructor who was one of the early Gateway Design Automation employees — Verilog was a simulation language before synthesis existed, and using the full expressive power of the sim to get the design right before starting RTL is what that lineage teaches. The same argument has cleaner form in Naur's 1985 *Programming as Theory Building*, Grothendieck's "rising sea" metaphor, and Vincent Lextrait's *Software Development or the Art of Abstraction Crafting* (see `DECISIONS.md` D006 for the full statement).

Phases:

1. **Idealised behavioural simulation** (JAX). Clean abstractions, ideal numerics, no process noise. *Current phase.*
2. **Noisy behavioural simulation** (still JAX). Process variation, transistor mismatch, capacitor leakage, quantized per-slot v. Validates architectural robustness against silicon reality before any circuit-level work. This is where the Thompson-trap (from the 1996 "evolved FPGA circuit exploits parasitics") is pre-emptively disarmed.
3. **Circuit-level validation** (ngspice). Specific analog primitives — LIF cell, slot cell, valence broadcast, multi-neuromodulator bias lines. Primitive-by-primitive, not the whole network.
4. **Layout and tape-out** (Magic + Xschem + KLayout + DRC/LVS). Efabless chipIgnite is the default fab path; TinyTapeout is also license-compatible.

---

## Present status (Phase 1, steps 1–5 complete)

What exists today is an idealised JAX simulation of a single plastic spiking network with one chemical-signal modulator beyond valence, driven by a direct-encoding genetic algorithm that evolves initial scaffolds. No chip, no circuit-level validation yet.

### Modules (`src/silicritter/`)

- **`lif.py`** — Leaky integrate-and-fire neuron forward simulation. `init_state`, `integrate_and_spike`, `step` (dense-weight variant), `simulate`. Module-level invariant asserts on V_REST / V_THRESH / V_RESET and positive τ_m / dt. 100% branch coverage.
- **`slotpool.py`** — Slot-pool synapse representation. `SlotPool` NamedTuple with per-slot `pre_ids`, `v`, `plasticity_rate`, and `active`; synaptic input computed via gather-and-sum over the pool. Effective dense-matrix view available for validation. Forward sim (`step`, `simulate`) byte-exactly matches the `lif.py` dense path at matched weights. 100% branch coverage.
- **`plasticity.py`** — Three-factor STDP on the slot pool, with pre-decayed eligibility traces (Song/Miller/Abbott convention) and two modulators:
  - `valence` — scalar, gates STDP sign and magnitude (dopamine-like phasic three-factor signal).
  - `adrenaline` — scalar, multiplies `i_total` at the LIF integration step (norepinephrine-like gain modulation, Aston-Jones & Cohen 2005).

  Weights clipped to `[v_min, v_max]`. Slot `plasticity_rate = 0` means innate / hardwired. 100% branch coverage.
- **`ga.py`** — Direct-encoding genetic-algorithm primitives. `Genome` (three parallel arrays per individual), `random_genome`, `random_population`, `decode_to_pool`, `tournament_select` (with-replacement sampling, documented), `uniform_crossover` (with a Baldwin-interference caveat documented), `mutate` with bound preservation. 100% branch coverage.

### Experiments (`experiments/`)

Each step's self-contained runnable. See `perf_history.md` for measured numbers on the reference machine (Huawei MateBook X Pro 2018, i7-8550U + NVIDIA MX150, 2 GB VRAM).

- **`step02_throughput.py`** — LIF forward-sim throughput on dense weights. Baseline: 8.35e6 neuron-steps / s, 40 Hz firing rate, N=1024, T=10 000.
- **`step03_slotpool_throughput.py`** — Same scenario with slot-pool synapses (K=64). Baseline: 4.17e7 neuron-steps / s (~5× over dense).
- **`step04_plastic_throughput.py`** — Slot pool + three-factor STDP + valence + adrenaline. Baseline: 1.78e7 neuron-steps / s (~2.4× slower than step 3 for the plasticity overhead). Mean v drifts 0.040 → 0.093 over 10 s simulated time; at least one slot saturates to `v_max`.
- **`step05_ga_target_rate.py`** — GA outer loop evolving scaffolds to track a time-varying target firing rate driven by a piecewise adrenaline signal. Inner-loop sims are `vmap`ped across the population. ~22 ms per generation on MX150; 80 generations in ~1.8 s at N=32, K=8, T=2 000, pop=48. Three of four adrenaline segments track within ~8 Hz; one segment exposes a floor problem in the multiplicative-gain mechanism (low adrenaline pushes some cells below V_THRESH, uncorrectable by weight tuning).

### Tests (`tests/`)

31 behavioural and unit tests. `pytest --cov --cov-branch` reports **100% branch coverage** across all four library modules.

### Quality gates

Every commit that changes functional code runs through:

- `flake8 --max-complexity=5 --max-line-length=80`
- `mypy --strict` (all type annotations checked)
- `pylint` with the Google Python Style Guide `pylintrc`
- `pytest --cov --cov-branch` (100% branch coverage target)

Plus independent code review by an isolated Claude subagent (clean context, no project knowledge) and advisory review by Gemini, both running from `~/tools/code-review/`.

### What is deliberately not yet implemented

- **Structural slot release / acquisition** — "exuberance and pruning" proper. Slots with `v = v_min = 0` are functionally silent but still structurally bound; the free pool is unused. This is the missing piece before the project can claim genuine developmental dynamics.
- **Additional chemical signals** — cortisol, oxytocin, vasopressin, BDNF, NO. Added one at a time as specific experiments need them.
- **Noisy behavioural sim (Phase 2)** — process variation, transistor mismatch, capacitor leakage, quantization. Starts when Phase 1 architecture is declared frozen.
- **Circuit-level validation (Phase 3)** and **layout / tape-out (Phase 4)**.
- **Indirect-encoding GA** — direct encoding blows up at N > a few hundred; indirect (CPPN / developmental rules, HyperNEAT-style) is the scaling path. Not needed yet at toy scale.
- **Multi-agent / social scenarios** — any social-intelligence target implies at minimum two silicritter instances whose spike outputs feed each other's inputs. Not yet wired.

---

## Running locally

Developed on Python 3.12 with a virtualenv in `.venv/`:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Dependencies pull in `jax[cuda12]` with bundled NVIDIA CUDA 12 runtime libraries. An NVIDIA GPU (CC 6.0+, ~1 GB free VRAM) is required for the GPU-backed experiments; everything falls back to CPU if no GPU is present but will be much slower.

Run the experiments:

```bash
.venv/bin/python experiments/step02_throughput.py
.venv/bin/python experiments/step03_slotpool_throughput.py
.venv/bin/python experiments/step04_plastic_throughput.py
.venv/bin/python experiments/step05_ga_target_rate.py
```

Run tests:

```bash
.venv/bin/pytest tests/
```

---

## Project documentation

- **`DECISIONS.md`** — architectural decision log. Immutable entries with supersession annotations on deprecated entries. Read this for the history of why the project looks the way it does.
- **`CLAUDE.md`** — project-local instructions for Claude Code sessions.
- **`perf_history.md`** — durable performance log. One entry per significant experiment run, with measured numbers and honest commentary on what was and wasn't demonstrated.
- **`SESSION-HANDOFF.md`** — origin context from the 2026-04-21 session that spawned this project.
- **Dev log (below)** — chronological narrative of what was explored and why, including blind alleys that got backed out. Lighter-weight than `DECISIONS.md` (no immutability constraint) and broader than `perf_history.md` (no performance-measurement focus). Read this for the research arc.

---

## Dev log

Append-forward narrative of exploration. Each entry is dated and
describes what was tried, what was learned, and the direction it
pushed the project. Unlike `DECISIONS.md` this is not immutable —
but entries are not retconned either; if something turns out wrong,
a later entry says so rather than editing the earlier one.

### 2026-04-23 — Fractional-noise stimulus direction

Motivated by Ed's parallel `~/math/fraccalc` exploration, pivoting
the "Step 5: prediction task" work toward stimulus models with
learnable temporal structure driven by fractional Gaussian noise.

**Why this direction:** the current `A_DRIVE_PROFILE` (four flat
levels held for 500 ms each) is predictable only at segment
transitions, and even then B has no history-based information to
exploit. Calling the existing MSE fitness a "prediction" metric
is honest only by the index offset; mechanically the task is
tracking with lag. To make prediction substantive, A's drive needs
temporal structure B can learn from — the simplest principled
choice is a stochastic process with tunable memory length.

**fGn specifically because:** the Hurst parameter H controls the
decay of the autocovariance kernel from white noise (H=0.5,
memoryless) to strong long-range dependence (H near 1, heavy-tailed
memory). The optimal predictor for fGn has a known closed-form
floor, giving us a theoretical target rather than just relative
baselines. And it connects cleanly to memory-kernel-style analysis
that Ed is working through independently in fraccalc, so stimulus
work and math work can inform each other.

**Landed:** `src/silicritter/fracnoise.py` — Davies-Harte FFT
synthesis of unit-variance fGn, O(N log N). `fgn_davies_harte(n,
hurst, rng)` returns a sample path; `fgn_drive_trace(T, N, H, mean,
std, rng)` builds a drive trace shaped (T, N) for use as
`i_ext_a`. 8 tests at 100% branch coverage, empirically verified
that autocorrelations match theoretical values at H=0.3, 0.5, 0.7.

**Next:** Step 14 experiment — swap `A_DRIVE_PROFILE` for an fGn
drive trace at a few Hurst values, rerun the step 10 closed-loop
architecture, measure how fitness changes as H sweeps across the
memory-length range. Comparison is against the step-function
baseline (−5.6e−5 closed-loop on the static ceiling) and
eventually against the Wiener-Kolmogorov optimal predictor floor
for the same stimulus.

**Backing-out clause:** this is exploratory. If Step 14 shows that
the current architecture doesn't distinguish between H=0.5
(white-noise, unpredictable) and H=0.7 (persistent, predictable) —
i.e., B's fitness is indistinguishable across H — then B has no
capacity to exploit memory structure, and the fractional-noise
direction is a blind alley for *this* architecture. Would then
pivot to examining what architectural changes (fractional EMA in
controller? non-zero plasticity + valence trace driven by
prediction error?) might open prediction up.

### 2026-04-23 — Step 14 result: not backing out, refining

fGn stimulus runs cleanly through the step-10 architecture, all four
H values (0.3, 0.5, 0.7, 0.9) produce sensible firing regimes, and
the closed-loop controller delivers a 20–30× tracking improvement
over open-loop at every H.

**The back-out clause is not triggered.** Fitness does depend on H.
Closed-loop tracking is 3× worse at H=0.7 than at H=0.5, and the
prediction-minus-tracking gap at H=0.9 (−1.71e−5) is 4× the gap at
H=0.3 (−4.67e−6). The architecture is not indifferent to memory
structure in the stimulus.

**But the dependence isn't what I predicted.** I expected the gap to
scale with `(1 − lag1)` — small gap at high-H (persistent signal,
easy to predict by just reporting current), large gap at low-H
(antipersistent, hard). Observed is the opposite. The mechanism is
that higher H inflates A's per-window rate variance (less within-
window averaging-out), which gives B a harder target to track even
though the signal is smoother. The controller's fixed-τ EMA fights
this badly. See `perf_history.md` (Step 14 entry) for the full
table and honest interpretation.

**Hypothesis for the mechanism behind "real" prediction in this
substrate: not present.** Controller EMA integrates on a 50 ms
time constant tuned to response dynamics, not to A's temporal
structure. B's pool is fixed (plasticity_rate=0). Valence trace
is zero. Whatever H-dependence we see is the architecture's
dynamics stumbling through different stimulus-variance regimes,
not genuine predictive learning.

**Next (Step 5c):** compute the Wiener-Kolmogorov optimal
predictor error for fGn at each H. That's the theoretical floor
for "best any predictor could do." If B's observed residuals sit
far above that floor, there's real architectural headroom to
chase. If B is already near the floor, prediction isn't
improvable within this substrate and a genuinely memory-capable
architecture (fractional controller, recurrent B, or learned
plasticity with prediction-error valence) is what's needed.

### 2026-04-23 — Step 15 result: architecture is 2× above the floor

Computed the Wiener-Kolmogorov floor empirically for each (H,
condition) in step 14 via a new `src/silicritter/wk.py` module
(Durbin-Levinson + windowed-fGn autocovariance; 37 tests across
5 independent cross-check paths including Monte Carlo agreement
between the Davies-Harte synthesizer and the theoretical
windowed autocov).

Finding: **closed-loop B sits at 1.9–2.4× above the WK floor at
every H.** For an architecture with no explicit memory (no
recurrent state, no plasticity, controller EMA only 50 ms long),
this is surprisingly close to the theoretical best linear
predictor. Open-loop is 12–19× above — the controller's implicit
smoothing is doing most of the predictive work.

**Re-reading of the step 14 result:** the gap I was chasing is
smaller than I'd framed it. The WK floor is only 9–12% below
A's window variance — the *absolute* room for prediction-based
improvement is small to begin with, because stationary fGn at
these H values isn't *that* predictable. Of that small room, B
is already capturing roughly half.

**Where this leaves the fractional-stimulus direction:** the
"memory upgrade will close a big gap" pitch is weak. Best-case
memory-capable architecture might close half the remaining gap,
so maybe a 1.5× improvement in prediction MSE. That's bounded
enough that architecture upgrades are probably not the biggest
lever.

**What IS worth pursuing** (if Ed wants to continue the
stimulus-centered direction):
  1. *Non-stationary A*: fGn with abrupt parameter changes, or
     a drifting mean. Much more memory-demanding. The WK floor
     for non-stationary signals is different (and usually much
     lower for an architecture that can detect the drift).
  2. *Longer prediction horizons*: predict A(t+k) for k > 1. As
     k grows, the EMA's implicit memory becomes insufficient
     and real architectural memory would dominate.
  3. *Multi-stimulus*: two or more concurrent fGn streams that
     B has to attend to selectively. Memory + gating load.
  4. *Drop the prediction framing entirely* and go back to
     tasks the substrate is built for (spike-timing-dependent
     learning, structural growth).

The back-out clause from the original entry is partially
triggered: fractional stimulus didn't fall flat, but the
scientifically load-bearing question it was supposed to
illuminate ("does memory help?") came back with a small answer.
Not wasted work — the WK floor machinery stays in the library,
and the diagnosis of "controller EMA is the implicit predictor"
is a real finding. But the direction doesn't automatically
continue.

---

## License and contributions

Licensed **GNU AGPL-3.0-or-later**. See `LICENSE` for the canonical license text and `COPYRIGHT` for the copyright statement, SPDX identifier, and contribution policy.

Ed Hodapp is the sole author and sole copyright holder. **External contributions (pull requests, patches) are not accepted at this time.** Consolidated copyright ownership preserves future licensing flexibility — relicensing, dual-licensing for commercial clients who cannot accept AGPL's network-copyleft, or opening to contributions under a CLA later. Accepting external contributions under AGPL would lock those options down. Pull requests will be declined, not merged.

Commercial users who need to deploy silicritter-based work without AGPL obligations should contact Ed for a commercial-license / consulting engagement.

---

## Acknowledgements and influences

- Pat Shipman, *The Invaders: How Humans and Their Dogs Drove Neanderthals to Extinction* (Harvard University Press, 2015) — directional pull toward social intelligence as the long-term target.
- Anthony Zador, "A critique of pure learning and what artificial neural networks can learn from animal brains" (*Nature Communications*, 2019) — the aligned manifesto for innate-scaffold-plus-lifetime-learning architecture.
- Carver Mead, *Analog VLSI and Neural Systems* (1989) and the subsequent Mahowald / Sarpeshkar / Indiveri / Mead-lineage neuromorphic literature — the analog substrate this project aims to land on eventually.
- Vincent Lextrait, *Software Development or the Art of Abstraction Crafting* — the methodological backbone; abstraction is not factorization.
- Peter Naur, "Programming as Theory Building" (1985) and Alexander Grothendieck's "rising sea" metaphor — the longer lineage of the same argument.
- Song, Miller, Abbott (2000), Izhikevich (2007), Frémaux & Gerstner (2016) — the STDP and three-factor-rule literature underlying `plasticity.py`.
- Aston-Jones & Cohen (2005) — adaptive-gain theory of the noradrenergic system, underlying the `adrenaline` modulator.
- Adrian Thompson (1996), "An evolved circuit, intrinsic in silicon, entwined with physics" — the cautionary prior art for evolvable analog hardware.
