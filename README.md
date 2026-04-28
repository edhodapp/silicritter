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

## Present status (Phase 1, steps 1–17 in progress; revalidation paused mid-flight)

What exists today is an idealised JAX simulation of a paired-agent plastic spiking substrate with E/I balance, a closed-loop adrenaline gain controller, CPPN indirect encoding, and a fractional-Gaussian-noise stimulus generator with a Wiener-Kolmogorov-floor cross-check. No chip, no circuit-level validation yet.

### Modules (`src/silicritter/`)

- **`lif.py`** — Leaky integrate-and-fire forward simulation. `init_state`, `integrate_and_spike`, `step` (dense-weight variant), `simulate`.
- **`slotpool.py`** — Slot-pool synapse representation. `SlotPool` NamedTuple with per-slot `pre_ids`, `v`, `plasticity_rate`, `active`, and `release_counter`. `synaptic_current` supports an optional `pre_is_inhibitory` mask with `i_weight_multiplier` (default 8.0 per D008). `assign_ei_identity` returns a deterministic E/I mask.
- **`plasticity.py`** — Three-factor STDP on the slot pool with pre-decayed eligibility traces (Song/Miller/Abbott). Modulators: `valence` (scalar, gates STDP sign/magnitude) and `adrenaline` (scalar, multiple gain mechanisms registered via `GAIN_MODULATORS`, with `tau_m_scale` as the current winner). `PlasticNetState` groups LIF + pool + traces.
- **`structural.py`** — Slot-release primitives. `StructuralParams`, `apply_release`. Acquisition from free pool not yet implemented.
- **`ga.py`** — Direct-encoding GA primitives. `Genome`, tournament selection, uniform crossover, bound-preserving mutation. Used as a reference; the CPPN indirect encoding is the current scaling path.
- **`cppn.py`** — CPPN (Compositional Pattern-Producing Network) indirect encoding. `CPPNGenome` (two weight matrices), `decode_cppn_to_pool`, tournament/crossover/mutation in CPPN space. ~480× search-space reduction vs. direct encoding at N=256, K=32.
- **`paired.py`** — Paired-agent substrate. `PairedState` wraps two `PlasticNetState`s with pre_ids indexing into a combined `[own, partner]` raster of length 2·n_neurons. `step_paired` runs both agents in two phases (LIF forward, then STDP update); `simulate_paired` scans the whole sim. E/I substrate is opt-in via `*_is_inhibitory` arguments.
- **`closedloop.py`** — Leaky-integrator adrenaline controller. `ControllerState` (rate EMAs + current adrenaline), `ControllerParams` (decay, baseline, gain, clip range). `step_closedloop` / `simulate_closedloop` broadcast B's adrenaline from an error signal `rate_a_ema − rate_b_ema` via `tau_m_scale`. Aston-Jones & Cohen 2005 adaptive-gain cast as a control loop.
- **`fracnoise.py`** — Fractional Gaussian noise stimulus generator. `fgn_autocov` (unit-variance autocov at arbitrary Hurst H), `fgn_davies_harte` (O(N log N) exact FFT synthesis), `fgn_drive_trace` (shaped drive trace for use as `i_ext_a`).
- **`wk.py`** — Wiener-Kolmogorov prediction-floor computations. `durbin_levinson` (O(n²) recursion on any autocov sequence, with divide-by-zero and float32 clamp guards), `windowed_fgn_autocov` (non-overlapping by default, arbitrary stride for future overlapping-window experiments), `wk_floor_windowed_fgn` (convenience wrapper).

### Experiments (`experiments/`)

Each step is a self-contained runnable. See `perf_history.md` for measured numbers on the reference machine (Huawei MateBook X Pro 2018, i7-8550U + NVIDIA MX150, 2 GB VRAM). Every significant experiment has a dedicated perf_history entry with caveats.

- **`step02_throughput.py`** — LIF dense-weight throughput baseline.
- **`step03_slotpool_throughput.py`** — Slot-pool forward-sim baseline (~5× speedup over dense at K=64).
- **`step04_plastic_throughput.py`** — Slot pool + STDP + modulators.
- **`step05_ga_target_rate.py`** — Direct-encoding GA evolving scaffolds to track a target rate; exposed the multiplicative-gain floor problem that motivated the `tau_m_scale` gain mechanism.
- **`step07_*.py` / `step07e_paired_cppn_n256.py`** — Paired-agent signal-following; direct and CPPN encodings at N=256.
- **`step09_handwired_n256_ei.py`** — E/I substrate validation at canonical values (D007).
- **`step10_closedloop_adrenaline.py`** — Closed-loop adrenaline breaks the static 30 Hz firing-rate ceiling; multi-seed confirms result is seed-independent.
- **`step11_cppn_closedloop.py`** — CPPN GA + E/I + closed-loop. Open-loop CPPN beats hand-wired cross-E-only by 10%; closed-loop CPPN discovers an 87.9% cross / 12.1% recurrent topology that beats hand-wired closed-loop by 12%.
- **`step12_tonic_sweep.py`** — Peeled tonic drive down from 16 mV. Tonic is load-bearing: closed-loop extends viable range to ~8 mV, then hits a physics cliff (sub-threshold neurons can't be rescued by shortening τ_m).
- **`step13_ei_perturbation.py`** — 5×5 grid of (inhibitory_fraction, i_weight_multiplier) around D007. Multi-seed comparison produced D008, tightening to `(0.2, 8.0)`.
- **`step14_fgn_stimulus.py`** — fGn-driven A at H ∈ {0.3, 0.5, 0.7, 0.9}. Architecture fitness is H-dependent but stumbles through variance regimes rather than exploiting memory structure.
- **`step15_wk_floor_comparison.py`** — Compared step 14's observed prediction MSE to the Wiener-Kolmogorov theoretical optimum. Closed-loop is 1.9–2.4× above the floor at every H (strikingly close to optimal for an architecture with no explicit memory).
- **`step16_stdp_learning.py`** — First experiment with `plasticity_rate > 0` everywhere. Sweep over plasticity rate from a random initial B pool with closed-loop adrenaline at gain=50. Lifetime STDP does measurable work but cannot reach GA-discovered fitness from a random init — the useful-slot fraction is frozen at initialization because STDP adjusts `v` but not `pre_ids`.
- **`step17_structural_growth.py`** — Slot acquisition + release combined with STDP — the "exuberance and pruning" developmental loop. Acquisition modes (`off`, `periodic`, `valence_gated`, `valence_inverted`) and release thresholds explored. Long-duration revalidation runs gated on the Phase 1 raster-retention fix landed 2026-04-26.

### Tests (`tests/`)

127 behavioural and unit tests. `pytest --cov --cov-branch` reports **100% branch coverage** across all ten library modules.

### Quality gates

Every commit that changes functional code runs through:

- `flake8 --max-complexity=5 --max-line-length=80`
- `mypy --strict` (all type annotations checked)
- `pylint` with the Google Python Style Guide `pylintrc`
- `pytest --cov --cov-branch` (100% branch coverage target)

Plus independent code review by an isolated Claude subagent (clean context, no project knowledge) and advisory review by Gemini, both running from `~/tools/code-review/`.

### What is deliberately not yet implemented

- **Slot acquisition from the free pool** — slot *release* exists in `structural.py`; genuine rebinding of released slots onto new pre-neurons is the missing half of "exuberance and pruning."
- **Inhibitory-specific STDP rules** — Vogels 2011 anti-Hebbian rule on I synapses. Deferred; may interact with D008's stronger `i_mult`.
- **Additional chemical signals** — cortisol, oxytocin, vasopressin, BDNF, NO. Added one at a time as specific experiments need them.
- **Noisy behavioural sim (Phase 2)** — process variation, transistor mismatch, capacitor leakage, quantization. Starts when Phase 1 architecture is declared frozen.
- **Circuit-level validation (Phase 3)** and **layout / tape-out (Phase 4)**.
- **Non-stationary stimulus tasks** — the step 15 finding suggests memory-capable architectures would need genuinely memory-demanding tasks (non-stationary signals, longer prediction horizons, multi-stream attention) to show advantage; stationary fGn already has very little memory headroom available.

---

## Running locally

Developed on Python 3.12 with a virtualenv in `.venv/`:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Dependencies pull in `jax[cuda12]` with bundled NVIDIA CUDA 12 runtime libraries. An NVIDIA GPU (CC 6.0+, ~1 GB free VRAM) is required for the GPU-backed experiments; everything falls back to CPU if no GPU is present but will be much slower.

Run an experiment (all are self-contained scripts; browse `experiments/` for the full set):

```bash
.venv/bin/python experiments/step10_closedloop_adrenaline.py --n-seeds=5
.venv/bin/python experiments/step11_cppn_closedloop.py
.venv/bin/python experiments/step13_ei_perturbation.py
.venv/bin/python experiments/step15_wk_floor_comparison.py
```

Run tests:

```bash
.venv/bin/pytest tests/
```

---

## Project documentation

- **`DECISIONS.md`** — architectural decision log. Immutable entries D001–D008 with supersession annotations on deprecated entries (D007 → D008 is the first supersession). Read this for the history of why the project looks the way it does.
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

### 2026-04-23 — Step 16: lifetime STDP does real work, but is bounded by topology

Pivoted back to the substrate after shelving the
fractional-stimulus direction. First silicritter experiment with
`plasticity_rate > 0` everywhere — every run from step 9 onward
had plasticity mechanically wired but doing nothing.

Random initial B pool, valence = shaped reward gated on rate
tracking error, closed-loop adrenaline on at gain=50, 20k training
steps. Sweep over `plasticity_rate` (0.01 / 0.1 / 0.3 / 1.0):

| rate  | fitness Δ | interpretation           |
|-------|-----------|---------------------------|
| 0.01  | noise     | STDP can't find the signal |
| 0.1   | ~flat     | dithering regime          |
| 0.3   | +1.1e−5   | learning emerges          |
| 1.0   | +3.7e−5   | ~14% improvement          |

**Lifetime STDP works.** Default STDP is LTD-dominant; the
learning mechanism is preferential weakening of slots whose pre
fires but post doesn't (B→A-I cross-inhibitory targets,
disadvantageous recurrent slots). Valence-gating prevents the
runaway-to-zero failure mode — weakening is targeted at bad-for-
tracking slots, not uniform.

**But STDP alone is bounded.** Post-training fitness at rate=1.0
is ~−2.2e−4, still ~4× worse than step 10's hand-wired
cross-E-only (−5.6e−5). The reason is structural: STDP adjusts
`v`, not `pre_ids`. With uniform-random initialization, ~50% of
slots sit on B→B recurrent targets and ~10% on B→A-I
inhibitory targets. STDP can drive those weights to zero but
cannot release the slot or reassign it to a new pre-neuron.
The useful-slot fraction is frozen at initialization.

**This validates the two-lever pitch.** The project has always
argued that evolution (outer loop, GA) and lifetime learning
(inner loop, STDP) complement each other. Step 16 shows both
that the inner loop *does work* and that it *cannot reach
GA-discovered fitness from a random init*. The missing lever is
structural plasticity — the release-and-reacquire dynamic that
would let topology evolve under pressure. Slot *release* lives in
`src/silicritter/structural.py`; *acquisition* (free-pool rebind
to a new random pre-neuron) does not.

**Step 17 is obvious:** implement slot acquisition. Combined with
existing release and LTD-driven weight collapse, this closes the
"exuberance + sculpting" developmental loop the project has been
pitching since the originating session.

### 2026-04-24 → 2026-04-26 — Revalidation paused; laptop migration; raster fix

Block 7 of an N=20 revalidation sweep collapsed step 10's
headline N=5 closed-loop result to a null. Audit found that most
prior claims inherit from a step-10 N=1 measurement; pausing the
revalidation pending a Phase 1 fix and a larger N=100/500 batch.

In parallel, migrated development from the Huawei MateBook X Pro
(MX150, 2 GB VRAM) to an ASUS VivoBook X580GD (GTX 1050 Mobile,
4 GB VRAM, ~1.6× compute / ~2.3× bandwidth). Phase 0 toolchain
validation confirmed bit-exact reproduction of the step 10 N=5
result across the hardware swap, on the same JAX 0.10.0 stack.
Workload remains launch-overhead-bound (~15% GPU util on a
scan-body-dispatch profile) — bottleneck is XLA dispatch, not raw
compute.

**Phase 1 fix (2026-04-26):** Both `step16` and `step17`
`_training_scan` returned per-step `(T, N_NEURONS)` spike rasters
that callers either discarded entirely (`step17`) or used only
for a global mean firing rate (`step16`). At T=10M the rasters
are ~5 GB each and exceed the 4 GB VRAM ceiling, gating any
long-duration revalidation run. Dropped the rasters; `step16`
now returns an already-computed scalar `rate` trace (free —
computed for the EMA controller). `tests/test_training_scan_memory.py`
pins the contract: no scan output may be a per-step raster.
Memory at T=10M: 5.0 GB → 80 MB (`step17`), 5.0 GB → 120 MB
(`step16`).

**Resumes from:** N=100 / N=500 revalidation batches with the
raster-retention fix in place, starting with step 10
hand-wired closed-loop as the root of the risk graph.

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
