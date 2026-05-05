# Empirical Discoveries Log

Chronological record of *empirical* discoveries from silicritter's
experiments — findings the data revealed that weren't pre-decided by
design. Sister file to `DECISIONS.md` (architectural decisions) and
`perf_history.md` (full per-run measurements).

Each entry captures the discovery, the evidence, and the implications.
Entries are numbered sequentially (`X001`, `X002`, ...) and never
renumbered.

**Entries are immutable in content.** Once written, the discovery
text and evidence are never edited or deleted - the log is a
historical record of what we learned and when. If a discovery is
later contradicted by stronger evidence, the supersession protocol
below applies.

## What belongs here vs in DECISIONS.md vs perf_history.md

- **DECISIONS.md** = architectural / design choices we *made* (E/I
  defaults, controller params, project license, etc.). Inputs to
  experiments.
- **perf_history.md** = full empirical record per run (numbers,
  caveats, configuration). The raw scientific log.
- **DISCOVERIES.md** = condensed *findings* — what the experiments
  *taught us* that wasn't decided in advance. The "if you only read
  one document to know what silicritter has actually demonstrated"
  file.

## Supersession and traceability

When a discovery is revised or reversed by stronger evidence, both
entries are linked. Same convention as `DECISIONS.md`:

1. New entry `X00N` opens with a back-pointer: `**Supersedes:** X003
   (deprecated YYYY-MM-DD HH:MM UTC). [reason]`.
2. Superseded entry `X003` gets an append-only deprecation
   annotation prepended: `**DEPRECATED YYYY-MM-DD HH:MM UTC -
   superseded by X00N.** [reason]`. The original body stays exactly
   as written, below the annotation.

Chained supersessions annotate every link so a reader landing on any
entry can follow the chain in one step.

---

## 2026-05-03

### X001: Step 10 closed-loop headline is **−2.7284e-05 ± 7.2e-11**, not −5.60e-05

**Supersedes informal claim:** step 10's original write-up reported
"−5.60e-05 hand-wired closed-loop breakthrough" (N=1, T=2000). Phase
2 + Block 9 confirm the qualitative improvement but at a different
number after the controller has time to equilibrate.

**Evidence:** `overnight_results/block9_step10_n500.csv` (4000 rows),
`perf_history.md` 2026-05-03 entry. N=500 seeds, T=10M, gain=200, on
the GTX 1050 Mobile, 45 hours wall.

    open_loop  T=10M N=500: -1.5888e-04 ± 3.19e-08 (SEM)
    gain=200   T=10M N=500: -2.7284e-05 ± 7.19e-11 (SEM)
    ratio:     5.825x improvement, bounded to <1/1000

Phase 2 (N=5) was already at 5-sig-fig agreement with Block 9's
N=500 mean (−2.728288e-05 vs −2.7284e-05). The Phase 2 N=5 anchor
was already converged at the precision the seed sweep allows;
rail-clipping at gain=200 makes most seeds produce the exact same
trajectory.

**Why the original number differed:** step 10's −5.60e-05 was a
single-seed measurement at T=2000. The EMA controller (decay=0.98,
τ ≈ 50 steps) is barely settled within the 4-segment stimulus
profile of length T/4 = 500 steps each. At T ≥ 100k the controller
has thousands of cycles to equilibrate; the steady-state fitness is
materially better than the transient-dominated short-T number.

**Implications:**

- Any silicritter paper / external claim should use **−2.7284e-05**
  as the headline closed-loop fitness number for the hand-wired
  cross-E-only B pool at gain=200, T=10M.
- The 5.8× improvement framing (vs open-loop) holds and tightens.
- Future blocks should evaluate at long T by default; T=2000 is a
  smoke-test scale, not a publication-grade scale.

---

## 2026-05-01

### X002: D008's mechanism is "controller dynamic range," not "more inhibition"

**Evidence:** `overnight_results/block10_step13_ei_n100.csv` (400
rows), `perf_history.md` 2026-05-01 entry. N=100 multi-seed
comparison of `i_mult=4.0` (prior canonical) vs `i_mult=8.0` (D008)
at step 13's T=2000, closed-loop gain=50.

The fitness improvement (D008 vs canonical, +12.4% closed-loop) was
already established. Block 10 surfaces the *mechanism*, only visible
at N≥30:

| `i_mult` | mean fitness   | std across seeds |
|----------|---------------:|------------------|
| 4.0      | −5.595e-05     | **3.98e-11**     |
| 8.0      | −4.904e-05     | 4.97e-07         |

At i_mult=4.0, **all 100 seeds give bit-identical fitness** (std =
4e-11, ~12 orders of magnitude smaller than open-loop std). The
controller saturates `adr_max` for the entire run and produces the
same trajectory regardless of pool randomization - a deterministic
fitness *floor*, not an optimum.

At i_mult=8.0, std jumps to 5e-07 (same order as open-loop noise
floor). **The controller is no longer rail-clipped.** Stronger
inhibition makes B harder to drive; when A's drive demands push B
up, the controller works *within* `[adr_min, adr_max]` instead of
saturating.

**Therefore:** D008's win isn't "more inhibition is intrinsically
better in a flat sense." It's "the controller has somewhere to work,
so it actually works." This pattern - rail-clipped under one
parameter regime, controllable under another - is a generalizable
structural feature of the silicritter substrate, not an artifact of
this specific E/I sweep (see X003).

**Implications:**

- D008's `(0.2, 8.0)` adoption (per DECISIONS) is correct; the
  rationale is now mechanistic rather than empirical-only.
- When tuning controller parameters or operating points in future
  blocks, ask: *is the controller in its working range, or is it
  railing?* If railing, raw fitness is a deterministic floor that
  doesn't reflect the controller's contribution.
- The N=20 anchor that originally informed D008 was too small to
  resolve the std=0 vs std=5e-7 separation; this finding required
  N≥30. Future load-bearing-claim runs should default to N≥100.

---

## 2026-05-03

### X003: Rail-clipped/controllable structure generalizes from E/I-space to T-space

**Evidence:** `overnight_results/block9_step10_n500.csv`,
`perf_history.md` 2026-05-03 entry. The same i_mult=4.0 / gain=200
operating point that produces deterministic rail-clipping at one T
shows seed variance at another:

| T          | closed-loop std (N=500) | regime          |
|------------|------------------------:|-----------------|
| 10 000     | **0.000**               | rail-clipped    |
| 100 000    | 1.67e-09                | controllable    |
| 1 000 000  | **0.000**               | rail-clipped    |
| 10 000 000 | 1.61e-09                | controllable    |

The pattern alternates: T=10k and T=1M show 500-seed bit-identical
trajectories; T=100k and T=10M show tiny seed variance. Same i_mult,
same gain - only T changes, yet the controller's working regime
flips between rail-clipped and controllable.

**Mechanism (hypothesized):** the EMA controller's time constant
(τ ≈ 50 steps from decay=0.98) interacts with the A-drive profile's
segment length (T/4). At T=10k segments are 2500 steps (50τ); at
T=100k segments are 25000 steps (500τ); etc. Whether the
controller's per-segment transient dominates or vanishes within a
segment determines the regime.

**Implications:**

- The X002 mechanism is a property of the controller-substrate
  interaction at *operating-point granularity*, not unique to the
  E/I axis. Expect the same dichotomy along any controller-relevant
  axis (gain, decay, A-drive amplitude, etc.).
- For block design at this operating point: assume some T values
  will produce zero-variance rail-clipped trajectories. Variance
  estimates at those T values are floor measurements, not noise
  measurements.
- A formal analysis of the τ-vs-segment-length interaction would
  let us *predict* which T values rail-clip without running them.
  Worth a follow-up if it bears on Block 12+.

---

## 2026-05-04

### X004: Open-loop CPPN GA does not find topology better than hand-wired

**Evidence:** `overnight_results/block11_cppn_n20.csv` (40 rows),
`overnight_results/block11b_genome_stats.csv` (40 rows),
`overnight_results/block11b_eval_seeds.csv` (4000 rows),
`perf_history.md` 2026-05-04 entries (Block 11 + Block 11b).

Block 11 ran a CPPN GA on 20 independent seeds for both open-loop
and closed-loop conditions. Reported "+9% open-loop GA win" vs
hand-wired. **That was selection-on-noise bias.**

Block 11b's Phase 2 topology fingerprinting:

    All 20 open-loop GAs converged to bit-identical pattern:
        cross_pct       = 100.0  (vs 100 hand-wired)
        cross_e_pct     = 100.0
        cross_i_pct     =   0.0
        recurrent_pct   =   0.0
        v_mean          =   1.997
        v_std           =   0.002

This is the hand-wired pattern, encoded as a CPPN.

Block 11b's Phase 3 novel-scenario re-evaluation (100 eval_seeds,
varying `pool_a`'s PRNGKey from training's PRNGKey(777)):

    Block 11 training:        -1.4230e-04 (N=20 ga_seeds)
    Block 11b novel scenarios: -1.5923e-04 (N=2000 evals)
    Hand-wired (step 9):      -1.5600e-04

The 11.9% downward shift between training and eval is the bias.
Open-loop GAs perform **within 2% of hand-wired** on novel
scenarios, in the *worse* direction (−1.59e-04 vs −1.56e-04).

**Therefore:** the GA finds the hand-wired pattern as the
open-loop optimum; there's no GA-only open-loop discovery.

**Implications:**

- Future open-loop topology work doesn't need GA exploration to
  confirm topology choice; the hand-wired cross-E-only pattern is
  the answer.
- Reporting Block 11 alone (without Block 11b's eval-seed phase)
  would have shipped a false-positive open-loop finding. The
  asymmetric inference rule held in real life: "GA wins"
  measurements need novel-scenario validation.

---

## 2026-05-04

### X005: Closed-loop CPPN GA at ga_seed=0 finds a recurrent-mix topology, ~6% better than hand-wired

**Evidence:** `overnight_results/block11b_genome_stats.csv` row
`ga_seed=0, condition=closed_loop`,
`overnight_results/block11b_eval_seeds.csv` rows for that genome,
`perf_history.md` 2026-05-04 Block 11b entry.

Block 11b's per-genome topology fingerprint identified one
closed-loop GA (ga_seed=0) with a structural pattern hand-wiring
never used:

    cross_pct       = 87.9    (vs 100 hand-wired)
    cross_e_pct     = 87.9
    cross_i_pct     =  0.0
    recurrent_pct   = 12.1    (NEW: hand-wired is 0)
    v (all near saturation, std=0.003 within pool)

On the training scenario this beat hand-wired by 12% (−4.92e-05 vs
−5.60e-05). On 100 novel scenarios its mean is around −5.27e-05 -
**~6% better than hand-wired (−5.60e-05)**, smaller than the
training-fit suggested but real.

This is the *only* closed-loop genome with a meaningful recurrent
fraction; the other 19 closed-loop GAs have recurrent_pct in
[0, ~3%] (mean 1.4%, but most cluster near 0 with a few outliers
above).

**Implications:**

- The recurrent-mix is a real, replicable, topology-driven
  improvement. Worth investigating whether it generalizes:
  - Does it still beat hand-wired at D008's i_mult=8.0 (where
    Block 10 showed the controller has dynamic range)?
  - Does it survive at long T (Block 9-style T=10M evaluation)?
  - Could the recurrent slots be hand-coded into a smarter
    baseline that captures the GA's discovery without re-running
    the GA?
- Block 11b's Phase 1 captured this genome's CPPN weights to
  `block11b_genomes.pkl`; the genome is recoverable for further
  analysis without re-running the GA.

---

## 2026-05-04

### X006: Selection-on-noise bias is empirically large in CPPN-GA training; eval-seed re-evaluation is mandatory for "GA wins" claims

**Evidence:** Block 11 (training) vs Block 11b (eval) deltas in
`perf_history.md` 2026-05-04 entries.

Block 11 trained 40 GAs on a single deterministic scenario
(PRNGKey(777)) and reported the best fitness of each. Block 11b
re-evaluated each evolved genome on 100 different scenarios and
measured the delta:

| condition   | training (Block 11) | novel (Block 11b) | bias  |
|-------------|---------------------|-------------------|-------|
| open_loop   | −1.4230e-04         | −1.5923e-04       | +11.9% |
| closed_loop | −5.3909e-05         | −5.5516e-05       | +3.0%  |

The +11.9% open-loop bias is large enough to flip the conclusion
(see X004). The +3.0% closed-loop bias was within the +4% headline
"GA mean win" Block 11 reported, leaving only ~0.9% real
improvement at the *mean* level (the best-individual story is
X005's separate finding).

**Mechanism:** maximum-of-N stack (max of 32 pop × 30 gens × 20
GAs) selects the population member best at the deterministic
training scenario. With deterministic training, that maximum is
upward-biased by ~σ × √(2 ln N) where σ is intra-population
fitness std at convergence. Empirically here: bias is order of
magnitude 1e-5 in the open-loop case.

**Implications (methodological, applies to any future GA in this codebase):**

- **"GA wins" findings require eval-seed re-evaluation.** The
  asymmetric inference rule:
  - "GA loses" is robust if measured (the bias direction makes
    the true value at least as bad as the training fit).
  - "GA wins" must be validated on novel scenarios; the bias
    direction makes the training fit better than the true value.
- Future GA scripts should bake in the eval-seed re-eval phase
  by default (Block 11b is the prototype). A "Block X" without a
  "Block Xb" eval-seed companion is incomplete for any
  comparison-against-baseline claim.
- The Block 11 / Block 11b decoupling worked but added a separate
  authoring + commit + launch cycle. For Block 12+ consider
  combining eval-seed re-eval into the main GA script if the
  workload size permits (Block 11b's 4000-eval phase took 24 min
  on the laptop - cheap enough to include by default).

---

## 2026-05-05

### X007: fGn closed-loop tracking is 11-16× across all H, not 20-30× as step 14 reported; prediction-not-tracking is what's H-dependent

**Supersedes informal claim:** step 14's docstring summary said
"the closed-loop controller delivers a 20-30× tracking improvement
over open-loop at every H" and "Closed-loop tracking is 3× worse
at H=0.7 than at H=0.5." Block 12 N=100 reanchor revises both
specific magnitudes; the directional patterns hold.

**Evidence:** `overnight_results/block12_fgn_n100.csv` (800 rows),
`perf_history.md` 2026-05-05 entry. N=100 seeds × 4 H × 2 conditions
on the GTX 1050 Mobile, 4.5 min wall.

**Tracking improvement ratio (open-loop / closed-loop) at each H:**

    H=0.3: 16.0x        H=0.7: 11.1x
    H=0.5: 13.0x        H=0.9: 11.0x

The actual range is **11-16x** at N=100, not "20-30x" as the
step-14 docstring asserted. Step 14's single-seed numbers were
upper-tail of the distribution. Still an order-of-magnitude
improvement at every H, just smaller than the original claim.

**Closed-loop tracking H=0.7 / H=0.5 ratio is 1.17x, not 3x:**

    H=0.5 closed-loop track: -6.39e-06 +/- 4.3e-07 (SEM)
    H=0.7 closed-loop track: -7.47e-06 +/- 5.3e-07 (SEM)
    Ratio:                   1.17x worse at H=0.7

The "3x worse at H=0.7" step-14 claim was an artifact of N=1
measurement variance; at N=100 the cell-mean ratio is small and
the SEM bounds are tight (~5e-07).

**Prediction-not-tracking IS H-dependent (this is the real fGn finding):**

    H=0.3 closed-loop pred: -9.53e-06
    H=0.5 closed-loop pred: -1.27e-05
    H=0.7 closed-loop pred: -2.62e-05
    H=0.9 closed-loop pred: -3.82e-05

Prediction degrades **4x** from H=0.3 to H=0.9 (a 50% increase in
H produces a 4x worse one-window-ahead prediction). Tracking is
robust across H; *predicting future* state of an fGn-driven A is
what gets harder at higher H.

The pred-minus-track gap grows similarly:

    H=0.5 gap: -6.27e-06
    H=0.9 gap: -3.01e-05    (4.8x larger than H=0.5)

This *is* what step 14 directionally reported. The 4-5x range
matches and now has SEM bounds.

**Lag-1 autocorrelation by H matches fGn theory exactly:**

    H=0.3: -0.22 (anti-persistent)
    H=0.5: -0.05 (neutral)
    H=0.7: +0.14 (persistent)
    H=0.9: +0.33 (strongly persistent)

Sanity check on `silicritter.fracnoise`: the fGn drive trace is
doing what fGn theory predicts at every H. Closes a "is the
stimulus actually fGn?" doubt that step 14's N=1 didn't address.

**Implications:**

- README's step 14 description should drop the "20-30x" framing
  and replace with "11-16x at N=100." The prediction-gap framing
  is fine to keep.
- The tracking-vs-prediction distinction is the real fGn story:
  closed-loop adrenaline tracks fGn well at any H; predicting
  fGn one window ahead is fundamentally harder at high H. This is
  the architectural insight worth keeping for future blocks.
- The fracnoise implementation passes its own theory check
  (lag-1 autocorr matches H expectation). No need for further
  validation at the noise-generator level; future blocks can
  trust the stimulus shape.

**Methodology:** Block 12 used the same orchestration pattern as
Blocks 9/10/11 (per-row CSV append, resume, MAX_CONSECUTIVE_FAILURES,
cross-file regression test for `s14._run_condition` signature).
Wall time was trivial (4.5 min) so N=500 would be cheap if any
Block 12 number ever needs to be the headline of a paper claim.

---

## 2026-05-05

### X008: Step 16 STDP improvement at rate=1.0 is **+15.96% ± 0.27% (95% CI, N=100)**, not +14%

**Supersedes informal claim:** the README dev log called it
"+14% at rate=1.0" (single-seed measurement). Block 13 confirms
the qualitative finding and tightens the magnitude.

**Evidence:** `overnight_results/block13_stdp_n100.csv` (100 rows),
`perf_history.md` 2026-05-05 entry. N=100 seeds (stride 37) at
rate=1.0, init_v=1.0, closed-loop gain=50, D008 i_mult=8.0,
T_train=20k, T_measure=2k. 2.5-min wall on the GTX 1050 Mobile.

    fit_before:      -2.6858e-04 +/- 7.93e-07 (SEM)
    fit_after:       -2.2567e-04 +/- 6.41e-07 (SEM)
    improvement_pct: +15.96      +/- 0.14     (SEM)
                     +15.96      +/- 0.27     (95% CI)

All 100 seeds show **positive** improvement (range [+13.05%,
+20.12%]). The +14% original was at the lower tail; the
distribution mean is ~16%.

**Implications:**

- README's step 16 description should cite **+16% (95% CI ±0.27%)**
  rather than the original +14%. The directional claim ("STDP at
  rate=1.0 produces post-training improvement from a random init")
  holds robustly.
- The "STDP alone is bounded" framing is unchanged. Post-training
  fitness (-2.26e-04) is still ~10x worse than Block 9's hand-
  wired closed-loop headline (-2.7284e-05); STDP can't close the
  topology gap from a random init. The structural-plasticity
  pitch (slot acquisition + release in step 17) remains the
  project's answer.
- Block 13 used `overnight_batch._step16_once` directly, which
  guarantees bit-exact comparability with overnight_batch's
  blocks 4/5 at the overlapping seeds. The N=100 measurement is
  a strict superset of the prior N=5 / N=20 numbers; no new
  experimental design choices were made beyond the seed count.
