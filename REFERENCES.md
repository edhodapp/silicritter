# References

External literature relevant to silicritter's design decisions and
ongoing thinking. Maintained as a curated reading list, not an
exhaustive bibliography. Each entry has a short "why it matters
here" note.

For per-experiment references and inline citations, see the
docstring of the relevant `experiments/stepNN_*.py` script.

---

## Reservoir computing

Mathematically solid framework where a fixed random recurrent
network projects input into a high-dimensional dynamical state, and
only a linear readout is trained. Demonstrably works as a machine
learning technique on temporal classification; suggestive (but not
proven) as a model of cortical microcircuit computation.

**silicritter relevance:** the paired-agent architecture isn't
strict RC, but it's RC-adjacent — fixed substrate (B's slot pool
topology) + closed-loop readout (the adrenaline controller learning
to track A). Worth knowing the literature when framing the project
to a computational-neuroscience audience.

- **Jaeger, H.** (2001). *The "echo state" approach to analysing and
  training recurrent neural networks.* GMD Technical Report 148,
  German National Research Center for Information Technology.
  The original Echo State Network (ESN) paper. Continuous-valued
  reservoirs.
- **Maass, W., Natschläger, T., & Markram, H.** (2002).
  *Real-Time Computing Without Stable States: A New Framework for
  Neural Computation Based on Perturbations.* Neural Computation
  14(11), 2531–2560. The Liquid State Machine (LSM) paper.
  Spiking-neuron version of the same idea, more directly relevant
  to silicritter's substrate.
- **Lukoševičius, M., & Jaeger, H.** (2009). *Reservoir computing
  approaches to recurrent neural network training.* Computer
  Science Review 3(3), 127–149. The standard review article;
  good entry point covering both ESN and LSM lineages.

**Status caveat:** RC is real and useful as ML; the *brain does
RC* claim is one of several plausible frames, not consensus. Use
as inspiration, not constraint.

---

## Thousand Brains / Numenta

Jeff Hawkins' theory that every cortical column independently
models the world via grid-cell-like reference frames; intelligence
emerges from ~150,000 mini-brain "votes." High-ambition, much
weaker scientific footing than RC.

- **Hawkins, J.** (2021). *A Thousand Brains: A New Theory of
  Intelligence.* Basic Books. The popular-science book.
- **Hawkins, J., et al.** Numenta papers on Hierarchical Temporal
  Memory (HTM) and the Thousand Brains theory at numenta.com/papers.

**Status caveat:** the *grid-cells-everywhere-in-cortex* claim has
some experimental support but is far from mainstream consensus.
The *every column models everything* claim conflicts with known
cortical specialization (visual columns process visual features,
etc.). HTM has produced ~zero notable benchmark wins despite ~20
years of development. Treat as inspiration, not as a design
constraint.

**silicritter relevance:** low. Don't design around it.

---

## Classical / mainstream computational neuroscience for silicritter's domain

These researchers' work is more directly relevant to silicritter's
spiking-network + closed-loop-controller + STDP architecture than
the popular-science framings above.

### Wolfgang Maass

Computational power of spiking networks; LSM lineage; how networks
of leaky-integrate-and-fire neurons can implement arbitrary
real-time computation. Foundational for any project that takes
spiking timing as fundamental.

- **Maass, W.** (1997). *Networks of spiking neurons: The third
  generation of neural network models.* Neural Networks 10(9),
  1659–1671. The "spiking nets are a strict superset of
  feedforward nets" argument.
- **Maass, Natschläger, & Markram (2002)** — see Reservoir Computing.

### Eve Marder

Lab studying circuit homeostasis and parameter degeneracy: the
*same circuit behavior* can arise from many different parameter
combinations, and biological circuits actively maintain
behavioral output across parameter drift. Directly relevant to
silicritter's X002 finding (controller dynamic range matters more
than specific i_mult/gain values; the mechanism is "is the
controller in working range," not "is this exact parameter value
right").

- **Prinz, A. A., Bucher, D., & Marder, E.** (2004). *Similar
  network activity from disparate circuit parameters.* Nature
  Neuroscience 7(12), 1345–1352. The classic "many parameter sets,
  one behavior" paper. STG (stomatogastric ganglion) work.
- **Marder, E., & Goaillard, J.-M.** (2006). *Variability,
  compensation and homeostasis in neuron and network function.*
  Nature Reviews Neuroscience 7, 563–574. Review of the
  parameter-degeneracy / homeostatic-tuning literature.

**silicritter relevance:** high. The X002 mechanism (controller
saturates at one operating point and works in range at another)
is a special case of the "circuit behavior is parameter-set
dependent in a non-obvious way" pattern Marder's lab has been
documenting for decades.

### Henry Markram

Blue Brain Project; cortical microcircuit detail; original co-author
of the LSM framework.

- **Markram, H.** (2006). *The Blue Brain Project.* Nature Reviews
  Neuroscience 7, 153–160. The detailed-cortical-simulation arc.
- **Markram, H., Lübke, J., Frotscher, M., & Sakmann, B.** (1997).
  *Regulation of synaptic efficacy by coincidence of postsynaptic
  APs and EPSPs.* Science 275, 213–215. The first STDP paper —
  directly relevant to silicritter's plasticity model.

**silicritter relevance:** the 1997 STDP paper is foundational for
silicritter's `silicritter.plasticity` module's three-factor STDP.
The Blue Brain work is more aspirational / cautionary tale.

### Larry Abbott

Theoretical neuroscience standard texts; controller and rate-
dynamics theory; co-author with Peter Dayan of the most-used
graduate textbook in the field.

- **Dayan, P., & Abbott, L. F.** (2001). *Theoretical
  Neuroscience: Computational and Mathematical Modeling of Neural
  Systems.* MIT Press. The canonical graduate textbook. Chapters
  on rate models, recurrent networks, plasticity, and stochastic
  dynamics are directly relevant.
- **Sussillo, D., & Abbott, L. F.** (2009). *Generating coherent
  patterns of activity from chaotic neural networks.* Neuron 63(4),
  544–557. The FORCE learning paper — relevant if silicritter ever
  explores reservoir-style supervised readout.

**silicritter relevance:** Dayan & Abbott is the right reference
for any rate-vs-spike, mean-field, or LIF-population analytic
question. FORCE is a useful reference if Block N+ ever tries
RC-style explicit readout training.

---

## Related: Aston-Jones & Cohen on adaptive gain (already cited in step 10)

The biological inspiration for silicritter's closed-loop adrenaline
controller. Locus coeruleus delivers a global gain signal modulating
cortical neurons' effective tau_m; controller architecture in step
10 mirrors this.

- **Aston-Jones, G., & Cohen, J. D.** (2005). *An integrative
  theory of locus coeruleus-norepinephrine function: Adaptive gain
  and optimal performance.* Annual Review of Neuroscience 28,
  403–450.

Cited in `experiments/step10_closedloop_adrenaline.py` and in
`src/silicritter/closedloop.py` docstrings.

---

## Honest reading-strategy notes

- **Cross-check YouTube neuroscience against papers.** The medium
  rewards confident framing of unsettled science; even good
  explainers flatten the controversy. Use Scholar / direct paper
  reading to calibrate any claim that sounds confident.
- **Popular-science books vs peer-reviewed papers.** Hawkins' books
  outpace his peer-reviewed papers; Marder's reviews are tightly
  tied to her group's experimental data. Calibrate accordingly.
- **Mathematical clarity ≠ biological accuracy.** RC is
  mathematically clean and easy to test, which makes it a
  *plausible* brain model — not a proven one. Same caution applies
  to any computational framework imported into neuroscience.
- **For silicritter specifically:** the project's value is in
  building a concrete substrate that exposes mechanisms (X002
  controller-headroom; X007 fGn tracking-vs-prediction asymmetry).
  Theory should *inform* design, not over-constrain it. Add
  references to this file as they prove useful, not as ceremony.
