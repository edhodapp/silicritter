# References

Curated reading list of mainstream computational-neuroscience
researchers whose work is directly relevant to silicritter's
spiking-network + closed-loop-controller + STDP architecture.
Recommended over popular-science framings (Reservoir Computing as
brain model, Hawkins' Thousand Brains) — both are real positions
but neither is settled science; the researchers below have decades
of peer-reviewed work tied to actual experimental data.

For per-experiment references and inline citations, see the
docstring of the relevant `experiments/stepNN_*.py` script.

---

## Wolfgang Maass

Computational power of spiking networks; how networks of leaky-
integrate-and-fire neurons can implement arbitrary real-time
computation. Foundational for any project that takes spiking timing
as fundamental.

- **Maass, W.** (1997). *Networks of spiking neurons: The third
  generation of neural network models.* Neural Networks 10(9),
  1659–1671. The "spiking nets are a strict superset of feedforward
  nets" argument.
- **Maass, W., Natschläger, T., & Markram, H.** (2002). *Real-Time
  Computing Without Stable States: A New Framework for Neural
  Computation Based on Perturbations.* Neural Computation 14(11),
  2531–2560. The Liquid State Machine (LSM) paper. The
  spiking-neuron version of reservoir computing — directly relevant
  to silicritter's substrate even though silicritter isn't strict
  RC.

## Eve Marder

Lab studying circuit homeostasis and parameter degeneracy: the
*same circuit behavior* can arise from many different parameter
combinations, and biological circuits actively maintain behavioral
output across parameter drift. **Directly relevant to silicritter's
X002 finding** — the "controller dynamic range matters more than
specific i_mult/gain values; the mechanism is 'is the controller in
working range,' not 'is this exact parameter value right'" pattern
is a special case of decades of Marder-lab observations.

- **Prinz, A. A., Bucher, D., & Marder, E.** (2004). *Similar
  network activity from disparate circuit parameters.* Nature
  Neuroscience 7(12), 1345–1352. The classic "many parameter sets,
  one behavior" paper. Stomatogastric ganglion (STG) work.
- **Marder, E., & Goaillard, J.-M.** (2006). *Variability,
  compensation and homeostasis in neuron and network function.*
  Nature Reviews Neuroscience 7, 563–574. Review of the parameter-
  degeneracy / homeostatic-tuning literature.

## Henry Markram

Blue Brain Project; cortical microcircuit detail; original
co-author of LSM; **discoverer of STDP**.

- **Markram, H., Lübke, J., Frotscher, M., & Sakmann, B.** (1997).
  *Regulation of synaptic efficacy by coincidence of postsynaptic
  APs and EPSPs.* Science 275, 213–215. The first STDP paper.
  Foundational for silicritter's `silicritter.plasticity` module.
- **Markram, H.** (2006). *The Blue Brain Project.* Nature Reviews
  Neuroscience 7, 153–160. The detailed-cortical-simulation arc.
  Useful as context / cautionary tale for "how detailed should the
  model be?"
- See also: Maass, Natschläger, & Markram (2002) under Maass above.

## Larry Abbott

Theoretical neuroscience standard texts; controller and rate-
dynamics theory. Co-author of the most-used graduate textbook in
the field.

- **Dayan, P., & Abbott, L. F.** (2001). *Theoretical Neuroscience:
  Computational and Mathematical Modeling of Neural Systems.* MIT
  Press. The canonical graduate textbook. Chapters on rate models,
  recurrent networks, plasticity, and stochastic dynamics are
  directly relevant.
- **Sussillo, D., & Abbott, L. F.** (2009). *Generating coherent
  patterns of activity from chaotic neural networks.* Neuron 63(4),
  544–557. The FORCE learning paper. Useful reference if silicritter
  ever explores reservoir-style supervised readout training.

---

## Aston-Jones & Cohen — adaptive gain

The biological inspiration for silicritter's closed-loop adrenaline
controller. Already cited in `experiments/step10_closedloop_adrenaline.py`
and `src/silicritter/closedloop.py` docstrings; included here for
discoverability.

- **Aston-Jones, G., & Cohen, J. D.** (2005). *An integrative theory
  of locus coeruleus-norepinephrine function: Adaptive gain and
  optimal performance.* Annual Review of Neuroscience 28, 403–450.

---

## Reading-strategy notes

- **Cross-check YouTube neuroscience against papers.** The medium
  rewards confident framing of unsettled science; even good
  explainers flatten the controversy. Use Scholar / direct paper
  reading to calibrate any claim that sounds confident.
- **Popular-science books vs peer-reviewed papers.** Some authors'
  books outpace their peer-reviewed work (Hawkins is the obvious
  example); Marder's reviews are tightly tied to her group's
  experimental data. Calibrate accordingly.
- **Mathematical clarity ≠ biological accuracy.** Reservoir
  computing is mathematically clean and easy to test, which makes
  it a *plausible* brain model — not a proven one. Same caution
  applies to any computational framework imported into neuroscience.
- **For silicritter specifically:** theory should *inform* design,
  not over-constrain it. Add references to this file as they prove
  useful, not as ceremony.
