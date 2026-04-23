"""Step 13: perturb E/I canonical values around D007 to check locality.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

D007 adopted (inhibitory_fraction, i_weight_multiplier) = (0.2, 4.0)
provisionally, citing literature consensus (van Vreeswijk-Sompolinsky
1996; cortical 4:1 E:I) with an explicit "we may be following the
herd" caveat and a promise to validate by perturbation once the
substrate was stable (steps 9-12 made it stable).

Step 13 runs the promised perturbation: a 5x5 grid of
(inhibitory_fraction, i_weight_multiplier), holding everything else
at step 10's configuration (hand-wired cross-E-only B pool, tonic=16
mV, closed-loop gain=50). Expected outcomes and what each means:

  (a) Smooth local optimum at or near canonical: D007 was a
      reasonable adoption; the substrate isn't sharply sensitive to
      the exact values.
  (b) Flat plateau including canonical: E/I detail doesn't matter
      much for this task; the controller compensates for variation.
  (c) Sharply better off-canonical point: D007 was wrong; supersede
      it with the observed-best values.
  (d) Catastrophic collapse at high I: biologically extreme
      inhibition (high fraction OR high multiplier) silences B
      entirely; the cliff from step 12 reappears in E/I form.

Grid:
  inhibitory_fraction in {0.0, 0.1, 0.2, 0.3, 0.4}
  i_weight_multiplier in {1.0, 2.0, 4.0, 6.0, 8.0}

Canonical is (0.2, 4.0). Reference point: (0.0, anything) is
equivalent to E/I-off, which was step 8's regime (fitness -1.70e-4
hand-wired).

Usage:
    .venv/bin/python experiments/step13_ei_perturbation.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from silicritter.closedloop import ControllerParams, simulate_closedloop
from silicritter.lif import init_state
from silicritter.paired import (
    PairedState,
    make_pool_for_partner,
    simulate_paired,
)
from silicritter.plasticity import (
    PlasticNetState,
    STDPParams,
    default_params,
    init_traces,
)
from silicritter.slotpool import SlotPool, assign_ei_identity


N_NEURONS: int = 256
K_SLOTS: int = 32
N_TIMESTEPS: int = 2_000
WINDOW_STEPS: int = 100
V_MAX: float = 2.0
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)
B_BASELINE_DRIVE_MV: float = 16.0

CONTROLLER_DECAY: float = 0.98
BASELINE_ADRENALINE: float = 1.0
CLOSED_LOOP_GAIN: float = 50.0
ADR_MIN: float = 0.5
ADR_MAX: float = 3.0

I_FRACTION_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)
I_MULT_GRID: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 8.0)
CANONICAL = (0.2, 4.0)


def _stdp_params() -> STDPParams:
    return default_params()._replace(v_max=V_MAX)


def _ctrl_params(gain: float) -> ControllerParams:
    return ControllerParams(
        decay=CONTROLLER_DECAY, baseline=BASELINE_ADRENALINE, gain=gain,
        adr_min=ADR_MIN, adr_max=ADR_MAX,
    )


def _cross_e_only_pool(seed: int, inhibitory_fraction: float) -> SlotPool:
    """Hand-wired cross-E-only: all B slots bound to A's E neurons."""
    rng = jax.random.PRNGKey(seed)
    n_excitatory = N_NEURONS - int(N_NEURONS * inhibitory_fraction)
    pre_ids = jax.random.randint(
        rng, (N_NEURONS, K_SLOTS),
        minval=N_NEURONS,
        maxval=N_NEURONS + n_excitatory,
        dtype=jnp.int32,
    )
    v = jnp.full((N_NEURONS, K_SLOTS), V_MAX, dtype=jnp.float32)
    return SlotPool(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=jnp.zeros_like(v),
        active=jnp.ones_like(v, dtype=jnp.bool_),
        release_counter=jnp.zeros_like(pre_ids, dtype=jnp.int32),
    )


def _build_drives() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    seg_len = N_TIMESTEPS // len(A_DRIVE_PROFILE)
    i_ext_a = jnp.concatenate(
        [
            jnp.full((seg_len, N_NEURONS), level, dtype=jnp.float32)
            for level in A_DRIVE_PROFILE
        ]
    )
    i_ext_b = jnp.full(
        (N_TIMESTEPS, N_NEURONS), B_BASELINE_DRIVE_MV, dtype=jnp.float32,
    )
    valence = jnp.zeros((N_TIMESTEPS,), dtype=jnp.float32)
    adrenaline_a = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    return i_ext_a, i_ext_b, valence, adrenaline_a


def _build_state(seed: int, inhibitory_fraction: float) -> PairedState:
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777)
    )
    pool_b = _cross_e_only_pool(seed + 1, inhibitory_fraction)
    return PairedState(
        a=PlasticNetState(
            lif=init_state(N_NEURONS),
            pool=pool_a,
            traces=init_traces(n_pre=2 * N_NEURONS, n_post=N_NEURONS),
        ),
        b=PlasticNetState(
            lif=init_state(N_NEURONS),
            pool=pool_b,
            traces=init_traces(n_pre=2 * N_NEURONS, n_post=N_NEURONS),
        ),
    )


def _prediction_fitness(
    spikes_a: jax.Array, spikes_b: jax.Array,
) -> float:
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    a_rate = spikes_a.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    b_rate = spikes_b.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    return float(-jnp.mean((b_rate[:-1] - a_rate[1:]) ** 2))


def _ei_masks(
    inhibitory_fraction: float,
) -> tuple[jax.Array | None, jax.Array | None]:
    """None at fraction=0.0 (E/I substrate disabled)."""
    if inhibitory_fraction == 0.0:
        return None, None
    mask = assign_ei_identity(N_NEURONS, inhibitory_fraction)
    return mask, mask


def _run_cell(
    inhibitory_fraction: float,
    i_mult: float,
    closed_loop: bool,
    seed: int,
) -> float:
    state = _build_state(seed, inhibitory_fraction)
    a_is_inh, b_is_inh = _ei_masks(inhibitory_fraction)
    i_ext_a, i_ext_b, valence, adr_a = _build_drives()
    if closed_loop:
        spikes_a, spikes_b, _ = simulate_closedloop(
            state, _ctrl_params(CLOSED_LOOP_GAIN),
            i_ext_a, i_ext_b, valence, valence, adr_a,
            _stdp_params(),
            gain_mode="tau_m_scale",
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=b_is_inh,
            i_weight_multiplier=i_mult,
        )
    else:
        adr_b = jnp.full_like(adr_a, BASELINE_ADRENALINE)
        _, spikes_a, spikes_b = simulate_paired(
            state, i_ext_a, i_ext_b, valence, valence, adr_a, adr_b,
            _stdp_params(),
            gain_mode="tau_m_scale",
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=b_is_inh,
            i_weight_multiplier=i_mult,
        )
    return _prediction_fitness(spikes_a, spikes_b)


def _print_grid(
    label: str,
    fits: dict[tuple[float, float], float],
) -> None:
    print(f"--- {label} fitness grid ---")
    header = "i_mult\\\\i_frac | " + " | ".join(
        f"{f:6.2f}" for f in I_FRACTION_GRID
    )
    print(header)
    print("-" * len(header))
    for m in I_MULT_GRID:
        row = [f"{m:12.1f}"]
        for f in I_FRACTION_GRID:
            key = (f, m)
            val = fits[key]
            mark = "*" if (f, m) == CANONICAL else " "
            row.append(f"{val:6.2e}{mark}")
        print(" | ".join(row))
    print("(*) = D007 canonical (i_frac=0.2, i_mult=4.0)")
    print()


def run(seed: int = 0) -> None:
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}, "
        f"tonic={B_BASELINE_DRIVE_MV} mV, gain_mode=tau_m_scale"
    )
    print(f"A drive profile (mV): {list(A_DRIVE_PROFILE)}")
    print(
        f"grid: i_fraction x i_mult = "
        f"{len(I_FRACTION_GRID)} x {len(I_MULT_GRID)} = "
        f"{len(I_FRACTION_GRID) * len(I_MULT_GRID)} cells per condition"
    )
    print()

    open_fits: dict[tuple[float, float], float] = {}
    closed_fits: dict[tuple[float, float], float] = {}
    for f in I_FRACTION_GRID:
        for m in I_MULT_GRID:
            open_fits[(f, m)] = _run_cell(f, m, closed_loop=False, seed=seed)
            closed_fits[(f, m)] = _run_cell(f, m, closed_loop=True, seed=seed)

    _print_grid("open-loop (const adr=1.0)", open_fits)
    _print_grid("closed-loop (gain=50)", closed_fits)

    canon_open = open_fits[CANONICAL]
    canon_closed = closed_fits[CANONICAL]
    best_open = max(open_fits.items(), key=lambda kv: kv[1])
    best_closed = max(closed_fits.items(), key=lambda kv: kv[1])
    print(
        "canonical (0.2, 4.0) vs best:"
    )
    print(
        f"  open-loop:   canonical={canon_open:.3e}   "
        f"best={best_open[1]:.3e} at i_frac={best_open[0][0]}, "
        f"i_mult={best_open[0][1]}"
    )
    print(
        f"  closed-loop: canonical={canon_closed:.3e}   "
        f"best={best_closed[1]:.3e} at i_frac={best_closed[0][0]}, "
        f"i_mult={best_closed[0][1]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)
