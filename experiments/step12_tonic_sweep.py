"""Step 12: tonic-drive sweep under E/I + closed-loop.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 7 introduced `B_BASELINE_DRIVE_MV = 16.0` as a kludge to keep B's
firing regime above zero under weak cross-input from A. Every step
since has inherited it. Step 12 asks: now that we have E/I substrate
(step 9) and closed-loop adrenaline (step 10), how much of the tonic
can we peel off before B's tracking breaks?

Inhibition cannot replace tonic outright -- no input means no spikes,
and inhibition only shapes existing excitation. But closed-loop gain
might compensate: if tonic is low during A's peaks, the controller
ramps adrenaline, shortening B's tau_m and letting B reach threshold
on cross-input alone. The hypothesis is that closed-loop lets tonic
go significantly lower than open-loop allows.

Sweep tonic in {16, 12, 8, 4, 0} mV. For each value, run:
  - open-loop (constant adr=1.0)
  - closed-loop (gain=50)
with step 10's hand-wired cross-E-only B pool (step 9's best static
configuration). Record fitness and B's per-segment firing rates.

Expected:
  - open-loop fitness degrades monotonically as tonic drops; at
    tonic=0, B may barely fire during A's quiet segments (28 Hz).
  - closed-loop fitness degrades more gracefully; gain ramps to
    compensate when tonic is low. If tonic=0 closed-loop still
    tracks A, the tonic kludge is formally redundant under
    closed-loop control.

Usage:
    .venv/bin/python experiments/step12_tonic_sweep.py
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
INHIBITORY_FRACTION: float = 0.2
I_WEIGHT_MULTIPLIER: float = 4.0

CONTROLLER_DECAY: float = 0.98
BASELINE_ADRENALINE: float = 1.0
CLOSED_LOOP_GAIN: float = 50.0
ADR_MIN: float = 0.5
ADR_MAX: float = 3.0

TONIC_SWEEP_MV: tuple[float, ...] = (16.0, 12.0, 8.0, 4.0, 0.0)


def _stdp_params() -> STDPParams:
    return default_params()._replace(v_max=V_MAX)


def _ctrl_params(gain: float) -> ControllerParams:
    return ControllerParams(
        decay=CONTROLLER_DECAY, baseline=BASELINE_ADRENALINE, gain=gain,
        adr_min=ADR_MIN, adr_max=ADR_MAX,
    )


def _make_cross_e_only_pool(seed: int) -> SlotPool:
    """Step 9/10's winning configuration: all slots bound to A's E neurons."""
    rng = jax.random.PRNGKey(seed)
    n_excitatory = N_NEURONS - int(N_NEURONS * INHIBITORY_FRACTION)
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


def _build_drives(tonic_mv: float) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array
]:
    """i_ext_a (segmented), i_ext_b (constant tonic), valence, adr_a."""
    seg_len = N_TIMESTEPS // len(A_DRIVE_PROFILE)
    i_ext_a = jnp.concatenate(
        [
            jnp.full((seg_len, N_NEURONS), level, dtype=jnp.float32)
            for level in A_DRIVE_PROFILE
        ]
    )
    i_ext_b = jnp.full(
        (N_TIMESTEPS, N_NEURONS), tonic_mv, dtype=jnp.float32,
    )
    valence = jnp.zeros((N_TIMESTEPS,), dtype=jnp.float32)
    adrenaline_a = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    return i_ext_a, i_ext_b, valence, adrenaline_a


def _build_state(pool_b_seed: int) -> PairedState:
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777)
    )
    pool_b = _make_cross_e_only_pool(pool_b_seed)
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


def _segment_rates(spikes: jax.Array) -> list[float]:
    """Mean firing rate (Hz) per A drive segment."""
    seg_len = N_TIMESTEPS // len(A_DRIVE_PROFILE)
    reshaped = spikes.astype(jnp.float32).reshape(
        len(A_DRIVE_PROFILE), seg_len, N_NEURONS
    )
    return [float(r) * 1000.0 for r in reshaped.mean(axis=(1, 2))]


def _run_condition(
    tonic_mv: float,
    closed_loop: bool,
    seed: int,
) -> tuple[float, list[float]]:
    """Return (fitness, B's per-segment rates in Hz)."""
    state = _build_state(seed + 1)
    a_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    i_ext_a, i_ext_b, valence, adr_a = _build_drives(tonic_mv)
    if closed_loop:
        spikes_a, spikes_b, _ = simulate_closedloop(
            state, _ctrl_params(CLOSED_LOOP_GAIN),
            i_ext_a, i_ext_b, valence, valence, adr_a,
            _stdp_params(),
            gain_mode="tau_m_scale",
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=b_is_inh,
            i_weight_multiplier=I_WEIGHT_MULTIPLIER,
        )
    else:
        adr_b = jnp.full_like(adr_a, BASELINE_ADRENALINE)
        _, spikes_a, spikes_b = simulate_paired(
            state, i_ext_a, i_ext_b, valence, valence, adr_a, adr_b,
            _stdp_params(),
            gain_mode="tau_m_scale",
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=b_is_inh,
            i_weight_multiplier=I_WEIGHT_MULTIPLIER,
        )
    return _prediction_fitness(spikes_a, spikes_b), _segment_rates(spikes_b)


def _a_rates(seed: int) -> list[float]:
    """A's per-segment rates are the reference target."""
    state = _build_state(seed + 1)
    a_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    # Use tonic=16 just for the reference A run; A is insensitive to B's tonic
    # except through the cross-feedback path, and we only need A's rates.
    i_ext_a, i_ext_b, valence, adr_a = _build_drives(16.0)
    adr_b = jnp.full_like(adr_a, BASELINE_ADRENALINE)
    _, spikes_a, _ = simulate_paired(
        state, i_ext_a, i_ext_b, valence, valence, adr_a, adr_b,
        _stdp_params(),
        gain_mode="tau_m_scale",
        a_is_inhibitory=a_is_inh,
        b_is_inhibitory=b_is_inh,
        i_weight_multiplier=I_WEIGHT_MULTIPLIER,
    )
    return _segment_rates(spikes_a)


def _fmt_rates(rates: list[float]) -> str:
    return "[" + ", ".join(f"{r:5.1f}" for r in rates) + "]"


def run(seed: int = 0) -> None:
    """Sweep tonic across open-loop and closed-loop; report fitness + rates."""
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}, "
        f"E:I = {1-INHIBITORY_FRACTION:.0%}:{INHIBITORY_FRACTION:.0%}, "
        f"i_mult={I_WEIGHT_MULTIPLIER}, gain_mode=tau_m_scale"
    )
    print(f"A drive profile (mV): {list(A_DRIVE_PROFILE)}")
    print(f"tonic sweep (mV): {list(TONIC_SWEEP_MV)}")
    print()

    a_segments = _a_rates(seed)
    print(f"A's per-segment rates (reference): {_fmt_rates(a_segments)} Hz")
    print()

    header = (
        "tonic".rjust(5) + " | "
        + "open-loop fit".rjust(14) + " | "
        + "open-loop B rates (Hz)".ljust(28) + " | "
        + "closed-loop fit".rjust(16) + " | "
        + "closed-loop B rates (Hz)"
    )
    print(header)
    print("-" * len(header))
    for tonic in TONIC_SWEEP_MV:
        ol_fit, ol_rates = _run_condition(tonic, closed_loop=False, seed=seed)
        cl_fit, cl_rates = _run_condition(tonic, closed_loop=True, seed=seed)
        print(
            f"{tonic:5.1f} | "
            f"{ol_fit:14.3e} | "
            f"{_fmt_rates(ol_rates):28s} | "
            f"{cl_fit:16.3e} | "
            f"{_fmt_rates(cl_rates)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)
