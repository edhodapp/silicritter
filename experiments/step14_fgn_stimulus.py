"""Step 14: fractional-Gaussian-noise stimulus to A.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 10's architecture (E/I + closed-loop adrenaline + hand-wired
cross-E-only B pool) hit a tracking-with-lag residual of ~-5.6e-5
against a piecewise-constant A_DRIVE_PROFILE. That stimulus has no
learnable temporal structure: all "prediction" reduces to tracking
with a one-window lag that concentrates error at segment boundaries.

Step 14 replaces the step-function drive with fractional Gaussian
noise (fGn, see src/silicritter/fracnoise.py) of tunable Hurst
parameter H. fGn is mean-zero stationary; we shift to mean_mv and
scale by std_mv to get a drive trace with the same rough amplitude
range as the step-function profile but with H-controlled
autocorrelation.

Two fitness metrics per run, separating tracking from prediction:
  tracking_fit   = -mean((B_rate[t]   - A_rate[t]  )^2)
  prediction_fit = -mean((B_rate[t]   - A_rate[t+1])^2)

For an architecture with no learnable memory, prediction_fit is
just tracking_fit minus a stimulus-dependent floor tied to A's
own lag-1 autocorrelation. The interesting signal is how both
metrics change as H sweeps.

Hypothesis (stated before running): the current architecture has
no mechanism to exploit long-range dependence. Tracking fit may
vary with H (different dynamics to track) but the gap
between tracking and prediction should be explained entirely by
A's lag-1 autocorrelation, not by any architectural prediction
capability. If that hypothesis holds, the fractional-stimulus
direction requires an architectural change (memory in B, or
fractional EMA in controller) to produce real prediction. If it
fails -- if B's prediction fit tracks H in a way the lag-1
autocorr doesn't explain -- something surprising is happening.

Usage:
    .venv/bin/python experiments/step14_fgn_stimulus.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from silicritter.closedloop import ControllerParams, simulate_closedloop
from silicritter.fracnoise import fgn_drive_trace
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
INHIBITORY_FRACTION: float = 0.2
I_WEIGHT_MULTIPLIER: float = 8.0  # D008

# fGn drive parameters. Matching rough amplitude of A_DRIVE_PROFILE
# (which ranged 18..24 mV, mean ~20.75, range ~6 mV -> std ~2 mV).
FGN_MEAN_MV: float = 20.75
FGN_STD_MV: float = 2.0
B_TONIC_MV: float = 16.0

CONTROLLER_DECAY: float = 0.98
BASELINE_ADRENALINE: float = 1.0
CLOSED_LOOP_GAIN: float = 50.0
ADR_MIN: float = 0.5
ADR_MAX: float = 3.0

HURST_GRID: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)


def _stdp_params() -> STDPParams:
    return default_params()._replace(v_max=V_MAX)


def _ctrl_params(gain: float) -> ControllerParams:
    return ControllerParams(
        decay=CONTROLLER_DECAY, baseline=BASELINE_ADRENALINE, gain=gain,
        adr_min=ADR_MIN, adr_max=ADR_MAX,
    )


def _make_cross_e_only_pool(seed: int) -> SlotPool:
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


def _build_state(seed: int) -> PairedState:
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777)
    )
    pool_b = _make_cross_e_only_pool(seed + 1)
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


def _build_drives(
    hurst: float,
    seed: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """fGn drive for A; constant tonic for B."""
    i_ext_a = fgn_drive_trace(
        N_TIMESTEPS, N_NEURONS,
        hurst=hurst, mean_mv=FGN_MEAN_MV, std_mv=FGN_STD_MV,
        rng=jax.random.PRNGKey(seed + 100),
    )
    i_ext_b = jnp.full(
        (N_TIMESTEPS, N_NEURONS), B_TONIC_MV, dtype=jnp.float32,
    )
    valence = jnp.zeros((N_TIMESTEPS,), dtype=jnp.float32)
    adrenaline_a = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    return i_ext_a, i_ext_b, valence, adrenaline_a


def _window_rates(spikes: jax.Array) -> jax.Array:
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    return spikes.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))


def _tracking_and_prediction_fit(
    spikes_a: jax.Array, spikes_b: jax.Array,
) -> tuple[float, float]:
    a_rate = _window_rates(spikes_a)
    b_rate = _window_rates(spikes_b)
    track = float(-jnp.mean((b_rate - a_rate) ** 2))
    pred = float(-jnp.mean((b_rate[:-1] - a_rate[1:]) ** 2))
    return track, pred


def _stimulus_lag1_autocorr(drive: jax.Array) -> float:
    """Empirical lag-1 autocorrelation of A's per-window mean drive."""
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    drive_mean = drive.reshape(n_windows, WINDOW_STEPS, N_NEURONS).mean(
        axis=(1, 2)
    )
    centered = drive_mean - drive_mean.mean()
    return float(
        (centered[:-1] * centered[1:]).sum() / (centered ** 2).sum()
    )


def _run_condition(
    hurst: float,
    closed_loop: bool,
    seed: int,
) -> tuple[float, float, float, float, float]:
    """Return (track_fit, pred_fit, stim_lag1, A_rate_mean, B_rate_mean)."""
    state = _build_state(seed)
    a_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    i_ext_a, i_ext_b, valence, adr_a = _build_drives(hurst, seed)
    if closed_loop:
        _, spikes_a, spikes_b, _ = simulate_closedloop(
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
    track, pred = _tracking_and_prediction_fit(spikes_a, spikes_b)
    stim_lag1 = _stimulus_lag1_autocorr(i_ext_a)
    a_mean = float(spikes_a.astype(jnp.float32).mean()) * 1000.0
    b_mean = float(spikes_b.astype(jnp.float32).mean()) * 1000.0
    return track, pred, stim_lag1, a_mean, b_mean


def _print_row(
    hurst: float,
    track: float,
    pred: float,
    lag1: float,
    a_rate: float,
    b_rate: float,
) -> None:
    gap = pred - track  # negative = prediction worse than tracking
    print(
        f"  H={hurst:.2f} | "
        f"track {track:11.3e} | "
        f"pred {pred:11.3e} | "
        f"pred-track {gap:+.2e} | "
        f"stim lag1 {lag1:+.3f} | "
        f"A {a_rate:5.1f} Hz | B {b_rate:5.1f} Hz"
    )


def run(seed: int = 0) -> None:
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}, "
        f"tonic(B)={B_TONIC_MV} mV, i_mult={I_WEIGHT_MULTIPLIER} (D008)"
    )
    print(
        f"A drive: fGn(H), mean={FGN_MEAN_MV} mV, std={FGN_STD_MV} mV"
    )
    print(f"Hurst sweep: {list(HURST_GRID)}")
    print()

    print("-- open-loop (const adr=1.0) --")
    for h in HURST_GRID:
        track, pred, lag1, a_r, b_r = _run_condition(
            h, closed_loop=False, seed=seed,
        )
        _print_row(h, track, pred, lag1, a_r, b_r)
    print()

    print("-- closed-loop (gain=50) --")
    for h in HURST_GRID:
        track, pred, lag1, a_r, b_r = _run_condition(
            h, closed_loop=True, seed=seed,
        )
        _print_row(h, track, pred, lag1, a_r, b_r)
    print()

    print("baselines (step-function A, not fGn):")
    print("  step 10 closed-loop gain=50 (D007 i_mult=4): prediction -5.60e-5")
    print()
    print(
        "interpretation: 'pred - track' gap should scale with "
        "stim_var * (1 - lag1) if B has no prediction capacity; "
        "anything richer means the architecture is exploiting "
        "temporal structure beyond the one-window lag."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)
