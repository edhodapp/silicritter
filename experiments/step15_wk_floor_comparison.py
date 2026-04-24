"""Step 15: compare step 14 architecture performance to the WK floor.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 14 showed B's prediction MSE under fGn stimulus depends on the
Hurst parameter H, but we have no way to tell whether B is near the
information-theoretic floor ("no predictor could do better") or far
from it ("a memory-capable architecture would close the gap").

Step 15 closes that gap by computing the Wiener-Kolmogorov floor
directly in step 14's units: firing-rate MSE on non-overlapping
100-ms windows. Method:

  1. Run step 14's scenario and extract A's per-window firing rate
     trace for each H.
  2. Estimate the empirical autocovariance of A's window-rate
     process (not the theoretical windowed-fGn autocov -- this
     captures A's actual f-I nonlinearity and whatever dynamics the
     paired sim imposes).
  3. Run Durbin-Levinson on the empirical autocov. The final
     variance entry v[n-1] is the WK floor for predicting
     A_rate[t+1] from A_rate[0..t].
  4. Compare to B's observed prediction MSE from step 14.

Interpretation:
  - If B's MSE is close to WK floor: the architecture is near-optimal
    for this task; memory upgrades won't help much.
  - If B's MSE is far above WK floor: architectural headroom exists;
    a memory-capable predictor could do better.

This experiment answers the "back out or proceed" question for the
fractional-stimulus direction posted in the README dev log.

Usage:
    .venv/bin/python experiments/step15_wk_floor_comparison.py
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
from silicritter.wk import durbin_levinson


N_NEURONS: int = 256
K_SLOTS: int = 32
N_TIMESTEPS: int = 2_000
WINDOW_STEPS: int = 100
V_MAX: float = 2.0
INHIBITORY_FRACTION: float = 0.2
I_WEIGHT_MULTIPLIER: float = 8.0  # D008

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


def _cross_e_only_pool(seed: int) -> SlotPool:
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
        pre_ids=pre_ids, v=v,
        plasticity_rate=jnp.zeros_like(v),
        active=jnp.ones_like(v, dtype=jnp.bool_),
        release_counter=jnp.zeros_like(pre_ids, dtype=jnp.int32),
    )


def _build_state(seed: int) -> PairedState:
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777)
    )
    pool_b = _cross_e_only_pool(seed + 1)
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


def _run_sim(
    hurst: float, closed_loop: bool, seed: int,
) -> tuple[jax.Array, jax.Array]:
    """Return (a_window_rates, b_window_rates) for one scenario."""
    state = _build_state(seed)
    a_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    i_ext_a, i_ext_b, valence, adr_a = _build_drives(hurst, seed)
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
    return _window_rates(spikes_a), _window_rates(spikes_b)


def _empirical_autocov(rates: jax.Array, max_lag: int) -> jax.Array:
    """Biased (consistent) estimator of autocov at lags 0..max_lag-1."""
    centered = rates - rates.mean()
    n = int(rates.shape[0])
    result = []
    for k in range(max_lag):
        valid = n - k
        result.append(float((centered[:valid] * centered[k:]).sum()) / n)
    return jnp.asarray(result, dtype=jnp.float32)


def _wk_floor_from_empirical(
    rates: jax.Array, max_lag: int,
) -> float:
    """Empirical-autocov -> DL -> v[-1]."""
    r_emp = _empirical_autocov(rates, max_lag)
    _, v = durbin_levinson(r_emp)
    return float(v[-1])


def run(seed: int = 0) -> None:
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}, "
        f"i_mult={I_WEIGHT_MULTIPLIER} (D008), "
        f"windows={n_windows}, W={WINDOW_STEPS}"
    )
    print(
        f"A drive: fGn(H), mean={FGN_MEAN_MV} mV, std={FGN_STD_MV} mV"
    )
    print()

    # max_lag: use 10 (half the windows) for DL; shorter is fine since
    # predictive value of distant history is bounded for stationary fGn.
    max_lag = n_windows // 2

    for cond_name, closed in (
        ("open-loop (const adr=1.0)", False),
        ("closed-loop (gain=50)", True),
    ):
        print(f"-- {cond_name} --")
        header = (
            "  H  | A win var  |  WK floor   | B pred MSE  | B/floor | "
            "interpretation"
        )
        print(header)
        print("-" * len(header))
        for h in HURST_GRID:
            a_rates, b_rates = _run_sim(h, closed, seed)
            a_var = float(a_rates.var())
            floor = _wk_floor_from_empirical(a_rates, max_lag)
            b_pred = float(jnp.mean((b_rates[:-1] - a_rates[1:]) ** 2))
            ratio = b_pred / floor if floor > 0.0 else float("inf")
            interp = _interpret(ratio)
            print(
                f" {h:.2f} | {a_var:10.3e} | {floor:11.3e} | "
                f"{b_pred:11.3e} | {ratio:7.2f} | {interp}"
            )
        print()


def _interpret(ratio: float) -> str:
    if ratio < 1.5:
        return "B near WK floor (architecture near-optimal)"
    if ratio < 3.0:
        return "moderate gap (some room)"
    if ratio < 10.0:
        return "large gap (architectural headroom likely)"
    return "very large gap (architectural limit, not task limit)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)
