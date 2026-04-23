"""Step 10: closed-loop adrenaline controller on the E/I substrate.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 9 validated the E/I substrate at canonical values. Step 10 tests
whether dynamic gain modulation, driven by a feedback controller
reading B's own firing rate, can push past the 30-Hz ceiling that
static cross-input configurations hit.

Controller design (Option B from the design discussion): a leaky-
integrator estimate of A's and B's firing rates, an error signal
(target_rate - measured_rate_B) where target_rate is A's estimate,
and adrenaline = clip(baseline + gain * error, a_min, a_max). The
adrenaline value is fed to step_paired via the `tau_m_scale` gain
mechanism (step 5.5 winner): higher adrenaline shortens B's
effective membrane tau, integrating input faster and raising the
firing-rate ceiling.

Hand-wired B pool: cross-E-only v=2.0 (step 9's best at -1.564e-4,
which hits the 30 Hz ceiling).

Expected outcome: closed-loop adrenaline pushes B's firing rate past
30 Hz when A's rate demands it (segments 1 and 3 at 44 and 50 Hz).
Fitness should drop meaningfully below -1.56e-4.

Usage:
    .venv/bin/python experiments/step10_closedloop_adrenaline.py
"""

from __future__ import annotations

import statistics
from typing import NamedTuple

import jax
import jax.numpy as jnp

from silicritter.lif import init_state
from silicritter.paired import (
    PairedState,
    make_pool_for_partner,
    step_paired,
)
from silicritter.plasticity import (
    PlasticNetState,
    STDPParams,
    default_params,
    init_traces,
)
from silicritter.slotpool import (
    SlotPool,
    assign_ei_identity,
)


N_NEURONS: int = 256
K_SLOTS: int = 32
N_TIMESTEPS: int = 2_000
WINDOW_STEPS: int = 100
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)
B_BASELINE_DRIVE_MV: float = 16.0
V_MAX: float = 2.0
INHIBITORY_FRACTION: float = 0.2
I_WEIGHT_MULTIPLIER: float = 4.0

# Closed-loop adrenaline controller defaults
CONTROLLER_DECAY: float = 0.98   # EMA decay per step (tau ~= 50 ms at dt=1ms)
BASELINE_ADRENALINE: float = 1.0
ADR_MIN: float = 0.5
ADR_MAX: float = 3.0


class ControllerState(NamedTuple):
    """Leaky-integrator state of the adrenaline controller."""

    rate_a_ema: jax.Array
    rate_b_ema: jax.Array
    adrenaline_b: jax.Array


def _stdp_params() -> STDPParams:
    return default_params()._replace(v_max=V_MAX)


def _make_cross_e_only_pool(cross_v: float, seed: int) -> SlotPool:
    """Step 9's winning configuration: all slots bound to A's E neurons."""
    rng = jax.random.PRNGKey(seed)
    n_excitatory = N_NEURONS - int(N_NEURONS * INHIBITORY_FRACTION)
    pre_ids = jax.random.randint(
        rng, (N_NEURONS, K_SLOTS),
        minval=N_NEURONS,
        maxval=N_NEURONS + n_excitatory,
        dtype=jnp.int32,
    )
    v = jnp.full((N_NEURONS, K_SLOTS), cross_v, dtype=jnp.float32)
    return SlotPool(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=jnp.zeros_like(v),
        active=jnp.ones_like(v, dtype=jnp.bool_),
        release_counter=jnp.zeros_like(pre_ids, dtype=jnp.int32),
    )


def _build_traces() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build i_ext_a, i_ext_b, valence (both agents), adrenaline_a traces."""
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


def _build_initial_state(pool_a: SlotPool, pool_b: SlotPool) -> PairedState:
    """Build the initial PairedState with both pools and zero traces."""
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


def _run_closed_loop(
    initial_state: PairedState,
    a_is_inh: jax.Array,
    b_is_inh: jax.Array,
    i_ext_a: jax.Array,
    i_ext_b: jax.Array,
    valence: jax.Array,
    adrenaline_a: jax.Array,
    gain: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run one closed-loop sim; return (spikes_a, spikes_b, adr_trace)."""
    stdp = _stdp_params()

    def scan_step(
        carry: tuple[PairedState, ControllerState],
        drive: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[PairedState, ControllerState],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        paired_state, ctrl = carry
        i_ext_a_t, i_ext_b_t, val_t, adr_a_t = drive
        adr_b_t = ctrl.adrenaline_b
        next_paired = step_paired(
            paired_state,
            i_ext_a=i_ext_a_t, i_ext_b=i_ext_b_t,
            valence_a=val_t, valence_b=val_t,
            adrenaline_a=adr_a_t, adrenaline_b=adr_b_t,
            stdp_params=stdp,
            gain_mode="tau_m_scale",
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=b_is_inh,
            i_weight_multiplier=I_WEIGHT_MULTIPLIER,
        )
        spikes_a = next_paired.a.lif.spikes
        spikes_b = next_paired.b.lif.spikes
        rate_a_now = spikes_a.astype(jnp.float32).mean()
        rate_b_now = spikes_b.astype(jnp.float32).mean()
        new_rate_a = ctrl.rate_a_ema * CONTROLLER_DECAY + rate_a_now * (
            1.0 - CONTROLLER_DECAY
        )
        new_rate_b = ctrl.rate_b_ema * CONTROLLER_DECAY + rate_b_now * (
            1.0 - CONTROLLER_DECAY
        )
        error = new_rate_a - new_rate_b
        new_adr = jnp.clip(
            jnp.float32(BASELINE_ADRENALINE) + jnp.float32(gain) * error,
            jnp.float32(ADR_MIN),
            jnp.float32(ADR_MAX),
        )
        next_ctrl = ControllerState(new_rate_a, new_rate_b, new_adr)
        return (next_paired, next_ctrl), (spikes_a, spikes_b, new_adr)

    initial_ctrl = ControllerState(
        rate_a_ema=jnp.float32(0.0),
        rate_b_ema=jnp.float32(0.0),
        adrenaline_b=jnp.float32(BASELINE_ADRENALINE),
    )
    drives = (i_ext_a, i_ext_b, valence, adrenaline_a)
    _, (spikes_a, spikes_b, adr_trace) = jax.lax.scan(
        scan_step, (initial_state, initial_ctrl), drives
    )
    return spikes_a, spikes_b, adr_trace


def _run_open_loop(
    initial_state: PairedState,
    a_is_inh: jax.Array,
    b_is_inh: jax.Array,
    i_ext_a: jax.Array,
    i_ext_b: jax.Array,
    valence: jax.Array,
    adrenaline_a: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Baseline: constant adrenaline = 1.0 on B. Equivalent to step 9."""
    stdp = _stdp_params()

    def scan_step(
        carry: PairedState,
        drive: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[PairedState, tuple[jax.Array, jax.Array, jax.Array]]:
        i_ext_a_t, i_ext_b_t, val_t, adr_a_t = drive
        adr_b_t = jnp.float32(BASELINE_ADRENALINE)
        next_paired = step_paired(
            carry,
            i_ext_a=i_ext_a_t, i_ext_b=i_ext_b_t,
            valence_a=val_t, valence_b=val_t,
            adrenaline_a=adr_a_t, adrenaline_b=adr_b_t,
            stdp_params=stdp,
            gain_mode="tau_m_scale",
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=b_is_inh,
            i_weight_multiplier=I_WEIGHT_MULTIPLIER,
        )
        return next_paired, (
            next_paired.a.lif.spikes,
            next_paired.b.lif.spikes,
            adr_b_t,
        )

    drives = (i_ext_a, i_ext_b, valence, adrenaline_a)
    _, (spikes_a, spikes_b, adr_trace) = jax.lax.scan(
        scan_step, initial_state, drives
    )
    return spikes_a, spikes_b, adr_trace


def _prediction_fitness(
    spikes_a: jax.Array, spikes_b: jax.Array
) -> float:
    n_windows = N_TIMESTEPS // WINDOW_STEPS
    a_rate = spikes_a.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    b_rate = spikes_b.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    return float(-jnp.mean((b_rate[:-1] - a_rate[1:]) ** 2))


def _summarize(
    label: str,
    spikes_a: jax.Array,
    spikes_b: jax.Array,
    adr_trace: jax.Array,
) -> None:
    """Print a one-line summary for a run."""
    fitness = _prediction_fitness(spikes_a, spikes_b)
    n_segments = len(A_DRIVE_PROFILE)
    b_seg = spikes_b.astype(jnp.float32).reshape(
        n_segments, -1, N_NEURONS
    ).mean(axis=(1, 2)) * 1000.0
    adr_seg = adr_trace.reshape(n_segments, -1).mean(axis=1)
    b_str = "[" + ", ".join(f"{float(x):5.1f}" for x in b_seg) + "]"
    adr_str = "[" + ", ".join(f"{float(x):4.2f}" for x in adr_seg) + "]"
    print(
        f"{label:<30} | fitness {fitness:11.3e} | "
        f"B {b_str} | adr {adr_str}"
    )


def _a_rates_by_segment(spikes_a: jax.Array) -> list[float]:
    n_segments = len(A_DRIVE_PROFILE)
    a_seg = spikes_a.astype(jnp.float32).reshape(
        n_segments, -1, N_NEURONS
    ).mean(axis=(1, 2)) * 1000.0
    return [float(x) for x in a_seg]


SetupBundle = tuple[
    PairedState, jax.Array, jax.Array,
    jax.Array, jax.Array, jax.Array, jax.Array,
]


def _setup_for_seed(seed: int) -> SetupBundle:
    """Build initial_state, E/I masks, and input traces for one seed."""
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777)
    )
    pool_b = _make_cross_e_only_pool(V_MAX, seed + 1)
    initial_state = _build_initial_state(pool_a, pool_b)
    a_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    i_ext_a, i_ext_b, valence, adrenaline_a = _build_traces()
    return (
        initial_state, a_is_inh, b_is_inh,
        i_ext_a, i_ext_b, valence, adrenaline_a,
    )


OPEN_LOOP_LABEL = "open-loop (const adr=1.0)"


def _evaluate_one_seed(
    seed: int,
) -> dict[str, float]:
    """Run all four conditions for one seed; return {condition: fitness}."""
    (initial_state, a_is_inh, b_is_inh,
     i_ext_a, i_ext_b, valence, adrenaline_a) = _setup_for_seed(seed)

    results: dict[str, float] = {}
    spikes_a, spikes_b, _ = _run_open_loop(
        initial_state, a_is_inh, b_is_inh,
        i_ext_a, i_ext_b, valence, adrenaline_a,
    )
    results[OPEN_LOOP_LABEL] = _prediction_fitness(spikes_a, spikes_b)
    for gain in (10.0, 50.0, 200.0):
        spikes_a, spikes_b, _ = _run_closed_loop(
            initial_state, a_is_inh, b_is_inh,
            i_ext_a, i_ext_b, valence, adrenaline_a, gain,
        )
        results[f"closed-loop gain={gain:g}"] = _prediction_fitness(
            spikes_a, spikes_b
        )
    return results


def _print_scenario_header() -> None:
    """Scenario parameters -- shared between single- and multi-seed paths."""
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS}, K={K_SLOTS}, T={N_TIMESTEPS}, "
        f"E:I = {1-INHIBITORY_FRACTION:.0%}:{INHIBITORY_FRACTION:.0%}, "
        f"i_mult={I_WEIGHT_MULTIPLIER}, gain_mode=tau_m_scale"
    )
    print(f"A drive profile (mV): {list(A_DRIVE_PROFILE)}")
    print(
        f"controller: EMA decay={CONTROLLER_DECAY}, baseline="
        f"{BASELINE_ADRENALINE}, adrenaline range=[{ADR_MIN}, {ADR_MAX}]"
    )
    print()


def _run_single_seed_verbose(seed: int) -> None:
    """Single-seed run with per-segment B rates (original step 10 output)."""
    (initial_state, a_is_inh, b_is_inh,
     i_ext_a, i_ext_b, valence, adrenaline_a) = _setup_for_seed(seed)

    spikes_a, spikes_b, adr = _run_open_loop(
        initial_state, a_is_inh, b_is_inh,
        i_ext_a, i_ext_b, valence, adrenaline_a,
    )
    _summarize(OPEN_LOOP_LABEL, spikes_a, spikes_b, adr)
    for gain in (10.0, 50.0, 200.0):
        spikes_a, spikes_b, adr = _run_closed_loop(
            initial_state, a_is_inh, b_is_inh,
            i_ext_a, i_ext_b, valence, adrenaline_a, gain,
        )
        _summarize(
            f"closed-loop gain={gain:g}", spikes_a, spikes_b, adr,
        )
    print()
    a_segments = _a_rates_by_segment(spikes_a)
    a_str = "[" + ", ".join(f"{r:.1f}" for r in a_segments) + "]"
    print(f"A's per-segment rates (Hz):  {a_str}")


def _run_multi_seed(seed: int, n_seeds: int) -> None:
    """Run n_seeds consecutive seeds; aggregate fitness mean/std/min/max."""
    per_condition: dict[str, list[float]] = {}
    for i in range(n_seeds):
        s = seed + i * 37  # stride to get independent draws
        fits = _evaluate_one_seed(s)
        print(f"  seed {s}: " + ", ".join(
            f"{k}={v:.3e}" for k, v in fits.items()
        ))
        for k, v in fits.items():
            per_condition.setdefault(k, []).append(v)
    print()
    header = (
        "condition".ljust(30) + " | "
        + "mean".rjust(11) + " | "
        + "std".rjust(10) + " | "
        + "min".rjust(11) + " | "
        + "max".rjust(11)
    )
    print(header)
    print("-" * len(header))
    for cond, vals in per_condition.items():
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mn = min(vals)
        mx = max(vals)
        print(
            f"{cond:<30} | {mean:11.3e} | {std:10.2e} | "
            f"{mn:11.3e} | {mx:11.3e}"
        )


def run(seed: int = 0, n_seeds: int = 1) -> None:
    """Compare open-loop vs closed-loop adrenaline.

    n_seeds=1: verbose single-seed output (original step 10 format).
    n_seeds>1: aggregate fitness across seeds with mean/std/min/max.
    """
    _print_scenario_header()
    if n_seeds == 1:
        _run_single_seed_verbose(seed)
    else:
        print(f"running {n_seeds} seeds starting at {seed}...")
        _run_multi_seed(seed, n_seeds)
    print()
    print("baselines:")
    print("  step 9 cross-E-only, no closed-loop (constant adr):  -1.56e-4")
    print("  step 8 CPPN GA (direct encoding, no E/I, constant adr): -1.70e-4")
    print()
    print(
        "test: closed-loop adrenaline should raise B's firing rate during "
        "A's peak segments (44, 50 Hz) by shortening tau_m, "
        "pushing past the 30 Hz ceiling."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=1)
    args = parser.parse_args()
    run(seed=args.seed, n_seeds=args.n_seeds)
