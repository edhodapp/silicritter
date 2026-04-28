"""Step 16: STDP-driven B-pool learning under closed-loop adrenaline.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Every experiment from step 9 onward has run with `plasticity_rate=0`
everywhere, so STDP was mechanically wired but did no work. Step 16
is the first silicritter experiment in which STDP *actually updates
weights* under a closed-loop reward signal.

Setup:
  - B's initial pool is RANDOM: uniform pre_ids over [0, 2N),
    truncated-Gaussian initial v around mid-range.
  - All B slots have plasticity_rate > 0 (uniform).
  - Valence on B is a shaped reward: v = max(0, 1 - |rate_a_ema -
    rate_b_ema| / valence_scale), computed online from EMA rates.
    Perfect tracking => v = 1 (full Hebbian reinforcement).
    Large error => v = 0 (no reinforcement, no unlearning).
  - Adrenaline controller runs as in step 10 (gain=50, tau_m_scale).
  - A is hand-wired (same as step 10-15), driven by the
    piecewise-constant A_DRIVE_PROFILE.

Three-phase measurement:
  - Phase A (pre-training): plasticity_rate frozen to 0, 2000 steps,
    measure prediction fitness. This is the "random pool" baseline.
  - Phase B (training): plasticity_rate full, 20000 steps, with
    valence-gated STDP and closed-loop adrenaline. Weights evolve.
  - Phase C (post-training): take the trained pool, freeze
    plasticity, 2000 steps, measure prediction fitness.

If Phase C fitness >> Phase A fitness, STDP *actually learns* under
this reward signal. If fitness unchanged or worse, STDP in this
substrate is broken or noise-dominated.

Three failure modes to watch for:
  - Runaway saturation: weights collapse to v_max or v_min.
  - Dithering: weights move but cancel out; mean weight and
    fitness unchanged.
  - Catastrophic weakening: valence signal inadvertently drives
    continuous weakening; B goes silent.

Usage:
    .venv/bin/python experiments/step16_stdp_learning.py
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from silicritter.closedloop import (
    ControllerParams,
    ControllerState,
    init_controller,
    simulate_closedloop,
)
from silicritter.lif import init_state
from silicritter.paired import (
    PairedState,
    cross_e_partner_mask,
    make_pool_for_partner,
    step_paired,
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
V_MAX: float = 2.0
WINDOW_STEPS: int = 100
A_DRIVE_PROFILE: tuple[float, ...] = (18.0, 22.0, 19.0, 24.0)
B_TONIC_MV: float = 16.0
INHIBITORY_FRACTION: float = 0.2
I_WEIGHT_MULTIPLIER: float = 8.0  # D008

# Partner (A) pool seed. Held fixed across all B-seed sweeps so fitness
# differences attribute to B's learning, not partner-substrate
# variation. Cross-partner robustness is a separate experimental
# design (vary this seed independently from B's seed).
PARTNER_SEED: int = 777

CONTROLLER_DECAY: float = 0.98
BASELINE_ADRENALINE: float = 1.0
CLOSED_LOOP_GAIN: float = 50.0
ADR_MIN: float = 0.5
ADR_MAX: float = 3.0

# Measurement-phase length (Phases A and C)
N_MEASURE_STEPS: int = 2_000
# Training-phase length (Phase B)
N_TRAIN_STEPS: int = 20_000

# Random B pool: v drawn from truncated Gaussian(mean, std) in [0, V_MAX].
INIT_V_MEAN: float = 1.0
INIT_V_STD: float = 0.3
# STDP learning rate. Pre-sweep derivation suggested 0.01 would
# produce "meaningful" accumulated weight change over 20k steps
# (dv/spike ~ plasticity_rate * a_plus * spikes); empirically that
# rate sat in the noise regime. The rate sweep (README dev log
# 2026-04-23) showed learning emerges at >= 0.3 with the headline
# +14% result at 1.0. Default reflects the empirical working rate.
PLASTICITY_RATE_TRAIN: float = 1.0

# Valence: rate error at which reward drops to zero. A and B fire
# at ~0.03-0.05 spike/step (30-50 Hz); error of 0.02 is "substantial
# tracking failure." Pick 0.015 to keep the reward signal informative.
VALENCE_SCALE: float = 0.015


def _stdp_params() -> STDPParams:
    return default_params()._replace(v_max=V_MAX)


def _ctrl_params() -> ControllerParams:
    return ControllerParams(
        decay=CONTROLLER_DECAY,
        baseline=BASELINE_ADRENALINE,
        gain=CLOSED_LOOP_GAIN,
        adr_min=ADR_MIN,
        adr_max=ADR_MAX,
    )


def _random_b_pool(
    seed: int,
    plasticity_rate: float,
    init_v_mean: float = INIT_V_MEAN,
    init_v_std: float = INIT_V_STD,
) -> SlotPool:
    """Random B pool: uniform pre_ids over [0, 2N), Gaussian v."""
    rng = jax.random.PRNGKey(seed)
    k_pre, k_v = jax.random.split(rng)
    pre_ids = jax.random.randint(
        k_pre, (N_NEURONS, K_SLOTS),
        minval=0, maxval=2 * N_NEURONS,
        dtype=jnp.int32,
    )
    v = jnp.clip(
        jax.random.normal(k_v, (N_NEURONS, K_SLOTS)) * init_v_std
        + init_v_mean,
        0.0, V_MAX,
    )
    return SlotPool(
        pre_ids=pre_ids,
        v=v,
        plasticity_rate=jnp.full_like(v, plasticity_rate),
        active=jnp.ones_like(v, dtype=jnp.bool_),
        release_counter=jnp.zeros_like(pre_ids, dtype=jnp.int32),
    )


def _build_state(pool_b: SlotPool) -> PairedState:
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(PARTNER_SEED)
    )
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
    n_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """A drive (piecewise-constant) and B tonic drive.

    A runs open-loop, so `adrenaline_a` is not threaded through —
    `step_paired` defaults `adrenaline_a` to ADRENALINE_OPEN_LOOP.
    """
    assert n_steps % len(A_DRIVE_PROFILE) == 0, (
        f"n_steps={n_steps} must be divisible by "
        f"{len(A_DRIVE_PROFILE)} segments"
    )
    seg_len = n_steps // len(A_DRIVE_PROFILE)
    i_ext_a = jnp.concatenate(
        [
            jnp.full((seg_len, N_NEURONS), level, dtype=jnp.float32)
            for level in A_DRIVE_PROFILE
        ]
    )
    i_ext_b = jnp.full(
        (n_steps, N_NEURONS), B_TONIC_MV, dtype=jnp.float32,
    )
    return i_ext_a, i_ext_b


def _training_scan(
    initial_state: PairedState,
    a_is_inh: jax.Array,
    b_is_inh: jax.Array,
    i_ext_a: jax.Array,
    i_ext_b: jax.Array,
) -> tuple[PairedState, jax.Array, jax.Array, jax.Array]:
    """Closed-loop adrenaline + valence-gated STDP, single scan.

    Returns per-step scalar mean rates rather than (T, N_NEURONS) spike
    rasters: at T=10M with N_NEURONS=256 the rasters would be ~5 GB and
    exceed the 4 GB GPU ceiling. Callers only need global means anyway.
    """
    stdp = _stdp_params()
    ctrl_p = _ctrl_params()

    def scan_step(
        carry: tuple[PairedState, ControllerState],
        drive: tuple[jax.Array, jax.Array],
    ) -> tuple[
        tuple[PairedState, ControllerState],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        state, ctrl = carry
        i_ext_a_t, i_ext_b_t = drive
        # A runs open-loop: step_paired's `adrenaline_a` defaults to
        # ADRENALINE_OPEN_LOOP (1.0), so we omit it here. Only B is
        # under closed-loop adrenaline control.
        # Causal: valence at step t reflects controller state through
        # t-1, since step-t spikes haven't happened yet when STDP
        # needs the modulator. The 1-step offset is dominated by the
        # EMA's intrinsic ~50-step averaging window. See D009.
        error_prev = ctrl.rate_a_ema - ctrl.rate_b_ema
        valence_b = jnp.maximum(
            jnp.float32(0.0),
            jnp.float32(1.0) - jnp.abs(error_prev) / VALENCE_SCALE,
        )
        next_state = step_paired(
            state,
            i_ext_a=i_ext_a_t, i_ext_b=i_ext_b_t,
            valence_a=jnp.float32(0.0),
            valence_b=valence_b,
            adrenaline_b=ctrl.adrenaline_b,
            stdp_params=stdp,
            gain_mode="tau_m_scale",
            a_is_inhibitory=a_is_inh,
            b_is_inhibitory=b_is_inh,
            i_weight_multiplier=I_WEIGHT_MULTIPLIER,
        )
        rate_a = next_state.a.lif.spikes.astype(jnp.float32).mean()
        rate_b = next_state.b.lif.spikes.astype(jnp.float32).mean()
        new_rate_a = (
            ctrl.rate_a_ema * ctrl_p.decay
            + rate_a * (1.0 - ctrl_p.decay)
        )
        new_rate_b = (
            ctrl.rate_b_ema * ctrl_p.decay
            + rate_b * (1.0 - ctrl_p.decay)
        )
        new_error = new_rate_a - new_rate_b
        raw_adr = (
            jnp.float32(ctrl_p.baseline)
            + jnp.float32(ctrl_p.gain) * new_error
        )
        new_adr = jnp.clip(
            raw_adr,
            jnp.float32(ctrl_p.adr_min),
            jnp.float32(ctrl_p.adr_max),
        )
        next_ctrl = ControllerState(new_rate_a, new_rate_b, new_adr)
        return (next_state, next_ctrl), (rate_a, rate_b, valence_b)

    initial_ctrl = init_controller(ctrl_p.baseline)
    (final_state, _), (rate_a_trace, rate_b_trace, val_trace) = jax.lax.scan(
        scan_step,
        (initial_state, initial_ctrl),
        (i_ext_a, i_ext_b),
    )
    return final_state, rate_a_trace, rate_b_trace, val_trace


def _measure_fitness(
    pool_b: SlotPool,
    a_is_inh: jax.Array,
    b_is_inh: jax.Array,
    n_steps: int,
) -> float:
    """Plasticity-frozen closed-loop measurement over n_steps."""
    frozen = pool_b._replace(
        plasticity_rate=jnp.zeros_like(pool_b.plasticity_rate),
    )
    state = _build_state(frozen)
    i_ext_a, i_ext_b = _build_drives(n_steps)
    val_zero = jnp.zeros((n_steps,), dtype=jnp.float32)
    # simulate_closedloop (separate controller helper) still requires
    # an adrenaline_a trace; A is open-loop so we pass jnp.ones.
    adr_a_open = jnp.ones((n_steps,), dtype=jnp.float32)
    spikes_a, spikes_b, _ = simulate_closedloop(
        state, _ctrl_params(),
        i_ext_a, i_ext_b, val_zero, val_zero, adr_a_open,
        _stdp_params(),
        gain_mode="tau_m_scale",
        a_is_inhibitory=a_is_inh,
        b_is_inhibitory=b_is_inh,
        i_weight_multiplier=I_WEIGHT_MULTIPLIER,
    )
    assert n_steps % WINDOW_STEPS == 0, (
        f"n_steps={n_steps} must be divisible by "
        f"WINDOW_STEPS={WINDOW_STEPS}"
    )
    n_windows = n_steps // WINDOW_STEPS
    a_rate = spikes_a.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    b_rate = spikes_b.astype(jnp.float32).reshape(
        n_windows, WINDOW_STEPS, N_NEURONS
    ).mean(axis=(1, 2))
    # Fitness = -MSE of (B_k predicting A_{k+1}); 1-window prediction
    # lag. Higher (less negative) is better.
    return float(-jnp.mean((b_rate[:-1] - a_rate[1:]) ** 2))


def _describe_pool(
    pool: SlotPool, a_is_inh: jax.Array,
) -> dict[str, float]:
    """Summary stats for the B pool.

    `a_is_inh` is the partner's E/I identity mask, used to classify
    cross-substrate slots without re-deriving `assign_ei_identity`'s
    layout invariant here.
    """
    v = pool.v
    total = v.size
    cross_mask = pool.pre_ids >= N_NEURONS
    cross_e_mask = cross_e_partner_mask(pool, N_NEURONS, a_is_inh)
    at_max = v >= V_MAX - 1e-4
    at_min = v <= 1e-4
    return {
        "v_mean": float(v.mean()),
        "v_std": float(v.std()),
        "cross_frac": float(cross_mask.sum()) / total,
        "cross_e_frac": float(cross_e_mask.sum()) / total,
        "saturated_at_max_frac": float(at_max.sum()) / total,
        "saturated_at_min_frac": float(at_min.sum()) / total,
    }


def _fmt_pool(stats: dict[str, float]) -> str:
    v_mean = stats["v_mean"]
    v_std = stats["v_std"]
    cross = stats["cross_frac"]
    cross_e = stats["cross_e_frac"]
    sat_max = stats["saturated_at_max_frac"]
    sat_min = stats["saturated_at_min_frac"]
    return (
        f"v={v_mean:.3f}+-{v_std:.3f}, "
        f"cross={cross:.3f}, cross-E={cross_e:.3f}, "
        f"saturated@max={sat_max:.3f}, saturated@min={sat_min:.3f}"
    )


def run(
    seed: int = 0,
    plasticity_rate: float = PLASTICITY_RATE_TRAIN,
    init_v_mean: float = INIT_V_MEAN,
    init_v_std: float = INIT_V_STD,
) -> None:
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS}, K={K_SLOTS}, "
        f"i_mult={I_WEIGHT_MULTIPLIER} (D008), gain_mode=tau_m_scale"
    )
    print(
        f"training: {N_TRAIN_STEPS} steps, "
        f"plasticity_rate={plasticity_rate}, "
        f"valence_scale={VALENCE_SCALE}"
    )
    print(f"A drive: {list(A_DRIVE_PROFILE)} mV, B tonic {B_TONIC_MV} mV")
    print()

    a_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)

    # Phase 0: build the initial random B pool.
    pool_b0 = _random_b_pool(
        seed + 1, plasticity_rate, init_v_mean, init_v_std,
    )
    stats_initial = _describe_pool(pool_b0, a_is_inh)
    print(f"initial pool: {_fmt_pool(stats_initial)}")

    # Phase A: pre-training fitness.
    t0 = time.perf_counter()
    fit_before = _measure_fitness(
        pool_b0, a_is_inh, b_is_inh, N_MEASURE_STEPS,
    )
    print(f"Phase A (pre-training fitness):  {fit_before:.3e} "
          f"({(time.perf_counter()-t0)*1000:.0f} ms)")

    # Phase B: run the training scan.
    print(f"Phase B: training for {N_TRAIN_STEPS} steps...")
    state0 = _build_state(pool_b0)
    i_ext_a, i_ext_b = _build_drives(N_TRAIN_STEPS)
    t0 = time.perf_counter()
    final_state, rate_a_trace, rate_b_trace, val_trace = _training_scan(
        state0, a_is_inh, b_is_inh, i_ext_a, i_ext_b,
    )
    jax.block_until_ready(  # type: ignore[no-untyped-call]
        final_state.b.pool.v)
    dt_train = time.perf_counter() - t0
    print(
        f"  train time: {dt_train:.1f}s "
        f"({N_TRAIN_STEPS / dt_train:.0f} steps/s)"
    )
    # Training-phase diagnostics
    val_mean = float(val_trace.mean())
    val_min = float(val_trace.min())
    val_max = float(val_trace.max())
    a_rate_hz = float(rate_a_trace.mean()) * 1000.0
    b_rate_hz = float(rate_b_trace.mean()) * 1000.0
    print(
        f"  valence during training: "
        f"mean={val_mean:.3f}, min={val_min:.3f}, max={val_max:.3f}"
    )
    print(
        f"  firing rates during training: "
        f"A={a_rate_hz:.1f} Hz, B={b_rate_hz:.1f} Hz"
    )

    trained_pool = final_state.b.pool
    stats_trained = _describe_pool(trained_pool, a_is_inh)
    print(f"trained pool: {_fmt_pool(stats_trained)}")

    # Phase C: post-training fitness.
    t0 = time.perf_counter()
    fit_after = _measure_fitness(
        trained_pool, a_is_inh, b_is_inh, N_MEASURE_STEPS,
    )
    print(f"Phase C (post-training fitness): {fit_after:.3e} "
          f"({(time.perf_counter()-t0)*1000:.0f} ms)")

    # Summary
    print()
    print("summary:")
    print(f"  fitness change: {fit_before:.3e} -> {fit_after:.3e}")
    improvement = fit_after - fit_before
    if improvement > 0:
        label = f"improved by {improvement:.3e}"
    else:
        label = f"degraded by {-improvement:.3e}"
    print(f"  {label}")
    print()
    print("reference (single-seed):")
    print("  step 10 hand-wired cross-E closed-loop: -5.60e-5")
    print("  step 11 CPPN-evolved closed-loop:       -4.92e-5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plasticity-rate", type=float, default=PLASTICITY_RATE_TRAIN,
    )
    parser.add_argument(
        "--init-v-mean", type=float, default=INIT_V_MEAN,
    )
    parser.add_argument(
        "--init-v-std", type=float, default=INIT_V_STD,
    )
    args = parser.parse_args()
    run(
        seed=args.seed,
        plasticity_rate=args.plasticity_rate,
        init_v_mean=args.init_v_mean,
        init_v_std=args.init_v_std,
    )
