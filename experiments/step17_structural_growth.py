"""Step 17: structural-growth sweep -- does release+acquisition help?

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 16 showed that STDP on a random B pool does real work (~14%
fitness improvement at plasticity_rate=1.0) but is bounded by
topology: STDP adjusts v, not pre_ids. Half the slots sit on
counterproductive post-targets (B->B recurrent, B->A-I) that STDP
can weaken to zero but cannot release or reassign.

Step 17 adds the missing lever: slot *acquisition*. Combined with
the pre-existing slot *release*, this closes the exuberance-and-
sculpting loop. Dynamics:
  1. STDP weakens counterproductive slots via LTD.
  2. apply_release retires slots whose v stays below threshold
     for `release_dwell_steps` consecutive steps.
  3. apply_acquisition rebinds inactive slots (with per-step
     Bernoulli probability) to new pre-neurons -- either uniform
     random or Hebbian-biased by recent pre-activity.
  4. New slots get a fresh initial v and full plasticity_rate,
     giving STDP a chance to re-evaluate each acquisition.

Parameter sweep (5 axes):
  - acq_mode:      stochastic | periodic | valence_gated
  - pre_id_source: uniform | hebbian
  - initial_v:     {0.1, 0.5, 1.0} * V_MAX
  - release_threshold: {0.05, 0.2}
  - release_duration:  {200, 1000}

Also two no-acquisition baselines:
  - baseline_stdp_only (step 16 at rate=1.0, no release, no acq)
  - baseline_release_only (release on, acquisition off)

Usage:
    .venv/bin/python experiments/step17_structural_growth.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections.abc import Callable
from typing import NamedTuple

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
from silicritter.structural import (
    StructuralParams,
    apply_acquisition,
    apply_release,
)


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

N_MEASURE_STEPS: int = 2_000
N_TRAIN_STEPS: int = 20_000

INIT_V_MEAN: float = 1.0
INIT_V_STD: float = 0.3
PLASTICITY_RATE: float = 1.0
VALENCE_SCALE: float = 0.015

# Baseline per-step acquisition probability for the stochastic mode:
# expected full pool turnover takes roughly 1/acq_prob steps. 0.001
# gives ~20 full-pool-turnovers in 20000 steps, which is biologically
# plausible (many rebindings over the "lifetime").
ACQ_PROB_STOCHASTIC: float = 0.001
# For periodic mode, rebind every this-many steps:
PERIODIC_INTERVAL_STEPS: int = 500
# For valence_gated mode, max acq prob (scaled by valence 0..1):
ACQ_PROB_VALENCE_MAX: float = 0.002

# Sentinel release_duration for the STDP-only baseline. 500x longer than
# N_TRAIN_STEPS so no slot ever accumulates enough sub-threshold dwell
# to release within the run; lets the baseline isolate STDP-only
# learning from release/acquisition dynamics.
RELEASE_DURATION_DISABLED: int = 10_000_000


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


def _random_b_pool(seed: int) -> SlotPool:
    rng = jax.random.PRNGKey(seed)
    k_pre, k_v = jax.random.split(rng)
    pre_ids = jax.random.randint(
        k_pre, (N_NEURONS, K_SLOTS),
        minval=0, maxval=2 * N_NEURONS,
        dtype=jnp.int32,
    )
    v = jnp.clip(
        jax.random.normal(k_v, (N_NEURONS, K_SLOTS)) * INIT_V_STD
        + INIT_V_MEAN,
        0.0, V_MAX,
    )
    return SlotPool(
        pre_ids=pre_ids, v=v,
        plasticity_rate=jnp.full_like(v, PLASTICITY_RATE),
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
    assert n_steps % len(A_DRIVE_PROFILE) == 0
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


class Config(NamedTuple):
    """Sweep axis tuple for one training run.

    Python-side configuration record (never passed through JAX as a
    pytree). NamedTuple chosen for consistency with the project's
    other record types (SlotPool, PairedState, ControllerState,
    StructuralParams).
    """

    # acq_mode: one of off, stochastic, periodic, valence_gated,
    #           valence_inverted (see _acq_prob_at_step dispatch)
    # pre_id_source: one of uniform, hebbian
    name: str
    acq_mode: str
    pre_id_source: str
    acq_initial_v: float
    release_threshold: float
    release_duration: int


# Acquisition-mode dispatch handlers. All take a uniform
# (step, valence) signature so they can be looked up by mode name in
# `_ACQ_MODE_DISPATCH`; some modes don't consume both arguments.
# pylint: disable=unused-argument
def _acq_prob_off(step: jax.Array, valence: jax.Array) -> jax.Array:
    return jnp.asarray(0.0, dtype=jnp.float32)


def _acq_prob_stochastic(step: jax.Array, valence: jax.Array) -> jax.Array:
    return jnp.asarray(ACQ_PROB_STOCHASTIC, dtype=jnp.float32)


def _acq_prob_periodic(step: jax.Array, valence: jax.Array) -> jax.Array:
    # Fire once per PERIODIC_INTERVAL_STEPS, with prob 1 at the firing
    # step (so all inactive slots rebind at once). The `step > 0` guard
    # skips the spurious step=0 trigger: the initial pool is fully
    # active, so no acquisition would fire anyway, but the test of
    # "first trigger after one full interval" matches the firing-cadence
    # intent.
    hit = jnp.logical_and(
        step > 0, jnp.equal(step % PERIODIC_INTERVAL_STEPS, 0),
    )
    return jnp.where(
        hit,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def _acq_prob_valence_gated(
    step: jax.Array, valence: jax.Array,
) -> jax.Array:
    # Exploit-when-succeeding: prob scales with current valence (high
    # when tracking is good).
    return jnp.asarray(ACQ_PROB_VALENCE_MAX, dtype=jnp.float32) * valence


def _acq_prob_valence_inverted(
    step: jax.Array, valence: jax.Array,
) -> jax.Array:
    # Explore-when-failing (adrenergic-arousal intuition). Prob scales
    # with (1 - valence) so poor tracking drives more rebinding.
    return jnp.asarray(ACQ_PROB_VALENCE_MAX, dtype=jnp.float32) * (
        jnp.asarray(1.0, dtype=jnp.float32) - valence
    )
# pylint: enable=unused-argument


_ACQ_MODE_DISPATCH: dict[
    str, Callable[[jax.Array, jax.Array], jax.Array],
] = {
    "off": _acq_prob_off,
    "stochastic": _acq_prob_stochastic,
    "periodic": _acq_prob_periodic,
    "valence_gated": _acq_prob_valence_gated,
    "valence_inverted": _acq_prob_valence_inverted,
}


def _acq_prob_at_step(
    mode: str, step: jax.Array, valence: jax.Array,
) -> jax.Array:
    """Per-slot acquisition probability this step, by mode.

    Raises ValueError if `mode` is not in the dispatch table — this
    catches typos and unhandled additions to the mode set, instead of
    silently falling through to one branch's logic. `mode` is a Python
    string at trace time (Config field, not a JIT-traced array), so the
    raise is legitimate under jit/scan.
    """
    fn = _ACQ_MODE_DISPATCH.get(mode)
    if fn is None:
        raise ValueError(
            f"unknown acq mode: {mode!r}; "
            f"expected one of {sorted(_ACQ_MODE_DISPATCH)}"
        )
    return fn(step, valence)


def _training_scan(
    initial_state: PairedState,
    a_is_inh: jax.Array,
    b_is_inh: jax.Array,
    i_ext_a: jax.Array,
    i_ext_b: jax.Array,
    config: Config,
    seed: int,
) -> tuple[PairedState, jax.Array, jax.Array]:
    """Closed-loop adrenaline + valence-gated STDP + release + acquisition.

    Per-step spike rasters are intentionally NOT returned: at T=10M with
    N_NEURONS=256 they would be ~5 GB and exceed the 4 GB GPU ceiling.
    Only scalar-per-step traces (valence, active-slot count) are kept;
    callers that need population-level rates compute them inside the scan.
    """
    stdp = _stdp_params()
    ctrl_p = _ctrl_params()
    struct_p = StructuralParams(
        v_release_threshold=config.release_threshold,
        release_dwell_steps=config.release_duration,
        # Safe placeholder. Per-step value is set via _replace inside
        # the scan body from _acq_prob_at_step. If a future refactor
        # ever drops the _replace, defaulting to 0.0 means "no
        # acquisition" rather than "rebind every inactive slot every
        # step" — failing closed.
        acquisition_prob=0.0,
        acquisition_initial_v=config.acq_initial_v,
        acquisition_plasticity_rate=PLASTICITY_RATE,
    )
    use_hebbian = config.pre_id_source == "hebbian"
    acq_mode = config.acq_mode

    def scan_step(
        carry: tuple[PairedState, ControllerState, jax.Array, jax.Array],
        drive: tuple[jax.Array, jax.Array],
    ) -> tuple[
        tuple[PairedState, ControllerState, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
    ]:
        state, ctrl, rng, step = carry
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
        # Apply release on B's pool.
        b_pool_released = apply_release(next_state.b.pool, struct_p)
        # Apply acquisition with per-step mode-dependent probability.
        acq_prob = _acq_prob_at_step(acq_mode, step, valence_b)
        acq_struct = struct_p._replace(acquisition_prob=acq_prob)
        rng, k_acq = jax.random.split(rng)
        pre_activity = (
            next_state.b.traces.pre if use_hebbian else None
        )
        b_pool_acquired = apply_acquisition(
            b_pool_released, acq_struct, k_acq,
            n_pre=2 * N_NEURONS, pre_activity=pre_activity,
        )
        struct_state = next_state._replace(
            b=next_state.b._replace(pool=b_pool_acquired),
        )
        rate_a = struct_state.a.lif.spikes.astype(jnp.float32).mean()
        rate_b = struct_state.b.lif.spikes.astype(jnp.float32).mean()
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
        n_active = b_pool_acquired.active.astype(jnp.int32).sum()
        return (struct_state, next_ctrl, rng, step + 1), (
            valence_b,
            n_active,
        )

    initial_ctrl = init_controller(ctrl_p.baseline)
    init_rng = jax.random.PRNGKey(seed + 7000)
    init_step = jnp.int32(0)
    (final_state, _, _, _), (val_trace, active_trace) = jax.lax.scan(
        scan_step,
        (initial_state, initial_ctrl, init_rng, init_step),
        (i_ext_a, i_ext_b),
    )
    return final_state, val_trace, active_trace


def _measure_fitness(
    pool_b: SlotPool,
    a_is_inh: jax.Array,
    b_is_inh: jax.Array,
    n_steps: int,
) -> float:
    frozen = pool_b._replace(
        plasticity_rate=jnp.zeros_like(pool_b.plasticity_rate),
    )
    state = _build_state(frozen)
    i_ext_a, i_ext_b = _build_drives(n_steps)
    val_zero = jnp.zeros((n_steps,), dtype=jnp.float32)
    # simulate_closedloop (separate controller helper) still requires
    # an adrenaline_a trace; A is open-loop so we pass jnp.ones.
    adr_a_open = jnp.ones((n_steps,), dtype=jnp.float32)
    _, spikes_a, spikes_b, _ = simulate_closedloop(
        state, _ctrl_params(),
        i_ext_a, i_ext_b, val_zero, val_zero, adr_a_open,
        _stdp_params(),
        gain_mode="tau_m_scale",
        a_is_inhibitory=a_is_inh,
        b_is_inhibitory=b_is_inh,
        i_weight_multiplier=I_WEIGHT_MULTIPLIER,
    )
    assert n_steps % WINDOW_STEPS == 0
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


def _run_config(config: Config, seed: int) -> dict[str, float]:
    """Run one config end-to-end; return metrics dict."""
    a_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    b_is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    pool_b0 = _random_b_pool(seed + 1)
    fit_before = _measure_fitness(pool_b0, a_is_inh, b_is_inh, N_MEASURE_STEPS)
    state0 = _build_state(pool_b0)
    i_ext_a, i_ext_b = _build_drives(N_TRAIN_STEPS)
    t0 = time.perf_counter()
    final_state, val_trace, active_trace = _training_scan(
        state0, a_is_inh, b_is_inh,
        i_ext_a, i_ext_b,
        config, seed,
    )
    jax.block_until_ready(  # type: ignore[no-untyped-call]
        final_state.b.pool.v)
    train_time = time.perf_counter() - t0
    trained_pool = final_state.b.pool
    fit_after = _measure_fitness(
        trained_pool, a_is_inh, b_is_inh, N_MEASURE_STEPS,
    )
    cross_e_mask = cross_e_partner_mask(
        trained_pool, N_NEURONS, a_is_inh,
    )
    return {
        "fit_before": fit_before,
        "fit_after": fit_after,
        "train_time": train_time,
        "v_mean": float(trained_pool.v.mean()),
        "v_std": float(trained_pool.v.std()),
        "active_frac_end": float(trained_pool.active.sum())
        / trained_pool.active.size,
        "cross_e_frac_end": float(cross_e_mask.sum())
        / trained_pool.pre_ids.size,
        "active_frac_min": float(active_trace.min())
        / trained_pool.active.size,
        "valence_mean": float(val_trace.mean()),
    }


def _enumerate_configs() -> list[Config]:
    configs: list[Config] = [
        Config(
            "baseline_stdp_only", "off", "uniform",
            0.2, 0.0, RELEASE_DURATION_DISABLED,
        ),
        Config(
            "baseline_release_only", "off", "uniform", 0.2, 0.05, 500,
        ),
    ]
    for (mode, source, init_v, thr, dur) in itertools.product(
        ("stochastic", "periodic", "valence_gated", "valence_inverted"),
        ("uniform", "hebbian"),
        (0.2, 1.0, 1.8),
        (0.05, 0.2),
        (200, 1000),
    ):
        name = (
            f"{mode}_{source}_iv{init_v:.1f}_thr{thr:.2f}_dur{dur}"
        )
        configs.append(Config(name, mode, source, init_v, thr, dur))
    return configs


def _print_header() -> None:
    print(
        "config".ljust(48) + " | "
        + "fit_before".rjust(11) + " | "
        + "fit_after".rjust(11) + " | "
        + "delta".rjust(10) + " | "
        + "v_mean".rjust(7) + " | "
        + "active%".rjust(8) + " | "
        + "cross-E%".rjust(9) + " | "
        + "time"
    )
    print("-" * 120)


def _print_row(config: Config, metrics: dict[str, float]) -> None:
    fit_b = metrics["fit_before"]
    fit_a = metrics["fit_after"]
    delta = fit_a - fit_b
    v_mean = metrics["v_mean"]
    active_pct = 100 * metrics["active_frac_end"]
    cross_e_pct = 100 * metrics["cross_e_frac_end"]
    train_time = metrics["train_time"]
    print(
        f"{config.name[:48]:48s} | "
        f"{fit_b:11.3e} | "
        f"{fit_a:11.3e} | "
        f"{delta:+10.2e} | "
        f"{v_mean:7.3f} | "
        f"{active_pct:7.1f}% | "
        f"{cross_e_pct:8.1f}% | "
        f"{train_time:5.1f}s"
    )


def run(seed: int = 0) -> None:
    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS}, K={K_SLOTS}, "
        f"i_mult={I_WEIGHT_MULTIPLIER} (D008), "
        f"train={N_TRAIN_STEPS} steps, measure={N_MEASURE_STEPS} steps"
    )
    configs = _enumerate_configs()
    print(f"running {len(configs)} configs...")
    print()
    _print_header()
    for config in configs:
        metrics = _run_config(config, seed)
        _print_row(config, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)
