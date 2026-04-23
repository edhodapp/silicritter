"""Step 9: hand-wired control at N=256 with E/I substrate.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

Step 7c established the architectural-leverage regime at tonic=16 mV,
K=32, v_max=2.0, WITHOUT inhibition. Step 9 adds E/I substrate
(canonical cortical E:I = 4:1, i_weight_multiplier = 4.0) and asks:

1. Does the best-cross configuration (all slots bound to A's E
   neurons) perform the same when the E/I substrate is present but
   not hit by any cross slot?
2. How does a random cross binding (canonical balanced, 80% E and
   20% I) behave -- does the expected-zero-mean input produce
   silence, fluctuation-driven firing, or something else?
3. Does deliberate I-targeted binding produce the expected
   inhibition?

The goal is not to optimize fitness on this task. The goal is to
verify the E/I substrate behaves as advertised so Phase 10 (closed-
loop adrenaline on top) has a validated substrate to run on.

Usage:
    .venv/bin/python experiments/step09_handwired_n256_ei.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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


Scenario = tuple[
    SlotPool,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,  # a_is_inhibitory
    jax.Array,  # b_is_inhibitory
]


def _stdp_params_with_v_max() -> STDPParams:
    return default_params()._replace(v_max=V_MAX)


def _build_scenario() -> Scenario:
    """Build A's fixed pool, drive traces, and E/I assignment for both."""
    seg_len = N_TIMESTEPS // len(A_DRIVE_PROFILE)
    i_ext_a_trace = jnp.concatenate(
        [
            jnp.full((seg_len, N_NEURONS), level, dtype=jnp.float32)
            for level in A_DRIVE_PROFILE
        ]
    )
    i_ext_b_trace = jnp.full(
        (N_TIMESTEPS, N_NEURONS), B_BASELINE_DRIVE_MV, dtype=jnp.float32,
    )
    valence_trace = jnp.zeros((N_TIMESTEPS,), dtype=jnp.float32)
    adrenaline_trace = jnp.ones((N_TIMESTEPS,), dtype=jnp.float32)
    pool_a = make_pool_for_partner(
        N_NEURONS, K_SLOTS, jax.random.PRNGKey(777)
    )
    is_inh = assign_ei_identity(N_NEURONS, INHIBITORY_FRACTION)
    return (
        pool_a,
        i_ext_a_trace, i_ext_b_trace,
        valence_trace, valence_trace,
        adrenaline_trace, adrenaline_trace,
        is_inh, is_inh,
    )


def _frozen_pool(pre_ids: jax.Array, v_matrix: jax.Array) -> SlotPool:
    return SlotPool(
        pre_ids=pre_ids,
        v=v_matrix,
        plasticity_rate=jnp.zeros_like(v_matrix),
        active=jnp.ones_like(v_matrix, dtype=jnp.bool_),
        release_counter=jnp.zeros_like(pre_ids, dtype=jnp.int32),
    )


def _make_pool_cross_random(cross_v: float, seed: int) -> SlotPool:
    """All slots bound uniformly to A-side indices (mix of E and I)."""
    rng = jax.random.PRNGKey(seed)
    pre_ids = jax.random.randint(
        rng, (N_NEURONS, K_SLOTS),
        minval=N_NEURONS, maxval=2 * N_NEURONS,
        dtype=jnp.int32,
    )
    v = jnp.full((N_NEURONS, K_SLOTS), cross_v, dtype=jnp.float32)
    return _frozen_pool(pre_ids, v)


def _make_pool_cross_e_only(cross_v: float, seed: int) -> SlotPool:
    """All slots bound only to A's E neurons (first 80% of A indices)."""
    rng = jax.random.PRNGKey(seed)
    n_excitatory = N_NEURONS - int(N_NEURONS * INHIBITORY_FRACTION)
    # A-side starts at N_NEURONS; A's E neurons occupy [N, N + n_excitatory).
    pre_ids = jax.random.randint(
        rng, (N_NEURONS, K_SLOTS),
        minval=N_NEURONS,
        maxval=N_NEURONS + n_excitatory,
        dtype=jnp.int32,
    )
    v = jnp.full((N_NEURONS, K_SLOTS), cross_v, dtype=jnp.float32)
    return _frozen_pool(pre_ids, v)


def _make_pool_cross_i_only(cross_v: float, seed: int) -> SlotPool:
    """All slots bound only to A's I neurons (last 20% of A indices)."""
    rng = jax.random.PRNGKey(seed)
    n_excitatory = N_NEURONS - int(N_NEURONS * INHIBITORY_FRACTION)
    pre_ids = jax.random.randint(
        rng, (N_NEURONS, K_SLOTS),
        minval=N_NEURONS + n_excitatory,
        maxval=2 * N_NEURONS,
        dtype=jnp.int32,
    )
    v = jnp.full((N_NEURONS, K_SLOTS), cross_v, dtype=jnp.float32)
    return _frozen_pool(pre_ids, v)


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


def _evaluate(
    pool_b: SlotPool,
    scenario: Scenario,
    use_ei: bool,
) -> tuple[float, jax.Array, jax.Array]:
    """Run paired sim; optionally with E/I substrate active."""
    (
        pool_a, i_ext_a, i_ext_b, val_a, val_b, adr_a, adr_b,
        a_is_inh, b_is_inh,
    ) = scenario
    state = PairedState(
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
    _, spikes_a, spikes_b = simulate_paired(
        state,
        i_ext_a, i_ext_b,
        val_a, val_b,
        adr_a, adr_b,
        _stdp_params_with_v_max(),
        a_is_inhibitory=a_is_inh if use_ei else None,
        b_is_inhibitory=b_is_inh if use_ei else None,
        i_weight_multiplier=I_WEIGHT_MULTIPLIER,
    )
    fitness = _prediction_fitness(spikes_a, spikes_b)
    n_segments = len(A_DRIVE_PROFILE)
    b_rate_hz = (
        spikes_b.astype(jnp.float32)
        .reshape(n_segments, -1, N_NEURONS)
        .mean(axis=(1, 2))
    ) * 1000.0
    a_rate_hz = (
        spikes_a.astype(jnp.float32)
        .reshape(n_segments, -1, N_NEURONS)
        .mean(axis=(1, 2))
    ) * 1000.0
    return fitness, a_rate_hz, b_rate_hz


def run(seed: int = 0) -> None:
    """Execute the Step 9 sweep and print results."""
    scenario = _build_scenario()
    n_inh = int(N_NEURONS * INHIBITORY_FRACTION)
    n_exc = N_NEURONS - n_inh

    print(f"device: {jax.default_backend()} / {jax.devices()[0]}")
    print(
        f"scenario: N={N_NEURONS} ({n_exc}E + {n_inh}I), "
        f"K={K_SLOTS}, T={N_TIMESTEPS}, tonic={B_BASELINE_DRIVE_MV} mV, "
        f"v_max={V_MAX}, i_weight_multiplier={I_WEIGHT_MULTIPLIER}"
    )
    print(f"A drive profile (mV): {list(A_DRIVE_PROFILE)}")
    print()

    configs: list[tuple[str, SlotPool, bool]] = [
        (
            "cross-random v=2.0, no E/I (step 7c baseline)",
            _make_pool_cross_random(2.0, seed + 1),
            False,
        ),
        (
            "cross-random v=2.0, E/I ON (canonical balanced)",
            _make_pool_cross_random(2.0, seed + 1),
            True,
        ),
        (
            "cross-E-only v=2.0, E/I ON",
            _make_pool_cross_e_only(2.0, seed + 2),
            True,
        ),
        (
            "cross-I-only v=2.0, E/I ON",
            _make_pool_cross_i_only(2.0, seed + 3),
            True,
        ),
        (
            "cross-E-only v=1.0, E/I ON",
            _make_pool_cross_e_only(1.0, seed + 4),
            True,
        ),
        (
            "cross-E-only v=0.5, E/I ON",
            _make_pool_cross_e_only(0.5, seed + 5),
            True,
        ),
    ]

    header_cols = [
        "config".ljust(48),
        "fitness".rjust(12),
        "B s0".rjust(6),
        "B s1".rjust(6),
        "B s2".rjust(6),
        "B s3".rjust(6),
    ]
    header = " | ".join(header_cols)
    print(header)
    print("-" * len(header))

    a_seg_cache: jax.Array | None = None
    for name, pool_b, use_ei in configs:
        fitness, a_rate_hz, b_rate_hz = _evaluate(pool_b, scenario, use_ei)
        if a_seg_cache is None:
            a_seg_cache = a_rate_hz
        b_seg = [float(x) for x in b_rate_hz]
        row = " | ".join(
            [
                name.ljust(48),
                f"{fitness:12.3e}",
                f"{b_seg[0]:6.1f}",
                f"{b_seg[1]:6.1f}",
                f"{b_seg[2]:6.1f}",
                f"{b_seg[3]:6.1f}",
            ]
        )
        print(row)

    assert a_seg_cache is not None
    a_seg = [float(x) for x in a_seg_cache]
    print()
    print(
        f"A's per-segment firing rates (Hz): "
        f"[{a_seg[0]:.1f}, {a_seg[1]:.1f}, "
        f"{a_seg[2]:.1f}, {a_seg[3]:.1f}]"
    )
    print()
    print("baselines:")
    print("  step 7c, N=256, no E/I, all-cross v=2.0:  -1.74e-4")
    print("  step 7,  N=32, direct GA with STDP:       -5.18e-4")
    print(
        "note: E/I substrate is expected to change the dynamic regime,"
    )
    print(
        "  not necessarily improve fitness on this specific task."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(seed=args.seed)
