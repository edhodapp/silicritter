"""Behavioral tests for fractional Gaussian noise synthesis.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.fracnoise import fgn_davies_harte, fgn_drive_trace


def _autocorr_lag1(x: jax.Array) -> float:
    """Empirical lag-1 autocorrelation (sample)."""
    x_centered = x - x.mean()
    num = float((x_centered[:-1] * x_centered[1:]).sum())
    den = float((x_centered ** 2).sum())
    return num / den


def test_fgn_shape_and_dtype() -> None:
    """fgn returns (n,) float32 of the requested length."""
    samples = fgn_davies_harte(256, 0.7, jax.random.PRNGKey(0))
    assert samples.shape == (256,)
    assert samples.dtype == jnp.float32


def test_fgn_determinism() -> None:
    """Same rng produces byte-identical output."""
    s1 = fgn_davies_harte(128, 0.7, jax.random.PRNGKey(42))
    s2 = fgn_davies_harte(128, 0.7, jax.random.PRNGKey(42))
    assert bool(jnp.all(s1 == s2))


def test_fgn_different_rng_differs() -> None:
    """Different rng produces different output."""
    s1 = fgn_davies_harte(128, 0.7, jax.random.PRNGKey(0))
    s2 = fgn_davies_harte(128, 0.7, jax.random.PRNGKey(1))
    assert not bool(jnp.all(s1 == s2))


def test_fgn_white_noise_at_h05() -> None:
    """H=0.5 should give near-zero lag-1 autocorrelation (white noise)."""
    samples = fgn_davies_harte(4096, 0.5, jax.random.PRNGKey(0))
    rho = _autocorr_lag1(samples)
    # Sampling std for rho at N=4096 is ~1/sqrt(4096) ~ 0.016;
    # 3 sigma bound ~ 0.05.
    assert abs(rho) < 0.05


def test_fgn_persistent_at_h07() -> None:
    """H=0.7 should give positive lag-1 autocorrelation."""
    samples = fgn_davies_harte(4096, 0.7, jax.random.PRNGKey(0))
    rho = _autocorr_lag1(samples)
    # Theoretical lag-1 autocorr at H=0.7:
    #   r(1) = 0.5 * (2^1.4 - 2 + 0) = 0.5 * (2.639 - 2) = 0.3195
    # Allow empirical bands around that.
    assert rho > 0.15


def test_fgn_antipersistent_at_h03() -> None:
    """H=0.3 should give negative lag-1 autocorrelation."""
    samples = fgn_davies_harte(4096, 0.3, jax.random.PRNGKey(0))
    rho = _autocorr_lag1(samples)
    # Theoretical lag-1 autocorr at H=0.3:
    #   r(1) = 0.5 * (2^0.6 - 2 + 0) = 0.5 * (1.516 - 2) = -0.242
    assert rho < -0.1


def test_fgn_mean_near_zero_and_unit_variance() -> None:
    """Large-N sample should have mean ~0, variance ~1 at any H."""
    samples = fgn_davies_harte(8192, 0.6, jax.random.PRNGKey(0))
    assert abs(float(samples.mean())) < 0.05
    assert 0.85 < float(samples.var()) < 1.15


def test_fgn_drive_trace_shape_and_scaling() -> None:
    """fgn_drive_trace returns (T, N) with requested mean and spread."""
    t, n = 1024, 32
    trace = fgn_drive_trace(
        t, n,
        hurst=0.7,
        mean_mv=20.0,
        std_mv=3.0,
        rng=jax.random.PRNGKey(0),
    )
    assert trace.shape == (t, n)
    # All neurons get the same per-step drive value.
    assert bool(jnp.all(trace[:, 0:1] == trace[:, :]))
    # Mean near 20 mV, std near 3 mV.
    values = trace[:, 0]
    assert 19.0 < float(values.mean()) < 21.0
    assert 2.5 < float(values.std()) < 3.5
