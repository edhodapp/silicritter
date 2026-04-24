"""Fractional Gaussian noise (fGn) stimulus generator.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

fGn with Hurst parameter H in (0, 1) is the stationary increment
process of fractional Brownian motion. Its autocovariance is

    r(k) = (1/2) * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))

which gives r(0) = 1 (unit variance) and decays as a power law in |k|
rather than exponentially. H = 0.5 recovers white noise (i.i.d.
Gaussian). H > 0.5 gives persistent / long-range positive
correlations: positive increments tend to be followed by positive
increments. H < 0.5 gives antipersistent / negatively correlated
increments.

We use the Davies-Harte FFT method (Davies & Harte 1987), which
generates N exact fGn samples in O(N log N) time via circulant
embedding. The circulant is positive-definite for H in (0, 1) and
length up to a few thousand; numerical clipping handles marginal
cases.

For silicritter this is a stimulus-generation primitive. Use
`fgn_drive_trace` to build an (n_timesteps, n_neurons) current-
injection trace that replaces the piecewise-constant
`A_DRIVE_PROFILE` used in steps 7-13.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def fgn_autocov(n: int, hurst: float) -> jax.Array:
    """fGn autocovariance r(k) for k = 0..n-1, unit variance.

    r(k) = 0.5 * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H)) with r(0) = 1.
    """
    k = jnp.arange(n, dtype=jnp.float32)
    h2 = 2.0 * hurst
    return 0.5 * (
        jnp.abs(k + 1.0) ** h2
        - 2.0 * jnp.abs(k) ** h2
        + jnp.abs(k - 1.0) ** h2
    )


def fgn_davies_harte(
    n: int,
    hurst: float,
    rng: jax.Array,
) -> jax.Array:
    """Generate n samples of unit-variance fGn with Hurst H.

    Args:
        n: number of samples.
        hurst: Hurst parameter, 0 < H < 1. H=0.5 gives white noise;
            H > 0.5 persistent; H < 0.5 antipersistent.
        rng: JAX PRNGKey.

    Returns:
        float32 array of shape (n,), mean ~0, variance ~1.
    """
    r = fgn_autocov(n, hurst)
    zero = jnp.zeros((1,), dtype=jnp.float32)
    c = jnp.concatenate([r, zero, r[1:][::-1]])  # length 2n
    m = 2 * n
    lam = jnp.real(jnp.fft.fft(c))
    lam = jnp.clip(lam, min=0.0)
    k_re, k_im = jax.random.split(rng)
    w_re = jax.random.normal(k_re, (m // 2 + 1,), dtype=jnp.float32)
    w_im = jax.random.normal(k_im, (m // 2 + 1,), dtype=jnp.float32)
    w_im = w_im.at[0].set(0.0)
    w_im = w_im.at[-1].set(0.0)
    freq = (w_re + 1j * w_im) * jnp.sqrt(lam[: m // 2 + 1]).astype(
        jnp.complex64
    )
    # irfft divides by m internally; the conjugate-symmetric
    # spectrum contributes sum(lam) = m * r(0) = m. Raw var ends up
    # at 2/m = 1/n, so scale by sqrt(n) to land at unit variance.
    samples = jnp.fft.irfft(freq, n=m) * jnp.sqrt(jnp.float32(n))
    return samples[:n].astype(jnp.float32)


def fgn_drive_trace(
    n_timesteps: int,
    n_neurons: int,
    hurst: float,
    mean_mv: float,
    std_mv: float,
    rng: jax.Array,
) -> jax.Array:
    """Build a drive trace (n_timesteps, n_neurons) from one fGn path.

    All neurons in the target population receive the same per-timestep
    drive value (spatially uniform). The temporal profile is
    `mean_mv + std_mv * fgn(t)`, where fgn is a unit-variance fGn of
    length n_timesteps at the requested Hurst parameter.
    """
    path = fgn_davies_harte(n_timesteps, hurst, rng)
    drive = mean_mv + std_mv * path
    return jnp.broadcast_to(drive[:, None], (n_timesteps, n_neurons))
