"""Wiener-Kolmogorov prediction-floor computations for fGn stimuli.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

For a zero-mean stationary Gaussian process with autocovariance r(k),
the minimum-MSE one-step-ahead prediction error variance after
n observations is computed exactly by the Durbin-Levinson recursion:

    v_0 = r(0)
    v_n = v_{n-1} * (1 - kappa_n^2)

where `kappa_n` are the partial autocorrelations. As n -> infty, v_n
converges to v_infty, the Wiener-Kolmogorov asymptotic prediction
floor -- no predictor (linear or nonlinear, Gaussian setting) can do
better. v_infty is zero only if the process is deterministic.

For silicritter this module provides:
  - durbin_levinson(r): the recursion on any autocov sequence.
  - windowed_fgn_autocov(n_windows, window_steps, hurst, stride):
    autocovariance of the window-averaged fGn process, generalized
    to arbitrary stride (stride=window_steps = non-overlapping;
    stride=1 = fully-overlapping sliding window).
  - wk_floor_windowed_fgn: convenience wrapper that runs DL on the
    windowed autocov and returns the final prediction-error variance.

The interesting WK floor for step 14's fitness metric is the
non-overlapping version (stride = window_steps). The generalization
is kept for future experiments that might use sliding-window rate
readouts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from silicritter.fracnoise import fgn_autocov


def durbin_levinson(
    r: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run the Durbin-Levinson recursion on autocovariance r.

    Args:
        r: autocovariance sequence of shape (n,), with r[0] > 0.

    Returns:
        Tuple (pacf, v).
        pacf: shape (n-1,), partial autocorrelations kappa_1..kappa_{n-1}.
        v:    shape (n,),   prediction error variances v_0..v_{n-1}.
              v[0] = r[0]; v[k] = v[k-1] * (1 - pacf[k-1]^2).

    At n=1, pacf is empty and v = [r[0]].
    """
    r_host = jnp.asarray(r, dtype=jnp.float32)
    n = int(r_host.shape[0])
    if n == 1:
        return (
            jnp.zeros((0,), dtype=jnp.float32),
            r_host.reshape((1,)),
        )
    # Work in host-side Python lists for this short recursion; the loop
    # length is tiny (n_windows, typically <= 20) and keeping it in
    # Python avoids JAX tracing complications for the variable-length
    # phi update. Output arrays are rebuilt as JAX arrays.
    r_list = [float(x) for x in r_host]
    phi: list[float] = []        # AR coefficients phi_{n,1..n}
    pacf_list: list[float] = []  # kappa_1..kappa_{n-1}
    v_list: list[float] = [r_list[0]]
    for k in range(1, n):
        # kappa_k = (r[k] - sum_{j=1..k-1} phi[j-1] * r[k-j]) / v[k-1]
        num = r_list[k]
        for j in range(1, k):
            num -= phi[j - 1] * r_list[k - j]
        v_prev = v_list[k - 1]
        if v_prev <= 0.0:
            # Deterministic or numerically collapsed process. Remaining
            # PACF entries and v values are zero.
            pacf_list.extend([0.0] * (n - k))
            v_list.extend([0.0] * (n - k))
            break
        kappa = num / v_prev
        pacf_list.append(kappa)
        # Update AR coefficients: phi_new[j] = phi[j] - kappa * phi[k-j-1]
        # for j=1..k-1, and phi_new[k] = kappa.
        new_phi = [phi[j - 1] - kappa * phi[k - j - 1] for j in range(1, k)]
        new_phi.append(kappa)
        phi = new_phi
        # Clamp 1 - kappa^2 to [0, inf). Float32 rounding near |kappa|=1
        # can produce tiny negative values that would propagate as
        # negative variances.
        v_next = v_prev * max(0.0, 1.0 - kappa * kappa)
        v_list.append(v_next)
    return (
        jnp.asarray(pacf_list, dtype=jnp.float32),
        jnp.asarray(v_list, dtype=jnp.float32),
    )


def windowed_fgn_autocov(
    n_windows: int,
    window_steps: int,
    hurst: float,
    stride: int | None = None,
) -> jax.Array:
    """Autocovariance of the window-averaged fGn process.

    For windows W_t = (1/W) * sum_{j=0}^{W-1} X_{t*stride + j} where
    {X} is unit-variance fGn with Hurst parameter `hurst`:

        Cov(W_t, W_{t+d}) = (1/W^2) * sum_{j,k=0..W-1} r(|d*stride + j - k|)

    Args:
        n_windows: number of lag values to return (r_W[0..n_windows-1]).
        window_steps: W, number of samples averaged per window.
        hurst: Hurst parameter H in (0, 1).
        stride: step between window starts. Default None means
            stride = window_steps (non-overlapping). stride < W gives
            overlapping windows; stride > W gives gapped windows.

    Returns:
        Array of shape (n_windows,), r_W[0..n_windows-1].
    """
    s = window_steps if stride is None else stride
    # Build a long-enough fGn autocov sequence to cover all needed lags.
    max_lag = (n_windows - 1) * s + (window_steps - 1) + 1
    r = fgn_autocov(max_lag, hurst)
    js = jnp.arange(window_steps)
    ks = jnp.arange(window_steps)
    # Outer difference j - k of shape (W, W).
    jk_diff = js[:, None] - ks[None, :]
    ds = jnp.arange(n_windows)
    # For each lag d, the argument is |d*s + j - k|.
    # Build a (n_windows, W, W) tensor of absolute lag indices.
    all_lags = jnp.abs(ds[:, None, None] * s + jk_diff[None, :, :])
    # Gather r at those indices and average.
    r_gathered = r[all_lags]
    return r_gathered.mean(axis=(1, 2)).astype(jnp.float32)


def wk_floor_windowed_fgn(
    n_windows: int,
    window_steps: int,
    hurst: float,
    stride: int | None = None,
) -> float:
    """Asymptotic one-step prediction error variance for windowed fGn.

    Runs the Durbin-Levinson recursion on the windowed autocovariance
    and returns the final entry of v, which approximates the
    infinite-history WK floor as n_windows grows.

    Args: same as `windowed_fgn_autocov`.

    Returns:
        Scalar float: v[n_windows - 1].
    """
    r_win = windowed_fgn_autocov(
        n_windows, window_steps, hurst, stride=stride,
    )
    _, v = durbin_levinson(r_win)
    return float(v[-1])
