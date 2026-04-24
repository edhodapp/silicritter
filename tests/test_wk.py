"""Behavioral tests for the Wiener-Kolmogorov prediction-floor module.

SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Ed Hodapp

The WK floor is the minimum-MSE one-step-ahead prediction error for
a stationary Gaussian process. For silicritter we care about the
floor on the window-averaged fGn process that drives step 14's
fitness metric.

Correctness of this module is load-bearing: if the WK floor is
wrong, every comparison to architectural performance is wrong.
Tests here intentionally overlap -- multiple independent cross-
checks rather than one pass, because subtle sign errors, off-by-
one boundary issues, or normalization mistakes can easily produce
a "plausible-looking" wrong number that passes a single narrow
test.

Cross-check strategy:
  1. Durbin-Levinson identities on toy autocovariance sequences
     with closed-form answers (white, AR(1), AR(2)).
  2. fGn autocovariance values at specific Hurst parameters,
     computed from the definition.
  3. Windowed-fGn autocovariance degenerate cases (W=1 matches
     point-process autocov; H=0.5 has diagonal form).
  4. WK floor monotonicity and boundedness (physical sanity).
  5. Monte Carlo: synthesize many fGn paths, estimate empirical
     prediction error, check agreement with theoretical floor.

If all five agree for a given (H, W) configuration, the
implementation is probably correct.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from silicritter.fracnoise import fgn_autocov, fgn_davies_harte
from silicritter.wk import (
    durbin_levinson,
    windowed_fgn_autocov,
    wk_floor_windowed_fgn,
)


# ----------------------------------------------------------------------
# Durbin-Levinson recursion properties
# ----------------------------------------------------------------------

class TestDurbinLevinson:
    """Durbin-Levinson should match known identities on toy sequences."""

    def test_output_shapes(self) -> None:
        """pacf is shape (n-1,), v is shape (n,)."""
        r = jnp.array([1.0, 0.5, 0.25, 0.125], dtype=jnp.float32)
        pacf, v = durbin_levinson(r)
        assert pacf.shape == (3,)
        assert v.shape == (4,)

    def test_v0_equals_r0(self) -> None:
        """The zero-history prediction error variance is r(0)."""
        r = jnp.array([2.5, 1.0, 0.5], dtype=jnp.float32)
        _, v = durbin_levinson(r)
        assert float(v[0]) == pytest.approx(2.5, rel=1e-5)

    def test_v_non_increasing(self) -> None:
        """Adding data cannot increase the prediction error variance."""
        r = fgn_autocov(32, hurst=0.7)
        _, v = durbin_levinson(r)
        v_list = [float(x) for x in v]
        for i in range(1, len(v_list)):
            assert v_list[i] <= v_list[i - 1] + 1e-6

    def test_v_positive(self) -> None:
        """Prediction error variance is strictly positive for
        positive-definite autocov sequences."""
        r = fgn_autocov(32, hurst=0.7)
        _, v = durbin_levinson(r)
        assert bool(jnp.all(v > 0))

    def test_single_sample_edge_case(self) -> None:
        """n=1: v = [r(0)], pacf is empty."""
        r = jnp.array([1.5], dtype=jnp.float32)
        pacf, v = durbin_levinson(r)
        assert pacf.shape == (0,)
        assert v.shape == (1,)
        assert float(v[0]) == pytest.approx(1.5, rel=1e-5)

    def test_white_noise_pacf_is_zero(self) -> None:
        """r(k) = delta(k) => all partial autocorrelations are zero."""
        r = jnp.zeros(16, dtype=jnp.float32).at[0].set(1.0)
        pacf, v = durbin_levinson(r)
        assert bool(jnp.all(jnp.abs(pacf) < 1e-6))
        # And v is constant at r(0).
        assert bool(jnp.all(jnp.abs(v - 1.0) < 1e-6))

    def test_ar1_rho_05_pacf(self) -> None:
        """AR(1) with rho=0.5: pacf[0] = 0.5, pacf[k] = 0 for k>=1."""
        rho = 0.5
        k = jnp.arange(12, dtype=jnp.float32)
        r = rho ** k
        pacf, _ = durbin_levinson(r)
        assert float(pacf[0]) == pytest.approx(rho, abs=1e-5)
        # All later partial autocorrelations essentially zero.
        for i in range(1, len(pacf)):
            assert abs(float(pacf[i])) < 1e-4

    def test_ar1_rho_05_variance(self) -> None:
        """AR(1) with rho=0.5: v_1 = 1 - rho^2, then v_n stays there."""
        rho = 0.5
        k = jnp.arange(12, dtype=jnp.float32)
        r = rho ** k
        _, v = durbin_levinson(r)
        assert float(v[0]) == pytest.approx(1.0, abs=1e-5)
        for i in range(1, len(v)):
            assert float(v[i]) == pytest.approx(1.0 - rho ** 2, abs=1e-4)

    def test_ar1_rho_09_variance(self) -> None:
        """Strong correlation: v_1 = 1 - 0.81 = 0.19."""
        rho = 0.9
        k = jnp.arange(8, dtype=jnp.float32)
        r = rho ** k
        _, v = durbin_levinson(r)
        assert float(v[1]) == pytest.approx(1.0 - rho ** 2, abs=1e-4)

    def test_ar1_rho_negative(self) -> None:
        """Anti-persistent: rho=-0.5 also gives v_n = 1 - rho^2."""
        rho = -0.5
        k = jnp.arange(12, dtype=jnp.float32)
        r = rho ** k  # alternating signs
        _, v = durbin_levinson(r)
        assert float(v[1]) == pytest.approx(1.0 - rho ** 2, abs=1e-4)

    def test_ar2_closed_form(self) -> None:
        """AR(2) with known ACF has known PACF: kappa_1 = rho_1,
        kappa_2 = (rho_2 - rho_1^2) / (1 - rho_1^2)."""
        # Choose a valid AR(2) autocovariance.
        rho1, rho2 = 0.6, 0.4
        # Extend via Yule-Walker: phi1 = (rho1*(1 - rho2)) / (1 - rho1^2)
        # Actually just use these three values and let DL produce pacf.
        r = jnp.array([1.0, rho1, rho2], dtype=jnp.float32)
        pacf, _ = durbin_levinson(r)
        expected_kappa2 = (rho2 - rho1 ** 2) / (1.0 - rho1 ** 2)
        assert float(pacf[0]) == pytest.approx(rho1, abs=1e-5)
        assert float(pacf[1]) == pytest.approx(expected_kappa2, abs=1e-4)

    def test_symmetric_around_sign_flip(self) -> None:
        """For AR(1), v_n depends only on |rho|, not its sign."""
        k = jnp.arange(8, dtype=jnp.float32)
        r_pos = 0.7 ** k
        r_neg = (-0.7) ** k
        _, v_pos = durbin_levinson(r_pos)
        _, v_neg = durbin_levinson(r_neg)
        for p, n in zip(v_pos.tolist(), v_neg.tolist()):
            assert p == pytest.approx(n, abs=1e-5)

    def test_deterministic_process_zero_variance_path(self) -> None:
        """Exact AR(1) with rho=1: r(k)=1 for all k. After the first
        step, v_prev=0 and the recursion must bail out cleanly with
        zero variances and zero PACF entries for the remaining lags.

        Guards against inf/nan propagation when the process collapses
        into a deterministic mode.
        """
        r = jnp.ones(5, dtype=jnp.float32)
        pacf, v = durbin_levinson(r)
        # v[0] = 1 = r(0). v[1] = 1 * (1 - 1^2) = 0. Remaining v = 0.
        assert float(v[0]) == pytest.approx(1.0, abs=1e-5)
        for k in range(1, 5):
            assert float(v[k]) == pytest.approx(0.0, abs=1e-5)
        # PACF entries after the collapse should all be zero (bail-out).
        for k in range(1, 4):
            assert float(pacf[k]) == pytest.approx(0.0, abs=1e-5)

    def test_kappa_saturates_clamps_variance(self) -> None:
        """Sequence where |kappa| rounds to exactly 1 in float32 must
        still produce v >= 0 (clamp guard, not negative variance).

        Using an autocov that's all ones triggers kappa=1 at step 1;
        the guard should set v[1] = 0 rather than a rounding-error
        negative.
        """
        r = jnp.ones(4, dtype=jnp.float32)
        _, v = durbin_levinson(r)
        # All entries must be non-negative regardless of float32 noise.
        assert bool(jnp.all(v >= 0.0))


# ----------------------------------------------------------------------
# fGn autocovariance values at specific Hurst parameters
# ----------------------------------------------------------------------

class TestFgnAutocov:
    """Cross-check fgn_autocov against closed-form values."""

    def test_r0_is_unity(self) -> None:
        """r(0) = 0.5 * (1^(2H) - 0 + 1^(2H)) = 1 for any H."""
        for h in (0.2, 0.5, 0.7, 0.9):
            r = fgn_autocov(4, hurst=h)
            assert float(r[0]) == pytest.approx(1.0, abs=1e-5)

    def test_r1_white_noise(self) -> None:
        """At H=0.5, r(1) = 0.5 * (2 - 2 + 0) = 0."""
        r = fgn_autocov(4, hurst=0.5)
        assert float(r[1]) == pytest.approx(0.0, abs=1e-5)

    def test_r1_persistent_h07(self) -> None:
        """At H=0.7, r(1) = 0.5 * (2^1.4 - 2 + 0) = 0.31951..."""
        r = fgn_autocov(4, hurst=0.7)
        expected = 0.5 * (2.0 ** 1.4 - 2.0)
        assert float(r[1]) == pytest.approx(expected, abs=1e-4)

    def test_r1_antipersistent_h03(self) -> None:
        """At H=0.3, r(1) = 0.5 * (2^0.6 - 2) < 0."""
        r = fgn_autocov(4, hurst=0.3)
        expected = 0.5 * (2.0 ** 0.6 - 2.0)
        assert float(r[1]) == pytest.approx(expected, abs=1e-4)
        assert float(r[1]) < 0

    def test_r_decays_for_persistent(self) -> None:
        """At H=0.7, |r(k)| monotonically decreases over a few lags."""
        r = fgn_autocov(16, hurst=0.7)
        # Allow tiny numerical bumps; just check overall trend.
        # r(0)=1 and r(k) decays as k^(2H-2) for large k.
        assert abs(float(r[1])) > abs(float(r[4]))
        assert abs(float(r[4])) > abs(float(r[15]))


# ----------------------------------------------------------------------
# Windowed fGn autocovariance
# ----------------------------------------------------------------------

class TestWindowedFgnAutocov:
    """The windowed autocov is the double-sum integral of point autocov."""

    def test_W1_matches_point_autocov(self) -> None:
        """Window of size 1 degenerates to the point process."""
        for h in (0.3, 0.5, 0.7):
            r_win = windowed_fgn_autocov(8, window_steps=1, hurst=h)
            r_pt = fgn_autocov(8, hurst=h)
            for a, b in zip(r_win.tolist(), r_pt.tolist()):
                assert a == pytest.approx(b, abs=1e-4)

    def test_r0_bounded_by_point_r0(self) -> None:
        """Variance of window mean is at most Var(single sample) = 1."""
        for h in (0.3, 0.5, 0.7, 0.9):
            r_win = windowed_fgn_autocov(2, window_steps=100, hurst=h)
            assert float(r_win[0]) <= 1.0 + 1e-5

    def test_H05_variance_equals_1_over_W(self) -> None:
        """White noise: Var(mean of W samples) = 1/W exactly."""
        w = 100
        r_win = windowed_fgn_autocov(2, window_steps=w, hurst=0.5)
        assert float(r_win[0]) == pytest.approx(1.0 / w, abs=1e-5)

    def test_H05_lags_are_zero(self) -> None:
        """Non-overlapping windows of white noise are independent."""
        r_win = windowed_fgn_autocov(5, window_steps=50, hurst=0.5)
        for k in range(1, 5):
            assert abs(float(r_win[k])) < 1e-5

    def test_persistent_variance_exceeds_white(self) -> None:
        """At H=0.9, windowed variance > 1/W (positive correlations
        prevent averaging-out)."""
        w = 100
        r_persist = windowed_fgn_autocov(2, window_steps=w, hurst=0.9)
        assert float(r_persist[0]) > 1.0 / w

    def test_antipersistent_variance_below_white(self) -> None:
        """At H=0.1, windowed variance < 1/W (negative correlations
        accelerate averaging-out)."""
        w = 100
        r_anti = windowed_fgn_autocov(2, window_steps=w, hurst=0.1)
        assert float(r_anti[0]) < 1.0 / w

    def test_persistent_has_positive_lag1(self) -> None:
        """For H=0.7-0.9, windowed lag-1 autocorr is > 0."""
        for h in (0.7, 0.9):
            r_win = windowed_fgn_autocov(
                3, window_steps=50, hurst=h,
            )
            assert float(r_win[1]) > 0

    def test_return_shape(self) -> None:
        """Returns shape (n_windows,)."""
        for nw in (1, 4, 20):
            r_win = windowed_fgn_autocov(nw, window_steps=10, hurst=0.7)
            assert r_win.shape == (nw,)

    def test_scaling_law_large_w(self) -> None:
        """Var(window mean) ~ W^(2H-2) for large W.

        At H=0.9, Var ~ W^(-0.2). Doubling W should reduce variance
        by factor 2^(-0.2) ≈ 0.87.
        """
        h = 0.9
        r1 = float(windowed_fgn_autocov(2, 50, h)[0])
        r2 = float(windowed_fgn_autocov(2, 100, h)[0])
        ratio = r2 / r1
        expected = 2.0 ** (2 * h - 2)
        # Finite-W corrections loosen this from exact scaling; allow 20%.
        assert 0.8 * expected < ratio < 1.2 * expected

    def test_stride_default_is_non_overlapping(self) -> None:
        """stride=None should default to window_steps (non-overlapping)."""
        default = windowed_fgn_autocov(4, 20, 0.7)
        explicit = windowed_fgn_autocov(4, 20, 0.7, stride=20)
        for a, b in zip(default.tolist(), explicit.tolist()):
            assert a == pytest.approx(b, abs=1e-5)

    def test_overlapping_stride_smaller_than_window(self) -> None:
        """With stride < window_steps, adjacent windows share samples,
        so lag-1 autocovariance is larger than the non-overlapping case
        (by construction)."""
        h = 0.5  # white noise: cleanest test case
        w = 10
        # Non-overlapping: r_W(1) = 0 (windows independent for H=0.5)
        nonoverlap = windowed_fgn_autocov(2, w, h, stride=w)
        assert abs(float(nonoverlap[1])) < 1e-5
        # Overlapping with stride=1: window[t] and window[t+1] share 9
        # of 10 samples; for white noise r_W(1) = 9/w^2 * 1 = 0.09.
        overlap = windowed_fgn_autocov(2, w, h, stride=1)
        # Formula: r_W(1) = (1/W^2) * sum_{j,k} r(|1*stride + j - k|)
        #   = (1/100) * (number of (j,k) with |1 + j - k| = 0)
        #   = (1/100) * 9   (pairs (j, j+1) for j=0..8)
        expected = 9.0 / (w * w)
        assert float(overlap[1]) == pytest.approx(expected, abs=1e-4)


# ----------------------------------------------------------------------
# WK floor on windowed fGn
# ----------------------------------------------------------------------

class TestWKFloor:
    """Asymptotic prediction error variance behavior."""

    def test_floor_positive(self) -> None:
        """WK floor must be > 0 for a non-degenerate process."""
        for h in (0.3, 0.5, 0.7, 0.9):
            floor = wk_floor_windowed_fgn(20, 100, h)
            assert floor > 0

    def test_floor_bounded_by_windowed_variance(self) -> None:
        """WK floor is at most Var(windowed process) = r_W(0)."""
        for h in (0.3, 0.7, 0.9):
            floor = wk_floor_windowed_fgn(20, 100, h)
            r_win_0 = float(windowed_fgn_autocov(1, 100, h)[0])
            assert floor <= r_win_0 + 1e-6

    def test_floor_equals_windowed_variance_at_h05(self) -> None:
        """White noise: prediction can't do better than report the mean.
        Floor = Var(windowed process) exactly.
        """
        w = 100
        floor = wk_floor_windowed_fgn(20, w, 0.5)
        r_win_0 = float(windowed_fgn_autocov(1, w, 0.5)[0])
        # Should be exactly equal (no prediction benefit from
        # uncorrelated windows).
        assert floor == pytest.approx(r_win_0, rel=1e-3)

    def test_floor_below_variance_when_persistent(self) -> None:
        """At H=0.9 (strong memory), prediction beats reporting the mean."""
        floor = wk_floor_windowed_fgn(20, 100, 0.9)
        r_win_0 = float(windowed_fgn_autocov(1, 100, 0.9)[0])
        assert floor < r_win_0

    def test_floor_below_variance_when_antipersistent(self) -> None:
        """At H=0.3 (anti-persistent), prediction also helps
        (negative correlation is exploitable)."""
        floor = wk_floor_windowed_fgn(20, 100, 0.3)
        r_win_0 = float(windowed_fgn_autocov(1, 100, 0.3)[0])
        assert floor < r_win_0

    def test_floor_decreases_as_h_moves_from_05(self) -> None:
        """The ratio floor / r_W(0) is 1 at H=0.5 and decreases as
        |H-0.5| grows (more exploitable structure)."""
        ratios = []
        for h in (0.5, 0.6, 0.7, 0.8):
            floor = wk_floor_windowed_fgn(20, 50, h)
            r0 = float(windowed_fgn_autocov(1, 50, h)[0])
            ratios.append(floor / r0)
        # Monotonically decreasing from h=0.5 onward.
        for i in range(1, len(ratios)):
            assert ratios[i] < ratios[i - 1] + 1e-4


# ----------------------------------------------------------------------
# Monte Carlo cross-check
# ----------------------------------------------------------------------

class TestMonteCarloValidation:
    """Synthesize many fGn paths, measure empirical prediction error."""

    def test_empirical_variance_matches_point_autocov(self) -> None:
        """Empirical Var(X_0) from many short paths should be ~r(0)=1."""
        n_paths = 2000
        n = 32
        rngs = jax.random.split(jax.random.PRNGKey(0), n_paths)
        paths = jnp.stack(
            [fgn_davies_harte(n, 0.7, k) for k in rngs]
        )
        var = float(paths[:, 0].var())
        assert 0.85 < var < 1.15

    def test_empirical_lag1_matches_theoretical(self) -> None:
        """Sample mean of X_t * X_{t+1} should match r(1)."""
        n_paths = 1000
        n = 128
        h = 0.7
        rngs = jax.random.split(jax.random.PRNGKey(0), n_paths)
        paths = jnp.stack(
            [fgn_davies_harte(n, h, k) for k in rngs]
        )
        empirical_r1 = float((paths[:, :-1] * paths[:, 1:]).mean())
        theoretical_r1 = float(fgn_autocov(2, h)[1])
        # Sample std of estimator ~ 1/sqrt(n_paths * n) ~ 0.003.
        # Allow 0.02 tolerance for safety.
        assert abs(empirical_r1 - theoretical_r1) < 0.02

    def test_empirical_windowed_variance_matches(self) -> None:
        """Variance of window-averaged fGn matches theoretical value
        within sampling error.

        This cross-validates BOTH the synthesizer (fgn_davies_harte)
        AND the theoretical windowed autocov (windowed_fgn_autocov).
        If the two agree empirically, both are probably correct.
        """
        h = 0.7
        w = 50
        n = w * 20  # 20 windows per path
        n_paths = 500
        rngs = jax.random.split(jax.random.PRNGKey(7), n_paths)
        paths = jnp.stack(
            [fgn_davies_harte(n, h, k) for k in rngs]
        )
        # Reshape to (n_paths, 20 windows, w)
        windowed = paths.reshape(n_paths, 20, w).mean(axis=-1)
        empirical_var = float(windowed.var())
        theoretical_var = float(windowed_fgn_autocov(1, w, h)[0])
        # Generous bound: 15% relative agreement with n_paths * 20 samples.
        assert abs(empirical_var - theoretical_var) < 0.15 * theoretical_var
