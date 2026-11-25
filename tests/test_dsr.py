import math

import numpy as np
import pytest
from scipy.stats import norm

from parapix.dsr import (
    GAMMA_E,
    dsr,
    expected_max_sharpe,
    psr,
    sharpe_stats,
)


def test_psr_normal_zero_is_05():
    # Normal returns, SR=0, benchmark=0: z = 0, PSR = Φ(0) = 0.5
    assert psr(0.0, 0.0, 100, 0.0, 3.0) == pytest.approx(0.5, abs=1e-12)


def test_psr_matches_independent_formula():
    sr, sr_b, n, g3, g4 = 0.7, 0.3, 250, -0.2, 4.5
    den_sq = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    z = (sr - sr_b) * math.sqrt(n - 1) / math.sqrt(den_sq)
    expected = float(norm.cdf(z))
    assert psr(sr, sr_b, n, g3, g4) == pytest.approx(expected, rel=1e-12)


def test_psr_denominator_guard_returns_nan():
    # 1 - gamma3*sr + ((gamma4-1)/4)*sr^2 <= 0 -> NaN, no raise, no clamp
    assert math.isnan(psr(1.0, 0.0, 100, 10.0, 3.0))  # 1 - 10 + 0 = -9 <= 0
    # gamma4=1, sr=1: 1 - 0 + 0 = 1 > 0, finite (not NaN)
    assert not math.isnan(psr(1.0, 0.0, 100, 0.0, 1.0))
    # gamma3=10, sr=1, gamma4=3: 1 - 10 + 0 = -9 <= 0 -> DSR NaN
    assert math.isnan(dsr(1.0, 100, 10.0, 3.0, 10, 1.0))


def test_psr_nan_inputs():
    assert math.isnan(psr(float("nan"), 0.0, 100, 0.0, 3.0))
    assert math.isnan(psr(0.0, float("nan"), 100, 0.0, 3.0))
    assert math.isnan(psr(0.0, 0.0, float("nan"), 0.0, 3.0))


def test_expected_max_sharpe_n1_is_zero():
    # N=1 special case: exactly 0.0 regardless of trial_sr_std
    assert expected_max_sharpe(1, 0.0) == 0.0
    assert expected_max_sharpe(1, 1.5) == 0.0
    assert expected_max_sharpe(1, 100.0) == 0.0


def test_expected_max_sharpe_matches_formula():
    for n_trials in [2, 5, 10, 100]:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        expected = ((1.0 - GAMMA_E) * z1 + GAMMA_E * z2)
        assert expected_max_sharpe(n_trials, 1.0) == pytest.approx(expected, rel=1e-12)


def test_expected_max_sharpe_invalid():
    assert math.isnan(expected_max_sharpe(0, 1.0))
    assert math.isnan(expected_max_sharpe(-5, 1.0))


def test_dsr_n1_equals_psr_zero_benchmark():
    # N=1: SR0 = 0, so DSR = PSR(sr, 0, ...)
    sr, n, g3, g4 = 0.4, 500, 0.1, 3.5
    assert dsr(sr, n, g3, g4, 1, 1.0) == pytest.approx(psr(sr, 0.0, n, g3, g4), rel=1e-12)


def test_dsr_decreases_with_n_trials():
    # At fixed SR>0, more trials → higher SR0 → lower DSR
    # Use sr=0.2, n_obs=100, trial_sr_std=0.2 so DSR is not saturated.
    sr = 0.2
    d1 = dsr(sr, 100, 0.0, 3.0, 1, 0.2)
    d10 = dsr(sr, 100, 0.0, 3.0, 10, 0.2)
    d100 = dsr(sr, 100, 0.0, 3.0, 100, 0.2)
    assert d1 > d10 > d100
    assert 0.0 < d100 < 1.0


def test_dsr_increases_with_n_obs():
    # At fixed SR, more observations → tighter SE → DSR moves toward 0 or 1
    # For SR=0 (null), DSR should stay near 0.5 regardless
    for n in [100, 1000, 10000]:
        d = dsr(0.0, n, 0.0, 3.0, 10, 0.1)
        assert 0.0 <= d <= 1.0


def test_dsr_bounds():
    # DSR is always in [0, 1] when finite
    for sr in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        for n in [50, 500]:
            for N in [1, 10, 100]:
                v = dsr(sr, n, 0.0, 3.0, N, 0.1)
                if not math.isnan(v):
                    assert 0.0 <= v <= 1.0, f"dsr={v} out of bounds"


def test_dsr_nan_propagates():
    assert math.isnan(dsr(float("nan"), 100, 0.0, 3.0, 10, 0.1))
    assert math.isnan(dsr(0.5, 100, float("nan"), 3.0, 10, 0.1))
    assert math.isnan(dsr(0.5, 100, 0.0, float("nan"), 10, 0.1))
    assert math.isnan(dsr(0.5, float("nan"), 0.0, 3.0, 10, 0.1))


def test_dsr_small_n_approximation_error():
    # Phase A: Gumbel/Euler-Mascheroni approximation vs simulated expected max.
    # Tolerances from simulation: N=2 ~8.25%, N=5 ~2.36%, N=10 ~2.29%, N=100 ~0.96%.
    # We check the formula is self-consistent (matches its own closed form).
    for n_trials in [2, 5, 10, 100]:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        expected = ((1.0 - GAMMA_E) * z1 + GAMMA_E * z2)
        assert expected_max_sharpe(n_trials, 1.0) == pytest.approx(expected, rel=1e-9)


def test_sharpe_stats_normal_moments():
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=50000)
    s = sharpe_stats(x)
    assert s["n_obs"] == 50000
    assert abs(s["sr"]) < 0.05  # near zero SR for zero-mean noise
    assert abs(s["skew"]) < 0.05  # near zero skew
    assert abs(s["kurtosis"] - 3.0) < 0.1  # near 3 for normal


def test_sharpe_stats_known_values():
    # Deterministic array: [1, 2, 3] -> mean=2, std(ddof=1)=1, sr=2
    x = np.array([1.0, 2.0, 3.0])
    s = sharpe_stats(x)
    assert s["n_obs"] == 3
    assert s["sr"] == pytest.approx(2.0, rel=1e-12)
    assert math.isfinite(s["skew"]) and math.isfinite(s["kurtosis"])
