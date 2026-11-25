"""Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR).

Sources
-------
PSR: Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier" (2012),
     Eq. (11), https://davidhbailey.com/dhbpapers/sharpe-frontier.pdf
DSR: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014),
     Eq. (1)-(2), https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

All Sharpe ratios are per-period, not annualised. γ₄ is raw (non-excess)
kurtosis: 3 for a Normal distribution. The standard error (SE) is the Wald
form evaluated at the estimated SR.

Assumptions
-----------
The analytic SE assumes iid returns. Under autocorrelation it understates the
true SE: at AR(1) φ=0.3 the variance inflation is ~1.857×, so the SE is
understated by ~1.36× (first order). DSR against SR₀ is therefore not a
size-calibrated FWER test under dependence; the reported over-rejection is
the design, not miscalibration.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, skew, kurtosis

GAMMA_E = 0.5772156649015329  # Euler-Mascheroni constant


def sharpe_stats(returns):
    """Per-period Sharpe ratio and return moments.

    Parameters
    ----------
    returns : array-like
        Per-period returns.

    Returns
    -------
    dict with keys ``sr`` (mean/std, ddof=1), ``skew`` (sample skewness,
    bias-corrected), ``kurtosis`` (raw/non-excess, bias-corrected; 3 for
    Normal), ``n_obs`` (number of observations).

    Notes
    -----
    Moment estimators: std uses ddof=1 (Bessel's correction). Skewness and
    kurtosis use scipy.stats with bias=False and fisher=False, so kurtosis is
    raw (not excess): Normal returns give kurtosis ≈ 3. The DSR paper assumes
    population-style moments but does not pin ddof/bias conventions; this is
    the code's explicit choice.
    """
    x = np.asarray(returns, dtype=float)
    n = x.shape[0]
    if n < 3:
        return {"sr": float("nan"), "skew": float("nan"), "kurtosis": float("nan"), "n_obs": n}
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return {"sr": float("nan"), "skew": float("nan"), "kurtosis": float("nan"), "n_obs": n}
    sr = float(np.mean(x)) / sd
    g3 = float(skew(x, bias=False))
    g4 = float(kurtosis(x, fisher=False, bias=False))
    return {"sr": sr, "skew": g3, "kurtosis": g4, "n_obs": n}


def psr(sr, sr_benchmark, n_obs, skew, kurtosis):
    """Probabilistic Sharpe Ratio at benchmark SR*.

    P(true SR > SR*) = Φ((sr − sr_benchmark)√(n−1) / √(1 − γ₃·sr + ((γ₄−1)/4)·sr²))

    The SE is the Wald form evaluated at the estimated ``sr``, not at
    ``sr_benchmark``.

    Returns NaN if n_obs < 2, the denominator is non-positive, or any input
    is NaN.
    """
    if n_obs < 2:
        return float("nan")
    den_sq = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if not math.isfinite(den_sq) or den_sq <= 0.0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(den_sq)
    if not math.isfinite(z):
        return float("nan")
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials, trial_sr_std):
    """Expected maximum Sharpe of N independent zero-skill trials.

    SR₀ = √V · ((1 − γ_E)·Φ⁻¹(1 − 1/N) + γ_E·Φ⁻¹(1 − 1/(N·e)))

    Parameters
    ----------
    n_trials : int
        Number of independent trials. N=1 returns exactly 0.0 by special case
        (Φ⁻¹(0) = −∞ would otherwise break the formula).
    trial_sr_std : float
        Standard deviation of the per-trial Sharpe estimates (√V).

    Returns
    -------
    float
        Expected maximum Sharpe in per-period units. 0.0 when n_trials == 1.
    """
    if n_trials == 1:
        return 0.0
    if n_trials < 1:
        return float("nan")
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(trial_sr_std * ((1.0 - GAMMA_E) * z1 + GAMMA_E * z2))


def dsr(sr, n_obs, skew, kurtosis, n_trials, trial_sr_std):
    """Deflated Sharpe Ratio: PSR evaluated at SR* = SR₀.

    DSR = Φ((sr − SR₀)√(n−1) / √(1 − γ₃·sr + ((γ₄−1)/4)·sr²))

    Returns a float in [0, 1], or NaN if the denominator guard triggers
    (1 − γ₃·SR + ((γ₄−1)/4)·SR² ≤ 0), n_trials < 1, or any input is NaN.
    Does not raise; does not clamp.
    """
    if n_trials < 1:
        return float("nan")
    sr0 = expected_max_sharpe(n_trials, trial_sr_std)
    if math.isnan(sr0):
        return float("nan")
    return psr(sr, sr0, n_obs, skew, kurtosis)
