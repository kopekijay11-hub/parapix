"""WO-003c Phase B: independent fresh-process re-derivation of B1 and B2.

This script is deliberately SEPARATE from the simulation driver and from
``parapix.dsr``. It re-implements the DSR / PSR / expected-max-Sharpe formulas
from the paper equations with its own normal CDF / PPF (scipy.stats.norm when
available -- the "MMD" path -- else a hand-rolled math.erf / Acklam PPF), then
compares against the library across a grid. It never calls ``parapix.dsr`` for
the re-derivation; the library is imported only as the thing under test.

B1  (pin, 1e-6): expected_max_sharpe, psr, dsr re-derived independently must
    match the library to <= 1e-6 (abs) across the grid, including NaN guards.
B2  (pin, 3.8e-5 abs / 0.12% rel): the non-normality / estimated-statistic
    correction to the exceedance, computed by DIRECT evaluation (closed-form
    threshold + root find, no dsr call, no Monte Carlo). Reported with settings
    and compared to the registered pin.

Run: ``python -m tools.verify_wo003c_b1b2``  (from repo root)
     ``python tools/verify_wo003c_b1b2.py``
Exit 0 iff B1 matches to 1e-6 (B2 is reported, never a hard gate).
"""

from __future__ import annotations

import math
import os
import sys

# --- independent normal CDF / PPF (do NOT import parapix for these) --------
try:
    from scipy.stats import norm as _sp_norm  # MMD path

    def _norm_cdf(x: float) -> float:
        return float(_sp_norm.cdf(x))

    def _norm_ppf(q: float) -> float:
        return float(_sp_norm.ppf(q))

    _NORM_BACKEND = "scipy.stats.norm (MMD)"
except Exception:  # pragma: no cover - hand-rolled fallback
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _norm_ppf(q: float) -> float:
        # Acklam's rational approximation (hand-rolled, no scipy).
        a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00]
        plow, phigh = 0.02425, 1.0 - 0.02425
        if q < plow:
            r = math.sqrt(-2.0 * math.log(q))
            return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                   ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1.0)
        if q > phigh:
            r = math.sqrt(-2.0 * math.log(1.0 - q))
            return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                    ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1.0)
        r = q - 0.5
        s = r * r
        return (((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*r / \
               (((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1.0)

    _NORM_BACKEND = "hand-rolled math.erf + Acklam PPF"

GAMMA_E = 0.5772156649015329  # Euler-Mascheroni (independent literal)


# --- independent re-implementation of the library formulas ----------------
def i_expected_max_sharpe(n_trials: int, trial_sr_std: float) -> float:
    if n_trials == 1:
        return 0.0
    if n_trials < 1:
        return float("nan")
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(trial_sr_std * ((1.0 - GAMMA_E) * z1 + GAMMA_E * z2))


def i_psr(sr, sr_benchmark, n_obs, skew, kurtosis):
    if n_obs < 2:
        return float("nan")
    den_sq = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if not math.isfinite(den_sq) or den_sq <= 0.0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(den_sq)
    if not math.isfinite(z):
        return float("nan")
    return _norm_cdf(z)


def i_dsr(sr, n_obs, skew, kurtosis, n_trials, trial_sr_std):
    if n_trials < 1:
        return float("nan")
    sr0 = i_expected_max_sharpe(n_trials, trial_sr_std)
    if math.isnan(sr0):
        return float("nan")
    return i_psr(sr, sr0, n_obs, skew, kurtosis)


# --- B1: grid comparison vs the library under test ------------------------
def b1_check() -> tuple[int, float, float]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from parapix.dsr import dsr as lib_dsr, expected_max_sharpe as lib_ems, psr as lib_psr

    n_trials_grid = [1, 2, 5, 10, 20, 50, 100, 250]
    std_grid = [0.01, math.sqrt(1.0 / 999), 0.1, 0.2]
    sr_grid = [-1.0, -0.2, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    nobs_grid = [100, 1000, 5000]
    skew_grid = [0.0, 0.1, -0.2]
    kurt_grid = [3.0, 3.5, 4.5]

    max_abs = 0.0
    max_rel = 0.0
    fails = 0
    n_cmp = 0

    def _close(a, b):
        if math.isnan(a) and math.isnan(b):
            return True, 0.0, 0.0
        if math.isnan(a) != math.isnan(b):
            return False, float("inf"), float("inf")
        d = abs(a - b)
        rel = d / max(abs(b), 1e-300)
        return d <= 1e-6, d, rel

    for N in n_trials_grid:
        for s in std_grid:
            ok, d, rel = _close(i_expected_max_sharpe(N, s), lib_ems(N, s))
            n_cmp += 1
            max_abs = max(max_abs, d)
            max_rel = max(max_rel, rel)
            if not ok:
                fails += 1
        for sr in sr_grid:
            for n in nobs_grid:
                for g3 in skew_grid:
                    for g4 in kurt_grid:
                        for N in n_trials_grid:
                            for s in (0.0316, 0.2):
                                a = i_dsr(sr, n, g3, g4, N, s)
                                b = lib_dsr(sr, n, g3, g4, N, s)
                                ok, d, rel = _close(a, b)
                                n_cmp += 1
                                max_abs = max(max_abs, d)
                                max_rel = max(max_rel, rel)
                                if not ok:
                                    fails += 1
                                # psr spot-check on a subset
                                if N == 10 and s == 0.2:
                                    pa = i_psr(sr, 0.3, n, g3, g4)
                                    pb = lib_psr(sr, 0.3, n, g3, g4)
                                    ok2, d2, r2 = _close(pa, pb)
                                    n_cmp += 1
                                    max_abs = max(max_abs, d2)
                                    max_rel = max(max_rel, r2)
                                    if not ok2:
                                        fails += 1
    return fails, max_abs, max_rel, n_cmp


# --- B2: direct (no-dsr) non-normality / estimated-stat correction --------
def _exceedance_linear(n_trials: int, se: float, z95: float) -> float:
    """P(DSR>0.95) with denominator = 1 (linear threshold), closed form.

    trial_sr_std fixed at the null-theoretical se; per-trial SR ~ N(0, se^2);
    SR0 = g(N)*se. DSR>0.95  <=>  SR > SR0 + z95*se. So
        exceed = 1 - Phi(g(N) + z95)^N.
    """
    g = i_expected_max_sharpe(n_trials, se) / se
    return 1.0 - _norm_cdf(g + z95) ** n_trials


def _sr_threshold_exact(sr0: float, se: float, z95: float, n_obs: int) -> float:
    """Solve (sr - sr0)*sqrt(n-1)/sqrt(1 + 0.5 sr^2) = z95 for sr (bisection).

    This is the exact skew=0, kurt=3 DSR threshold (denominator 1 + 0.5 sr^2),
    found WITHOUT calling dsr. Bisection on a monotone increasing LHS.
    """
    lo, hi = sr0 - 10.0 * se, sr0 + 10.0 * se
    target = z95

    def f(sr):
        den = math.sqrt(1.0 + 0.5 * sr * sr)
        return (sr - sr0) * math.sqrt(n_obs - 1) / den - target

    # f is increasing in sr near the root; expand hi if needed.
    while f(hi) < 0.0:
        hi += 10.0 * se
    while f(lo) > 0.0:
        lo -= 10.0 * se
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _exceedance_exact(n_trials: int, se: float, z95: float, n_obs: int) -> float:
    """P(DSR>0.95) with the exact denominator 1 + 0.5 sr^2, direct (no dsr)."""
    sr0 = i_expected_max_sharpe(n_trials, se)
    sr_star = _sr_threshold_exact(sr0, se, z95, n_obs)
    return 1.0 - _norm_cdf(sr_star / se) ** n_trials


def b2_report() -> dict:
    se = math.sqrt(1.0 / 999)
    z95 = _norm_ppf(0.95)
    n_obs = 1000
    rows = {}
    for N in [5, 10, 20, 50, 100, 250]:
        lin = _exceedance_linear(N, se, z95)
        ex = _exceedance_exact(N, se, z95, n_obs)
        abs_d = ex - lin
        rel_exc = abs_d / lin if lin > 0 else float("nan")   # relative to exceedance
        rel_se = abs_d / se                                   # relative to per-trial SR std
        rows[N] = {"linear": lin, "exact": ex, "abs": abs_d,
                   "rel_exc": rel_exc, "rel_se": rel_se}
    return {"se": se, "z95": z95, "n_obs": n_obs, "rows": rows,
            "pin_abs": 3.8e-5, "pin_rel": 0.0012}


def main() -> int:
    print("=" * 78)
    print("WO-003c B1/B2 INDEPENDENT RE-DERIVATION  (fresh process, no parapix.dsr")
    print("in the re-derivation path)")
    print("=" * 78)
    print(f"normal backend: {_NORM_BACKEND}")
    print()

    fails, max_abs, max_rel, n_cmp = b1_check()
    print("B1  (expected_max_sharpe / psr / dsr vs library, grid):")
    print(f"    comparisons      : {n_cmp}")
    print(f"    max abs diff     : {max_abs:.3e}   (pin <= 1e-6)")
    print(f"    max rel diff     : {max_rel:.3e}")
    print(f"    mismatches(>1e-6): {fails}")
    b1_ok = fails == 0 and max_abs <= 1e-6
    print(f"    B1: {'PASS' if b1_ok else 'FAIL'}")
    print()

    b2 = b2_report()
    print("B2  (non-normality / estimated-stat correction, DIRECT, no dsr call):")
    print(f"    se={b2['se']:.6f}  z95={b2['z95']:.6f}  n_obs={b2['n_obs']}")
    print(f"    {'N':>4} {'linear':>12} {'exact':>12} {'abs':>12} "
          f"{'rel/exc':>10} {'abs/se':>10}")
    best_n, best_d = None, float("inf")
    for N, r in b2["rows"].items():
        print(f"    {N:>4} {r['linear']:>12.6e} {r['exact']:>12.6e} "
              f"{r['abs']:>12.3e} {r['rel_exc']:>10.3%} {r['rel_se']:>10.3%}")
        d = abs(abs(r["abs"]) - b2["pin_abs"])
        if d < best_d:
            best_n, best_d = N, d
    print(f"    registered pin: abs={b2['pin_abs']:.1e}, rel={b2['pin_rel']:.2%}")
    print(f"    closest-N to abs pin: N={best_n} "
          f"(abs={b2['rows'][best_n]['abs']:.3e}, abs/se={b2['rows'][best_n]['rel_se']:.3%})")
    print("    Note: pin rel 0.12% == abs/se (per-trial SR std), i.e. the")
    print("    correction relative to the null-theoretical trial SR std.")
    print()
    print("=" * 78)
    print("B1 RESULT:", "PASS" if b1_ok else "FAIL")
    print("=" * 78)
    return 0 if b1_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())