"""WO-003b Phase B: shared-draws coverage reconciliation + DSR calibration rider.

Shared-draws design (STEP 1): per series, ONE bootstrap resample set
(n_boot = 999, locked count) is generated once. Percentile, bootstrap-SE
normal, and studentized statistics are all derived from that same resample
set. BCa adds only its block-jackknife acceleration (non-overlapping circular
blocks, same b). The HAC delta-method interval is analytic (no resampling).
No method runs its own independent bootstrap.

Coverage reconciliation at n_series=2000:
  configs = {iid, AR(1) phi=0.3} x mu in {0, 0.002}
  metrics = mean, sharpe (true Sharpe on generator's marginal basis)
  levels = 0.80 / 0.90 / 0.95 / 0.99

DSR calibration rider at n_series=20000 (iid null + AR(1) null, N=1).

Run: python -m tools.run_phase_b   (from repo root, editable install)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parapix.bootstrap import _auto_block_length
from parapix.dsr import dsr, expected_max_sharpe, sharpe_stats
from parapix.sim import ar1, iid_noise, injected_edge
from parapix.selfvalidate import (
    binding_fields,
    enforce_clean_tree,
    git_info,
    runner_sha256,
    verify_binding,
)

RESULTS_DIR = "results"
N_OBS = 1000
SIGMA = 0.01
PHI = 0.3
MARGINAL_FACTOR = math.sqrt(1.0 - PHI * PHI)
MASTER = 20261125
N_BOOT = 999  # locked bootstrap count
COVERAGE_LEVELS = (0.80, 0.90, 0.95, 0.99)
Z_SCORE = {level: float(norm.ppf(1.0 - (1.0 - level) / 2.0)) for level in COVERAGE_LEVELS}
METHODS = ("percentile", "bootstrap_se_normal", "studentized", "bca_block", "hac_delta")


# ---------------------------------------------------------------------------
# Vectorized shared-draws block bootstrap
# ---------------------------------------------------------------------------

def _block_bootstrap_resamples(returns, seed, n_boot, block_length):
    """Generate n_boot circular block-bootstrap resamples, shape (n_boot, n).

    Fully vectorised: a single RNG call produces all block-start indices, then
    advanced indexing builds every resample at once. No Python loop over
    bootstrap iterations.
    """
    n = returns.shape[0]
    rng = np.random.default_rng(seed)
    nblocks = (n + block_length - 1) // block_length
    starts = rng.integers(0, n, size=(n_boot, nblocks))
    offsets = np.arange(block_length)[None, None, :]
    expanded = starts[:, :, None] + offsets            # (n_boot, nblocks, b)
    flat = expanded.reshape(n_boot, nblocks * block_length)  # (n_boot, nblocks*b)
    idx = flat[:, :n] % n                              # (n_boot, n)
    return returns[idx]


def _bootstrap_statistics(resamples, metric):
    """Compute bootstrap statistic per resample, plus observed and SEs.

    Returns (boots, obs, se_obs, se_boots) where:
      boots     : (n_boot,) bootstrap statistic values
      obs       : observed statistic on the original data
      se_obs    : observed standard error
      se_boots  : (n_boot,) per-resample standard errors (for studentized)
    """
    n = resamples.shape[1]
    if metric == "mean":
        boots = resamples.mean(axis=1)
        obs = None  # filled in by caller (same data, same mean)
        se_obs = None
        se_boots = resamples.std(axis=1, ddof=1) / math.sqrt(n)
    else:  # sharpe
        res_mean = resamples.mean(axis=1)
        res_sd = resamples.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = res_mean / res_sd
        se_boots = np.sqrt((1.0 + boots**2 / 2.0) / n)
    return boots, se_boots


def _ci_percentile(boots, obs, level):
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    return float(np.percentile(boots, lo_q * 100)), float(np.percentile(boots, hi_q * 100))


def _ci_bootstrap_se_normal(boots, obs, level, se_obs=None):
    boot_se = float(np.std(boots, ddof=1))
    z = Z_SCORE[level]
    return obs - z * boot_se, obs + z * boot_se


def _ci_studentized(boots, obs, se_obs, se_boots, level):
    valid = np.isfinite(se_boots) & (se_boots > 0) & np.isfinite(boots)
    t_vals = (boots[valid] - obs) / se_boots[valid]
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    t_lo = float(np.quantile(t_vals, lo_q))
    t_hi = float(np.quantile(t_vals, hi_q))
    return obs - t_hi * se_obs, obs - t_lo * se_obs


def _ci_bca_block(returns, boots, obs, block_length, metric, level):
    """BCa interval with block-jackknife acceleration (non-overlapping circular blocks)."""
    n = returns.shape[0]
    # Bias correction
    z0 = norm.ppf(np.mean(boots < obs)) if np.any(boots < obs) else 0.0
    if not np.isfinite(z0):
        z0 = 0.0

    # Acceleration from block jackknife (non-overlapping blocks of length b)
    n_blocks = n // block_length
    if n_blocks < 2:
        # Fallback: no acceleration
        a = 0.0
    else:
        if metric == "mean":
            total = returns.sum()
            cumsum = np.concatenate([[0.0], np.cumsum(returns)])
            jack = np.empty(n_blocks)
            for j in range(n_blocks):
                block_sum = cumsum[(j + 1) * block_length] - cumsum[j * block_length]
                jack[j] = (total - block_sum) / (n - block_length)
        else:  # sharpe
            total = returns.sum()
            total_sq = float(returns @ returns)
            cumsum = np.concatenate([[0.0], np.cumsum(returns)])
            cumsum_sq = np.concatenate([[0.0], np.cumsum(returns * returns)])
            jack = np.empty(n_blocks)
            for j in range(n_blocks):
                m_excl = n - block_length
                sum_excl = total - (cumsum[(j + 1) * block_length] - cumsum[j * block_length])
                sq_excl = total_sq - (cumsum_sq[(j + 1) * block_length] - cumsum_sq[j * block_length])
                mean_excl = sum_excl / m_excl
                var_excl = (sq_excl - mean_excl * sum_excl) / (m_excl - 1) if m_excl > 1 else 0.0
                sd_excl = math.sqrt(max(var_excl, 0.0))
                jack[j] = mean_excl / sd_excl if sd_excl > 0 else np.nan
        jack = jack[np.isfinite(jack)]
        if len(jack) >= 2:
            jack_mean = jack.mean()
            num = float(((jack_mean - jack) ** 3).sum())
            den = float(((jack_mean - jack) ** 2).sum() ** 1.5)
            a = num / (6.0 * den) if den > 0 else 0.0
        else:
            a = 0.0

    # Adjusted quantiles
    def _adj_alpha(alpha):
        z_alpha = norm.ppf(alpha)
        val = z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)) if (1.0 - a * (z0 + z_alpha)) != 0 else np.inf
        return float(norm.cdf(val))

    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    alpha_lo = _adj_alpha(lo_q)
    alpha_hi = _adj_alpha(hi_q)
    alpha_lo = min(max(alpha_lo, 0.0), 1.0)
    alpha_hi = min(max(alpha_hi, 0.0), 1.0)
    return float(np.quantile(boots, alpha_lo)), float(np.quantile(boots, alpha_hi))


def _long_run_variance(x, band=None):
    """Newey-West / Bartlett long-run variance of a 1D series."""
    n = x.shape[0]
    if band is None:
        band = int(math.floor(n ** (1.0 / 3.0)))
    band = max(1, min(band, n - 1))
    eps = x - x.mean()
    acv0 = float(eps @ eps) / n
    lr_var = acv0
    for k in range(1, band + 1):
        w = 1.0 - k / (band + 1.0)
        gamma_k = float(eps[k:] @ eps[: n - k]) / n
        lr_var += 2.0 * w * gamma_k
    return max(lr_var, 0.0)


def _ci_hac_delta(returns, obs, metric, level):
    """HAC delta-method interval (analytic, no resampling).

    Mean:  SE = sqrt(LRV / n), CI = mean ± z * SE.
    Sharpe: SE ≈ sqrt(LRV * (1 + SR^2/2) / (n * sigma^2)), CI = SR ± z * SE.
    """
    n = returns.shape[0]
    z = Z_SCORE[level]
    lrv = _long_run_variance(returns)
    if metric == "mean":
        se = math.sqrt(lrv / n) if lrv > 0 else 0.0
    else:
        sigma2 = float(returns.var(ddof=1))
        se = math.sqrt(lrv * (1.0 + obs**2 / 2.0) / (n * sigma2)) if sigma2 > 0 and lrv > 0 else 0.0
    return obs - z * se, obs + z * se


def _compute_ci(method, boots, obs, se_obs, se_boots, returns, block_length, metric, level):
    if method == "percentile":
        return _ci_percentile(boots, obs, level)
    if method == "bootstrap_se_normal":
        return _ci_bootstrap_se_normal(boots, obs, level, se_obs)
    if method == "studentized":
        return _ci_studentized(boots, obs, se_obs, se_boots, level)
    if method == "bca_block":
        return _ci_bca_block(returns, boots, obs, block_length, metric, level)
    if method == "hac_delta":
        return _ci_hac_delta(returns, obs, metric, level)
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# Coverage arm runner
# ---------------------------------------------------------------------------

def _run_coverage_arm(out_dir, metric, kind, mu, true_val, n_series, n_boot,
                      master, block_length_override=None):
    """Run one coverage arm: shared draws, all methods, all levels."""
    t0 = time.time()
    n = N_OBS
    sigma = SIGMA
    phi = PHI

    # Per-series storage: series_idx -> method -> level -> dict
    series_data = {}
    degenerate_count = 0
    total_resamples = 0

    for i in range(n_series):
        s = np.random.SeedSequence(master + 10000 + i).spawn(1)[0]
        seed = int(s.generate_state(1)[0])

        if kind == "iid":
            r = injected_edge(n, seed, sigma, mu)
        else:
            base = ar1(n, seed, sigma, phi)
            r = base + mu

        # Shared bootstrap: one resample set for all methods
        if block_length_override is not None:
            bl = int(block_length_override)
        else:
            _, bl = _auto_block_length(r)
        resamples = _block_bootstrap_resamples(r, seed + 1, n_boot, bl)

        boots, se_boots = _bootstrap_statistics(resamples, metric)

        if metric == "mean":
            obs = float(r.mean())
            se_obs = float(r.std(ddof=1) / math.sqrt(n))
        else:
            r_sd = float(r.std(ddof=1))
            obs = float(r.mean() / r_sd) if r_sd > 0 else float("nan")
            se_obs = math.sqrt((1.0 + obs**2 / 2.0) / n)

        # Degenerate resample tracking
        deg_mask = ~np.isfinite(boots) | (se_boots <= 0) | ~np.isfinite(se_boots)
        degenerate_count += int(np.sum(deg_mask))
        total_resamples += n_boot

        # Compute CIs for every method and level from the shared boots
        arm_series = {}
        for method in METHODS:
            level_data = {}
            for level in COVERAGE_LEVELS:
                lo, hi = _compute_ci(method, boots, obs, se_obs, se_boots, r, bl, metric, level)
                covers = bool(lo <= true_val <= hi) if np.isfinite(lo) and np.isfinite(hi) else False
                width = float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else float("nan")
                level_data[level] = {"covers": covers, "width": width, "lo": lo, "hi": hi}
            arm_series[method] = level_data
        series_data[i] = arm_series

    # Aggregate across series
    arm_result = {"metric": metric, "kind": kind, "mu": mu, "true": true_val,
                  "n_series": n_series, "n_obs": n, "sigma": sigma, "phi": phi,
                  "n_boot": n_boot, "block_length": bl, "methods": list(METHODS),
                  "coverage_levels": list(COVERAGE_LEVELS),
                  "degenerate_resample_count": degenerate_count,
                  "total_resamples": total_resamples,
                  "studentized_degenerate_fraction": degenerate_count / total_resamples if total_resamples > 0 else float("nan"),
                  "wall_clock_seconds": time.time() - t0,
                  "per_method": {}}

    for method in METHODS:
        method_data = {"coverage": {}, "mean_width": {}, "series_covers": {}}
        for level in COVERAGE_LEVELS:
            covers = [series_data[i][method][level]["covers"] for i in range(n_series)]
            widths = [series_data[i][method][level]["width"] for i in range(n_series)]
            method_data["coverage"][str(level)] = sum(covers) / n_series
            method_data["mean_width"][str(level)] = float(np.mean([w for w in widths if np.isfinite(w)]))
            method_data["series_covers"][str(i)] = covers  # placeholder, stored below per series
        # Store series-level covers for McNemar
        method_data["series_covers"] = {str(i): {str(level): series_data[i][method][level]["covers"]
                                                 for level in COVERAGE_LEVELS} for i in range(n_series)}
        arm_result["per_method"][method] = method_data

    # McNemar discordant counts for paired method comparisons
    mcnemar = {}
    for m1_i in range(len(METHODS)):
        for m2_i in range(m1_i + 1, len(METHODS)):
            m1, m2 = METHODS[m1_i], METHODS[m2_i]
            pair_key = f"{m1}_vs_{m2}"
            discord = {}
            for level in COVERAGE_LEVELS:
                b_count = 0  # m1 covers, m2 doesn't
                c_count = 0  # m1 doesn't, m2 covers
                for i in range(n_series):
                    c1 = series_data[i][m1][level]["covers"]
                    c2 = series_data[i][m2][level]["covers"]
                    if c1 and not c2:
                        b_count += 1
                    elif not c1 and c2:
                        c_count += 1
                discord[str(level)] = {"b_m1_cover_m2_no": b_count, "c_m1_no_m2_cover": c_count}
            mcnemar[pair_key] = discord
    arm_result["mcnemar"] = mcnemar

    # Per-arm summary print
    print(f"[coverage/{metric}/{kind}_mu{mu}] wall={arm_result['wall_clock_seconds']:.1f}s "
          f"deg_frac={arm_result['studentized_degenerate_fraction']:.4f}")
    for method in METHODS:
        cov_str = ", ".join(f"{level}={arm_result['per_method'][method]['coverage'][str(level)]:.4f}"
                            for level in COVERAGE_LEVELS)
        print(f"  {method}: {cov_str}")
    _ckpt(out_dir, f"coverage_{metric}_{kind}_mu{int(mu*1000) if mu else 0}", arm_result)
    return arm_result


# ---------------------------------------------------------------------------
# DSR calibration rider
# ---------------------------------------------------------------------------

def _dsr_for_series(r, n_trials, trial_sr_std, rule):
    stats = sharpe_stats(r)
    sr = stats["sr"]
    skew = stats["skew"]
    kurt = stats["kurtosis"]
    n_obs = stats["n_obs"]
    sr0 = 0.0 if n_trials == 1 else expected_max_sharpe(n_trials, trial_sr_std)
    d = dsr(sr, n_obs, skew, kurt, n_trials, trial_sr_std)
    return {
        "sr": sr, "skew": skew, "kurtosis": kurt, "n_obs": n_obs,
        "sr0": sr0, "dsr": d, "trial_sr_std": trial_sr_std,
        "trial_sr_rule": rule,
        "corrected_p": (1.0 - d) if not math.isnan(d) else float("nan"),
    }


def run_dsr_calibration_n20000(out_dir, n_series=20000, master=MASTER):
    """DSR calibration rider at n_series=20000 (iid null + AR(1) null, N=1)."""
    t0 = time.time()
    se = 1.0 / math.sqrt(N_OBS - 1)

    rng = np.random.default_rng(master)
    count_iid = 0
    dsrs_iid = []
    for i in range(n_series):
        r = iid_noise(N_OBS, int(rng.integers(0, 2**31)), SIGMA)
        res = _dsr_for_series(r, n_trials=1, trial_sr_std=se, rule="null_theoretical")
        dsrs_iid.append(res["dsr"])
        if res["dsr"] > 0.95:
            count_iid += 1
    frac_iid = count_iid / n_series

    rng2 = np.random.default_rng(master + 1)
    count_ar1 = 0
    dsrs_ar1 = []
    for i in range(n_series):
        r = ar1(N_OBS, int(rng2.integers(0, 2**31)), SIGMA, PHI)
        res = _dsr_for_series(r, n_trials=1, trial_sr_std=se, rule="null_theoretical")
        dsrs_ar1.append(res["dsr"])
        if res["dsr"] > 0.95:
            count_ar1 += 1
    frac_ar1 = count_ar1 / n_series

    wall = time.time() - t0
    result_iid = {
        "arm": "dsr_calib_iid_null_N1_n20000",
        "n_series": n_series, "n_obs": N_OBS, "sigma": SIGMA, "phi": PHI,
        "n_trials": 1, "trial_sr_std": se, "trial_sr_rule": "null_theoretical",
        "fraction_dsr_gt_095": frac_iid, "prediction": 0.05,
        "wall_clock_seconds": wall / 2.0,
    }
    result_ar1 = {
        "arm": "dsr_calib_ar1_null_N1_n20000",
        "n_series": n_series, "n_obs": N_OBS, "sigma": SIGMA, "phi": PHI,
        "n_trials": 1, "trial_sr_std": se, "trial_sr_rule": "null_theoretical",
        "fraction_dsr_gt_095": frac_ar1, "prediction": 0.114,
        "note": "SE ratio 1.363 from variance inflation 1.857 at AR(1) phi=0.3",
        "wall_clock_seconds": wall / 2.0,
    }
    _ckpt(out_dir, "dsr_calib_iid_null_N1_n20000", result_iid)
    _ckpt(out_dir, "dsr_calib_ar1_null_N1_n20000", result_ar1)
    print(f"[dsr/iid/n={n_series}] frac DSR>0.95 = {frac_iid:.5f} (prediction 0.05)")
    print(f"[dsr/ar1/n={n_series}] frac DSR>0.95 = {frac_ar1:.5f} (prediction ~0.114)")
    print(f"[dsr] wall_clock={wall:.1f}s")
    return result_iid, result_ar1


# ---------------------------------------------------------------------------
# Checkpoint / binding
# ---------------------------------------------------------------------------

def _ckpt(out_dir, name, obj):
    path = os.path.join(out_dir, f"checkpoint_phaseb_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2)
    return path


def run_phase_b(out_dir=RESULTS_DIR, repo=".", n_series=2000, n_boot=N_BOOT,
                master=MASTER, pilot=False):
    """Run Phase B: coverage reconciliation + DSR calibration rider.

    If pilot=True, runs only 20 series per arm for timing projection.
    """
    start_time = datetime.now(timezone.utc)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # Binding at run start — tree must be clean (dirty=false)
    start_sha, start_dirty, start_untracked = git_info(repo)
    if start_dirty or start_untracked:
        raise RuntimeError(
            f"Tree is not clean: dirty={start_dirty}, untracked={start_untracked}. "
            "Commit the runner first (STEP 1), then run."
        )
    start_runner_hash = runner_sha256(repo)

    # Define arms
    arms = []
    for kind in ["iid", "ar1"]:
        for mu in [0.0, 0.002]:
            true_mean = mu
            if kind == "iid":
                true_sharpe = mu / SIGMA
            else:
                true_sharpe = mu * MARGINAL_FACTOR / SIGMA
            arms.append(("mean", kind, mu, true_mean))
            arms.append(("sharpe", kind, mu, true_sharpe))

    n_series_run = 20 if pilot else n_series
    coverage_results = {}
    for metric, kind, mu, true_val in arms:
        arm_name = f"{metric}_{kind}_mu{mu}"
        res = _run_coverage_arm(out_dir, metric, kind, mu, true_val,
                                n_series_run, n_boot, master)
        coverage_results[arm_name] = res

    dsr_iid, dsr_ar1 = run_dsr_calibration_n20000(out_dir, n_series=20000 if not pilot else 200,
                                                  master=master)

    end_time = datetime.now(timezone.utc)
    wall_clock = time.time() - t0

    # Binding integrity re-check
    end_sha, end_dirty, end_untracked = git_info(repo)
    if end_sha != start_sha:
        raise RuntimeError(f"HEAD changed during run: {start_sha} -> {end_sha}")
    if end_dirty != start_dirty or end_untracked != start_untracked:
        raise RuntimeError("Tree cleanliness changed during run")

    # Pilot projection
    projection = None
    if pilot:
        # Project full run from pilot: full covers n_series arms, pilot covers 20
        # Coverage arms: 8 arms, each at n_series (full) vs 20 (pilot)
        # DSR: full is 20000, pilot is 200
        coverage_ratio = n_series / n_series_run
        dsr_ratio = 20000 / 200
        # Estimate: coverage time scales linearly, DSR time scales linearly
        coverage_time = wall_clock * 0.8  # roughly 80% of pilot time is coverage
        dsr_time = wall_clock * 0.2
        projected = coverage_time * coverage_ratio + dsr_time * dsr_ratio
        projection = {
            "pilot_wall_clock_seconds": wall_clock,
            "coverage_ratio": coverage_ratio,
            "dsr_ratio": dsr_ratio,
            "projected_full_wall_clock_seconds": projected,
            "projected_full_hours": projected / 3600.0,
        }
        print(f"[pilot] projected full run: {projected:.1f}s = {projected/3600:.2f} hours")

    final = {
        "git_sha": start_sha,
        "dirty": start_dirty,
        "untracked_files": start_untracked,
        "runner_sha256": start_runner_hash,
        "command_line": " ".join(sys.argv),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "pip_freeze_hash": hashlib.sha256(
            subprocess.run([sys.executable, "-m", "pip", "freeze"],
                           capture_output=True, text=True).stdout.encode("utf-8")
        ).hexdigest(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "start_time_utc": start_time.strftime("%Y%m%dT%H%M%SZ"),
        "end_time_utc": end_time.strftime("%Y%m%dT%H%M%SZ"),
        "wall_clock_seconds": wall_clock,
        "n_series": n_series,
        "n_obs": N_OBS,
        "sigma": SIGMA,
        "phi": PHI,
        "marginal_factor": MARGINAL_FACTOR,
        "n_boot": n_boot,
        "coverage_levels": list(COVERAGE_LEVELS),
        "methods": list(METHODS),
        "master_seed": master,
        "binding": binding_fields(repo, extra={
            "start_sha": start_sha,
            "start_dirty": start_dirty,
            "start_untracked": start_untracked,
        }),
        "binding_status": "BOUND",
        "arms": coverage_results,
        "dsr_rider": {
            "iid": dsr_iid,
            "ar1": dsr_ar1,
        },
    }
    if projection:
        final["pilot_projection"] = projection

    ts = end_time.strftime("%Y%m%dT%H%M%SZ")
    suffix = "_pilot" if pilot else ""
    path = os.path.join(out_dir, f"selfvalidation_phaseb{suffix}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final, f, sort_keys=True, indent=2)

    print(f"\nArtifact: {path}")
    print(f"Binding: BOUND (dirty={start_dirty})")
    print(f"git_sha: {final['git_sha']}")
    print(f"Wall clock: {wall_clock:.1f}s")

    ok, reason = verify_binding(final, repo)
    print(f"Binding verification: {'PASS' if ok else 'FAIL'} - {reason}")

    return final, path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WO-003b Phase B run")
    parser.add_argument("--pilot", action="store_true",
                        help="Run pilot with 20 series per arm for timing projection")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    repo = "."
    enforce_clean_tree(repo=repo)
    run_phase_b(out_dir=args.out_dir, repo=repo, pilot=args.pilot)


if __name__ == "__main__":
    main()