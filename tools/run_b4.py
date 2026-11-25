"""WO-003 B4: Self-validation v3 bound run.

Runs DSR calibration, selection experiment, shared-factor arm, positive
control, and CI probes. Produces per-arm checkpoints and a final bound
artifact. Requires a clean committed tree (BOUND) unless --unbound.

Run: python -m tools.run_b4   (from repo root, editable install)
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

# Ensure package importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parapix.bootstrap import ci_mean, ci_sharpe, _auto_block_length
from parapix.dsr import dsr, expected_max_sharpe, sharpe_stats, psr
from parapix.evaluate import evaluate_sharpe
from parapix.permtest import permutation_pvalue
from parapix.prereg import PreReg
from parapix.sim import ar1, iid_noise
from parapix.verdict import verdict
from parapix.selfvalidate import (
    binding_fields,
    enforce_clean_tree,
    git_info,
    runner_sha256,
    verify_binding,
)

RESULTS_DIR = "results"
N_OBS = 1000
SE = 1.0 / math.sqrt(N_OBS - 1)  # null-theoretical SE of SR
R_SELECTION = 10_000
N_SERIES_CAL = 500
ALPHA = 0.05
MASTER = 20261125
SIGMA = 0.01


def _ckpt(out_dir, name, obj):
    path = os.path.join(out_dir, f"checkpoint_b4_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2)
    return path


def _dsr_for_series(r, n_trials, trial_sr_std, rule):
    """Compute DSR and related fields for a single series."""
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


def arm_dsr_calibration_iid(out_dir):
    """DSR calibration, iid null, N=1. Prediction: fraction DSR>0.95 ~ 0.05."""
    rng = np.random.default_rng(MASTER)
    count_gt_095 = 0
    dsrs = []
    for i in range(N_SERIES_CAL):
        r = iid_noise(N_OBS, int(rng.integers(0, 2**31)), SIGMA)
        res = _dsr_for_series(r, n_trials=1, trial_sr_std=SE, rule="null_theoretical")
        dsrs.append(res["dsr"])
        if res["dsr"] > 0.95:
            count_gt_095 += 1
    frac = count_gt_095 / N_SERIES_CAL
    result = {
        "arm": "dsr_calib_iid_null_N1",
        "n_series": N_SERIES_CAL,
        "fraction_dsr_gt_095": frac,
        "prediction": 0.05,
        "dsrs": dsrs,
    }
    _ckpt(out_dir, "dsr_calib_iid_null_N1", result)
    print(f"[cal/iid/N=1] frac DSR>0.95 = {frac:.4f} (prediction ~0.05)")
    return result


def arm_dsr_calibration_ar1(out_dir):
    """DSR calibration, AR(1) phi=0.3 null, N=1. Prediction ~0.114."""
    rng = np.random.default_rng(MASTER + 1)
    count_gt_095 = 0
    dsrs = []
    for i in range(N_SERIES_CAL):
        r = ar1(N_OBS, int(rng.integers(0, 2**31)), SIGMA, 0.3)
        res = _dsr_for_series(r, n_trials=1, trial_sr_std=SE, rule="null_theoretical")
        dsrs.append(res["dsr"])
        if res["dsr"] > 0.95:
            count_gt_095 += 1
    frac = count_gt_095 / N_SERIES_CAL
    result = {
        "arm": "dsr_calib_ar1_null_N1",
        "n_series": N_SERIES_CAL,
        "fraction_dsr_gt_095": frac,
        "prediction": 0.114,
        "note": "SE ratio 1.363 from variance inflation 1.857 at AR(1) phi=0.3",
    }
    _ckpt(out_dir, "dsr_calib_ar1_null_N1", result)
    print(f"[cal/ar1/N=1] frac DSR>0.95 = {frac:.4f} (prediction ~0.114)")
    return result


def arm_selection_experiment(out_dir, n_trials, v_rule="sample", n_reps=R_SELECTION):
    """Selection experiment: independent null trials, analytic DSR only.

    Runs one V rule: "sample" (V from sample std across trials) or
    "true_sigma" (V = 1/(n_obs-1)).
    Readouts: mean(SR_max) - SR0 in SE units, median DSR of max,
    fraction with DSR > 0.95 (vs prediction P = 1 - Phi(c)^N).
    """
    rng = np.random.default_rng(MASTER + 100 + n_trials + (0 if v_rule == "sample" else 1000))
    se = SE
    # Prediction P = 1 - Phi(c)^N, c = SR0/SE + 1.645
    results = []
    max_srs = []
    dsrs = []
    count_gt_095 = 0

    for rep in range(n_reps):
        # Generate N independent null series
        series = rng.normal(0.0, SIGMA, size=(n_trials, N_OBS))
        srs = series.mean(axis=1) / series.std(axis=1, ddof=1)
        srs = np.where(np.isfinite(srs), srs, 0.0)

        if v_rule == "sample":
            v = float(np.var(srs, ddof=1)) if n_trials > 1 else 0.0
            trial_sr_std = math.sqrt(v) if v > 0 else 0.0
        else:
            trial_sr_std = se

        sr0 = expected_max_sharpe(n_trials, trial_sr_std)
        idx_max = int(np.argmax(srs))
        sr_max = float(srs[idx_max])

        # DSR for the max trial
        r_max = series[idx_max]
        stats = sharpe_stats(r_max)
        d = dsr(stats["sr"], N_OBS, stats["skew"], stats["kurtosis"], n_trials, trial_sr_std)

        max_srs.append(sr_max)
        dsrs.append(d)
        results.append({
            "rep": rep, "sr_max": sr_max, "sr0": sr0,
            "dsr": d, "trial_sr_std": trial_sr_std,
            "v_rule": v_rule,
            "dsr_gt_095": d > 0.95 if not math.isnan(d) else False,
        })
        if not math.isnan(d) and d > 0.95:
            count_gt_095 += 1

    max_srs = np.array(max_srs)
    dsrs = np.array([x for x in dsrs if not math.isnan(x)])
    sr0_val = expected_max_sharpe(n_trials, trial_sr_std) if v_rule == "sample" else expected_max_sharpe(n_trials, se)

    # mean(SR_max) - SR0 in SE units
    mean_sr_max = float(np.mean(max_srs))
    sr0_at_se = expected_max_sharpe(n_trials, se)
    diff_se_units = (mean_sr_max - sr0_at_se) / se

    # median DSR of max
    median_dsr = float(np.median(dsrs)) if len(dsrs) > 0 else float("nan")

    # fraction with DSR > 0.95
    frac_gt_095 = count_gt_095 / n_reps

    # Prediction P = 1 - Phi(c)^N
    c = sr0_at_se / se + 1.645
    from scipy.stats import norm
    p_pred = 1.0 - norm.cdf(c) ** n_trials

    result = {
        "arm": f"selection_N{n_trials}_{v_rule}",
        "n_trials": n_trials,
        "n_reps": n_reps,
        "v_rule": v_rule,
        "mean_sr_max": mean_sr_max,
        "sr0": sr0_val,
        "diff_se_units": diff_se_units,
        "median_dsr": median_dsr,
        "fraction_dsr_gt_095": frac_gt_095,
        "prediction_p": p_pred,
        "prediction_note": "P = 1 - Phi(c)^N, c = SR0/SE + 1.645",
    }
    _ckpt(out_dir, f"selection_N{n_trials}_{v_rule}", result)
    print(
        f"[sel/N={n_trials}/{v_rule}] "
        f"mean(SR_max)-SR0 in SE = {diff_se_units:+.3f}, "
        f"median DSR = {median_dsr:.3f}, "
        f"frac DSR>0.95 = {frac_gt_095:.4f} (pred P = {p_pred:.4f})"
    )
    return result


def arm_shared_factor(out_dir, n_trials=10, rho=0.3, n_reps=R_SELECTION):
    """Shared-factor arm (secondary): trials share common factor at rho.

    Expect over-deflation (effective N < N). Report same readouts, never pooled
    with independent arm.
    """
    rng = np.random.default_rng(MASTER + 200)
    se = SE
    sqrt_rho = math.sqrt(rho)
    sqrt_1rho = math.sqrt(1.0 - rho)

    results = []
    max_srs = []
    dsrs = []
    count_gt_095 = 0

    for rep in range(n_reps):
        common = rng.normal(0.0, 1.0, size=N_OBS)
        idio = rng.normal(0.0, 1.0, size=(n_trials, N_OBS))
        series = sqrt_rho * common[None, :] + sqrt_1rho * idio
        series = series * SIGMA  # scale to sigma

        srs = series.mean(axis=1) / series.std(axis=1, ddof=1)
        srs = np.where(np.isfinite(srs), srs, 0.0)

        v = float(np.var(srs, ddof=1)) if n_trials > 1 else 0.0
        trial_sr_std = math.sqrt(v) if v > 0 else 0.0
        sr0 = expected_max_sharpe(n_trials, trial_sr_std)
        idx_max = int(np.argmax(srs))
        sr_max = float(srs[idx_max])

        r_max = series[idx_max]
        stats = sharpe_stats(r_max)
        d = dsr(stats["sr"], N_OBS, stats["skew"], stats["kurtosis"], n_trials, trial_sr_std)

        max_srs.append(sr_max)
        dsrs.append(d)
        if not math.isnan(d) and d > 0.95:
            count_gt_095 += 1

    max_srs = np.array(max_srs)
    dsrs_arr = np.array([x for x in dsrs if not math.isnan(x)])
    mean_sr_max = float(np.mean(max_srs))
    sr0_at_se = expected_max_sharpe(n_trials, se)
    diff_se_units = (mean_sr_max - sr0_at_se) / se
    median_dsr = float(np.median(dsrs_arr)) if len(dsrs_arr) > 0 else float("nan")
    frac_gt_095 = count_gt_095 / n_reps

    result = {
        "arm": f"shared_factor_N{n_trials}_rho{rho}",
        "n_trials": n_trials,
        "rho": rho,
        "n_reps": n_reps,
        "mean_sr_max": mean_sr_max,
        "sr0": sr0_at_se,
        "diff_se_units": diff_se_units,
        "median_dsr": median_dsr,
        "fraction_dsr_gt_095": frac_gt_095,
        "note": "Secondary arm; never pooled with independent arm",
    }
    _ckpt(out_dir, f"shared_factor_N{n_trials}_rho{rho}", result)
    print(
        f"[shared/N={n_trials}/rho={rho}] "
        f"mean(SR_max)-SR0 in SE = {diff_se_units:+.3f}, "
        f"median DSR = {median_dsr:.3f}, "
        f"frac DSR>0.95 = {frac_gt_095:.4f}"
    )
    return result


def arm_positive_control(out_dir, true_sharpe=0.5, n_reps=500):
    """Positive control: injected-Sharpe series.

    DSR PASS fraction at N=1 and at selection N=10.
    """
    rng = np.random.default_rng(MASTER + 300)
    mu = true_sharpe * SIGMA  # injected mean per period

    # N=1
    pass_n1 = 0
    for i in range(n_reps):
        r = rng.normal(mu, SIGMA, size=N_OBS)
        res = _dsr_for_series(r, n_trials=1, trial_sr_std=SE, rule="null_theoretical")
        if not math.isnan(res["dsr"]) and res["dsr"] > 0.95:
            pass_n1 += 1
    frac_n1 = pass_n1 / n_reps

    # N=10 selection (max of 10 null + injected on the selected one)
    # Simpler: injected edge on each trial, take max, compute DSR
    pass_n10 = 0
    for i in range(n_reps):
        series = rng.normal(mu, SIGMA, size=(10, N_OBS))
        srs = series.mean(axis=1) / series.std(axis=1, ddof=1)
        srs = np.where(np.isfinite(srs), srs, 0.0)
        idx_max = int(np.argmax(srs))
        r_max = series[idx_max]
        stats = sharpe_stats(r_max)
        v = float(np.var(srs, ddof=1))
        trial_sr_std = math.sqrt(v) if v > 0 else 0.0
        sr0 = expected_max_sharpe(10, trial_sr_std)
        d = dsr(stats["sr"], N_OBS, stats["skew"], stats["kurtosis"], 10, trial_sr_std)
        if not math.isnan(d) and d > 0.95:
            pass_n10 += 1
    frac_n10 = pass_n10 / n_reps

    result = {
        "arm": "positive_control",
        "true_sharpe": true_sharpe,
        "n_reps": n_reps,
        "frac_pass_N1": frac_n1,
        "frac_pass_N10": frac_n10,
    }
    _ckpt(out_dir, "positive_control", result)
    print(
        f"[pos/control/SR={true_sharpe}] "
        f"PASS frac N=1 = {frac_n1:.3f}, N=10 = {frac_n10:.3f}"
    )
    return result


def arm_ci_probes(out_dir):
    """Cheap Sharpe-CI probes (diagnostic, non-gating).

    ci_sharpe coverage on AR(1) at block length 1x/2x/4x automatic choice;
    PAIRED mean-vs-Sharpe coverage on same series; bootstrap mean of Sharpe
    vs sample Sharpe (bias check). Small series count due to ci_sharpe cost.
    """
    n_probe = 20
    rng = np.random.default_rng(MASTER + 400)

    # Determine automatic block length on a representative AR(1) series
    rep = ar1(N_OBS, 999, SIGMA, 0.3)
    _, bl_auto = _auto_block_length(rep)

    results = {}
    for mult in [1.0, 2.0, 4.0]:
        bl = int(bl_auto * mult)
        cover_sharpe = 0
        cover_mean = 0
        for i in range(n_probe):
            r = ar1(N_OBS, int(rng.integers(0, 2**31)), SIGMA, 0.3)
            # ci_sharpe coverage: true Sharpe = 0 (AR(1) zero mean)
            lo_s, hi_s = ci_sharpe(r, 0.95, int(rng.integers(0, 2**31)), n_boot=1999, block_length=bl)
            if lo_s <= 0.0 <= hi_s:
                cover_sharpe += 1
            # Paired mean coverage: true mean = 0
            lo_m, hi_m = ci_mean(r, 0.95, int(rng.integers(0, 2**31)), n_boot=1999, block_length=bl)
            if lo_m <= 0.0 <= hi_m:
                cover_mean += 1
        results[f"bl_{mult}x"] = {
            "block_length": bl,
            "ci_sharpe_coverage": cover_sharpe / n_probe,
            "ci_mean_coverage": cover_mean / n_probe,
            "n_probe": n_probe,
        }
        print(
            f"[ci/bl={mult}x bl={bl}] "
            f"ci_sharpe cov = {cover_sharpe/n_probe:.2f}, "
            f"ci_mean cov = {cover_mean/n_probe:.2f}"
        )

    # Bootstrap mean of Sharpe vs sample Sharpe (bias check)
    bias_diffs = []
    for i in range(n_probe):
        r = ar1(N_OBS, int(rng.integers(0, 2**31)), SIGMA, 0.3)
        _, bl = _auto_block_length(r)
        sample_sr = r.mean() / r.std(ddof=1)
        boots = []
        for b in range(1999):
            nblocks = (N_OBS + bl - 1) // bl
            starts = rng.integers(0, N_OBS, size=nblocks)
            idx = np.concatenate([(s + np.arange(bl)) % N_OBS for s in starts])[:N_OBS]
            samp = r[idx]
            sd = samp.std(ddof=1)
            boots.append(samp.mean() / sd if sd > 0 else float("nan"))
        boots = np.array([x for x in boots if not math.isnan(x)])
        if len(boots) > 0:
            bias_diffs.append(float(np.mean(boots) - sample_sr))
    mean_bias = float(np.mean(bias_diffs)) if bias_diffs else float("nan")
    results["sharpe_bootstrap_bias"] = {
        "mean_bootstrap_minus_sample": mean_bias,
        "n_probe": n_probe,
    }
    print(f"[ci/bias] bootstrap Sharpe - sample Sharpe = {mean_bias:+.6f}")

    _ckpt(out_dir, "ci_probes", results)
    return results


def run_b4(out_dir=RESULTS_DIR, unbound=False, repo="."):
    """Run all B4 arms and produce a bound artifact."""
    start_time = datetime.now(timezone.utc)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # Binding at run start
    start_sha, start_dirty, start_untracked = enforce_clean_tree(
        allow_unbound=unbound, repo=repo
    )
    start_runner_hash = runner_sha256(repo)

    # Run arms
    cal_iid = arm_dsr_calibration_iid(out_dir)
    cal_ar1 = arm_dsr_calibration_ar1(out_dir)

    sel_10_sample = arm_selection_experiment(out_dir, n_trials=10, v_rule="sample")
    sel_10_true = arm_selection_experiment(out_dir, n_trials=10, v_rule="true_sigma")
    sel_100_sample = arm_selection_experiment(out_dir, n_trials=100, v_rule="sample")
    sel_100_true = arm_selection_experiment(out_dir, n_trials=100, v_rule="true_sigma")

    shared = arm_shared_factor(out_dir, n_trials=10, rho=0.3)
    pos = arm_positive_control(out_dir, true_sharpe=0.5)
    ci = arm_ci_probes(out_dir)

    end_time = datetime.now(timezone.utc)
    wall_clock = time.time() - t0

    # Binding integrity re-check
    end_sha, end_dirty, end_untracked = git_info(repo)
    if end_sha != start_sha:
        raise RuntimeError(f"HEAD changed during run: {start_sha} -> {end_sha}")
    if end_dirty != start_dirty or end_untracked != start_untracked:
        raise RuntimeError("Tree cleanliness changed during run")

    final = {
        # Binding fields at top level (required by verify_binding)
        "git_sha": start_sha,
        "dirty": start_dirty,
        "untracked_files": start_untracked,
        "runner_sha256": start_runner_hash,
        "command_line": " ".join(sys.argv),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "pip_freeze_hash": pip_freeze_hash(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "blas_threads": blas_threads(),
        "start_time_utc": start_time.strftime("%Y%m%dT%H%M%SZ"),
        "end_time_utc": end_time.strftime("%Y%m%dT%H%M%SZ"),
        "wall_clock_seconds": wall_clock,
        "unbound": bool(unbound),
        "binding": binding_fields(repo, extra={
            "start_sha": start_sha,
            "start_dirty": start_dirty,
            "start_untracked": start_untracked,
            "start_runner_hash": start_runner_hash,
        }),
        "arms": {
            "dsr_calib_iid_null_N1": cal_iid,
            "dsr_calib_ar1_null_N1": cal_ar1,
            "selection_N10_sample": sel_10_sample,
            "selection_N10_true_sigma": sel_10_true,
            "selection_N100_sample": sel_100_sample,
            "selection_N100_true_sigma": sel_100_true,
            "shared_factor_N10_rho0.3": shared,
            "positive_control": pos,
            "ci_probes": ci,
        },
    }
    if unbound:
        final["binding_status"] = "UNBOUND"
        final["binding"]["binding_status"] = "UNBOUND"
    else:
        final["binding_status"] = "BOUND"
        final["binding"]["binding_status"] = "BOUND"

    ts = end_time.strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"selfvalidation_v3_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final, f, sort_keys=True, indent=2)

    print(f"\nArtifact: {path}")
    print(f"Binding: {final['binding']['binding_status']}")
    print(f"git_sha: {final['binding']['git_sha']}")
    print(f"Wall clock: {wall_clock:.1f}s")

    # Verify binding on the artifact we just wrote
    with open(path) as f:
        written = json.load(f)
    ok, reason = verify_binding(written, repo)
    print(f"Binding verification: {'PASS' if ok else 'FAIL'} - {reason}")

    return final, path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WO-003 B4 self-validation v3 bound run")
    parser.add_argument("--unbound", action="store_true",
                        help="Run on non-clean tree; artifact labeled UNBOUND")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    repo = "."
    run_b4(out_dir=args.out_dir, unbound=args.unbound, repo=repo)


if __name__ == "__main__":
    main()
