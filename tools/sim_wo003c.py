"""WO-003c Phase B: sample-V over-deflation simulation driver (B1-B6).

Measures the finite-sample behaviour of the Deflated Sharpe Ratio's
trial-variance estimate V. The library DSR is the single source of truth: the
driver calls ``parapix.dsr`` (``expected_max_sharpe`` + ``psr`` / ``dsr``) and
never re-derives the formula. ``expected_max_sharpe`` is exactly linear in
``trial_sr_std``, so ``dsr(sr, n, 0, 3, N, s) == psr(sr, g(N)*s, n, 0, 3)``;
the driver uses the ``psr`` form with ``g(N) = expected_max_sharpe(N, 1.0)``
for throughput and asserts the identity once at start-up.

Arms (registered in tools/predict_wo003c.py, Phase A):
  B3  independent-trials sweep, cells (a) sample-V, (b) null-theoretical,
      (d) leave-selected-out exceedance, per N in {5,10,20,50,100,250};
      (c) independent-std by quadrature (+ small MC confirm);
      decomposition (b/a = (b/c)*(c/a)); mechanism corr(SR_max, sample std).
  A2  shared-factor (rho) cells, N in {10,50}, rho in {0.3,0.6,0.9},
      sample-V exceedance + oracle-V comparator V=(1-rho)sigma^2.
  B5  power arm (DSR-only gaps, theoretical-V minus sample-V) over
      SR in {0.10,0.15,0.20} x N grid; pass-fraction arm (evaluate_sharpe
      conjunction) at N in {10,50}.
  B6  return-series bridge (N=10 at 400k, consistency vs the 2M N=10 cell);
      convergence doubling spot check; budget/counts/runtime.
  A4  floor rule on PAIRED (a)/(b) draws (common random numbers).
  A5  warning ladder from the floor (A4) and power materiality (A3).

Counts/seeds are the registered A6 values (scaled by --scale for pilot).
Cells are independent (own SeedSequence) and run in a process pool; each
writes results/checkpoint_wo003c_<cell>.json. The driver refuses a dirty tree
(BOUND) unless --unbound, and re-verifies binding at run end.

Run: ``python -m tools.sim_wo003c``            (full, from repo root)
     ``python -m tools.sim_wo003c --pilot``     (2000 reps/cell, timing)
     ``python -m tools.sim_wo003c --cells ind_rho0_10``  (subset)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parapix.dsr import dsr, expected_max_sharpe, psr  # noqa: E402
from parapix.evaluate import evaluate_sharpe  # noqa: E402
from parapix.prereg import PreReg  # noqa: E402
from parapix.verdict import verdict  # noqa: E402
from parapix.selfvalidate import (  # noqa: E402
    binding_fields,
    enforce_clean_tree,
    git_info,
    runner_sha256,
    verify_binding,
)

RESULTS_DIR = "results"

# --- registered settings (identical to Phase A) ---------------------------
N_OBS = 1000
SE = math.sqrt(1.0 / (N_OBS - 1))     # null-theoretical per-trial SR std
SIGMA = SE                             # per-period return std (SR is scale-free)
ALPHA = 0.05
DSR_THRESH = 0.95
Z_95 = float(norm.ppf(DSR_THRESH))     # 1.6448536...
Z_99 = float(norm.ppf(0.99))           # one-sided 99% for the A4 floor
GRID = [5, 10, 20, 50, 100, 250]
POWER_SR_GRID = [0.10, 0.15, 0.20]
POWER_N_GRID = [10, 50, 100, 250]
RHO_GRID = [0.3, 0.6, 0.9]
RHO_N_GRID = [10, 50]

# --- registered A6 counts -------------------------------------------------
R_RHO0 = 2_000_000
R_RHO_GT0 = 200_000
R_POWER = 200_000
R_BRIDGE = 400_000
R_PASSFRAC = 10_000
PASSFRAC_N = [10, 50]
MASTER = 20261126

# g(N) = SR0(N, 1.0)/1.0, computed once per N from the library.
_G = {N: expected_max_sharpe(N, 1.0) for N in set(GRID) | set(POWER_N_GRID)}


# ---------------------------------------------------------------------------
# Core vectorised statistics (heavy numeric part only; DSR via library psr)
# ---------------------------------------------------------------------------
def _srs(series: np.ndarray) -> np.ndarray:
    """Per-trial Sharpe (mean/std, ddof=1); non-finite -> 0.0. Shape (..., N)."""
    sd = series.std(axis=-1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = series.mean(axis=-1) / sd
    return np.where(np.isfinite(sr), sr, 0.0)


def _exceed_psr(sr_max: np.ndarray, sr0: np.ndarray) -> np.ndarray:
    """Boolean DSR>0.95 via library psr (skew=0, kurt=3). sr0 = g(N)*std."""
    out = np.empty(sr_max.shape, dtype=bool)
    flat_sr = sr_max.ravel()
    flat_sr0 = sr0.ravel()
    flat = out.ravel()
    for i in range(flat_sr.size):
        flat[i] = psr(float(flat_sr[i]), float(flat_sr0[i]), N_OBS, 0.0, 3.0) > DSR_THRESH
    return out


def _assert_psr_equals_dsr() -> None:
    rng = np.random.default_rng(0)
    for N in (5, 10, 250):
        g = _G[N]
        for _ in range(50):
            s = float(rng.uniform(0.01, 0.5))
            sr = float(rng.normal(0.0, 0.1))
            a = dsr(sr, N_OBS, 0.0, 3.0, N, s)
            b = psr(sr, g * s, N_OBS, 0.0, 3.0)
            if not (math.isnan(a) and math.isnan(b)) and abs(a - b) > 1e-12:
                raise RuntimeError(f"psr/dsr identity broken at N={N}: {a} vs {b}")


# ---------------------------------------------------------------------------
# B3: independent-trials cell (a)/(b)/(d) + mechanism, paired CRN
# ---------------------------------------------------------------------------
def cell_ind_rho0(N: int, R: int, seed: int, batch: int | None = None) -> dict:
    g = _G[N]
    if batch is None:
        batch = max(1, min(4096, 8_000_000 // (N * N_OBS)))
    rng = np.random.default_rng(seed)
    cnt_a = cnt_b = cnt_d = cnt_ab = 0
    sum_sr = sum_sd = sum_sr_sd = sum_sr2 = sum_sd2 = 0.0
    done = 0
    t0 = time.time()
    while done < R:
        b = min(batch, R - done)
        series = rng.normal(0.0, SIGMA, size=(b, N, N_OBS))
        srs = _srs(series)                                  # (b, N)
        idx = np.argmax(srs, axis=1)
        sr_max = srs[np.arange(b), idx]                     # (b,)
        sd = srs.std(axis=1, ddof=1)                        # sample-V (b,)
        # leave-selected-out std (ddof=1 over the N-1 non-max trials)
        s_sum = srs.sum(axis=1)
        s_sq = (srs * srs).sum(axis=1)
        m = N - 1
        loo_mean = (s_sum - sr_max) / m
        loo_var = (s_sq - sr_max * sr_max - m * loo_mean * loo_mean) / (m - 1) if m > 1 else 0.0
        sd_loo = np.sqrt(np.maximum(loo_var, 0.0))
        # (a) sample-V, (b) null-theoretical, (d) leave-selected-out
        ea = _exceed_psr(sr_max, g * sd)
        eb = _exceed_psr(sr_max, g * np.full(b, SE))
        ed = _exceed_psr(sr_max, g * sd_loo)
        cnt_a += int(ea.sum())
        cnt_b += int(eb.sum())
        cnt_d += int(ed.sum())
        cnt_ab += int((ea & eb).sum())
        sum_sr += float(sr_max.sum())
        sum_sd += float(sd.sum())
        sum_sr_sd += float((sr_max * sd).sum())
        sum_sr2 += float((sr_max * sr_max).sum())
        sum_sd2 += float((sd * sd).sum())
        done += b
    p_a, p_b, p_d = cnt_a / R, cnt_b / R, cnt_d / R
    # mechanism correlation
    cov = sum_sr_sd / R - (sum_sr / R) * (sum_sd / R)
    var_sr = sum_sr2 / R - (sum_sr / R) ** 2
    var_sd = sum_sd2 / R - (sum_sd / R) ** 2
    corr = cov / math.sqrt(var_sr * var_sd) if var_sr > 0 and var_sd > 0 else float("nan")
    # A4 paired one-sided 99% lower bound on p_a/p_b (delta method on log ratio)
    p_ab = cnt_ab / R
    cov_ab = p_ab - p_a * p_b
    if p_a > 0 and p_b > 0:
        var_logratio = ((1.0 - p_a) / (R * p_a)
                        + (1.0 - p_b) / (R * p_b)
                        - 2.0 * cov_ab / (R * p_a * p_b))
    else:
        var_logratio = float("nan")
    ratio = p_a / p_b if p_b > 0 else float("nan")
    if var_logratio > 0:
        ratio_lwr99 = math.exp(math.log(ratio) - Z_99 * math.sqrt(var_logratio))
    else:
        ratio_lwr99 = float("nan")
    return {
        "cell": f"ind_rho0_N{N}", "N": N, "R": R, "batch": batch,
        "exceed_a_sampleV": p_a, "exceed_b_nullTheo": p_b, "exceed_d_loo": p_d,
        "count_a": cnt_a, "count_b": cnt_b, "count_d": cnt_d, "count_ab": cnt_ab,
        "ratio_a_over_b": ratio, "ratio_a_over_b_lwr99": ratio_lwr99,
        "mech_corr_sr_max_sample_std": corr,
        "mean_sr_max": sum_sr / R, "mean_sample_std": sum_sd / R,
        "se": SE, "gN": g, "z95": Z_95,
        "wall_clock_seconds": time.time() - t0,
    }


def var_of_ind(cnt: int, R: int, p: float) -> float:
    """Per-draw indicator variance p(1-p) (retained for reference)."""
    return p * (1.0 - p)


# ---------------------------------------------------------------------------
# A2: shared-factor (rho) cell, sample-V + oracle-V
# ---------------------------------------------------------------------------
def cell_rho(N: int, rho: float, R: int, seed: int, batch: int | None = None) -> dict:
    g = _G[N]
    if batch is None:
        batch = max(1, min(2048, 8_000_000 // (N * N_OBS)))
    rng = np.random.default_rng(seed)
    sr_rho = math.sqrt(rho)
    sr_1rho = math.sqrt(1.0 - rho)
    oracle_var = (1.0 - rho) * SIGMA * SIGMA      # V=(1-rho)sigma^2 (wrong benchmark)
    oracle_std = math.sqrt(oracle_var)
    cnt_sample = cnt_oracle = 0
    done = 0
    t0 = time.time()
    while done < R:
        b = min(batch, R - done)
        common = rng.normal(0.0, 1.0, size=(b, N_OBS))
        idio = rng.normal(0.0, 1.0, size=(b, N, N_OBS))
        series = (sr_rho * common[:, None, :] + sr_1rho * idio) * SIGMA
        srs = _srs(series)
        idx = np.argmax(srs, axis=1)
        sr_max = srs[np.arange(b), idx]
        sd = srs.std(axis=1, ddof=1)
        es = _exceed_psr(sr_max, g * sd)
        eo = _exceed_psr(sr_max, g * np.full(b, oracle_std))
        cnt_sample += int(es.sum())
        cnt_oracle += int(eo.sum())
        done += b
    return {
        "cell": f"rho_N{N}_rho{rho}", "N": N, "rho": rho, "R": R, "batch": batch,
        "exceed_sampleV": cnt_sample / R, "exceed_oracleV": cnt_oracle / R,
        "oracle_std": oracle_std, "sampleV_note": "oracle V=(1-rho)sigma^2 is the wrong benchmark",
        "se": SE, "gN": g, "wall_clock_seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# B5: power arm (DSR-only gaps) and pass-fraction arm (conjunction)
# ---------------------------------------------------------------------------
def cell_power(SR: float, N: int, R: int, seed: int, batch: int | None = None) -> dict:
    g = _G[N]
    mu = SR * SIGMA
    if batch is None:
        batch = max(1, min(2048, 8_000_000 // (N * N_OBS)))
    rng = np.random.default_rng(seed)
    cnt_theo = cnt_sample = 0
    done = 0
    t0 = time.time()
    while done < R:
        b = min(batch, R - done)
        series = rng.normal(mu, SIGMA, size=(b, N, N_OBS))
        srs = _srs(series)
        idx = np.argmax(srs, axis=1)
        sr_max = srs[np.arange(b), idx]
        sd = srs.std(axis=1, ddof=1)
        et = _exceed_psr(sr_max, g * np.full(b, SE))
        es = _exceed_psr(sr_max, g * sd)
        cnt_theo += int(et.sum())
        cnt_sample += int(es.sum())
        done += b
    p_theo, p_samp = cnt_theo / R, cnt_sample / R
    gap = p_theo - p_samp
    ratio = (p_samp / p_theo) if p_theo > 0 else float("nan")
    return {
        "cell": f"power_SR{SR}_N{N}", "SR": SR, "N": N, "R": R, "batch": batch,
        "power_theoreticalV": p_theo, "power_sampleV": p_samp,
        "gap_theo_minus_sample": gap, "power_ratio_sample_over_theo": ratio,
        "materiality_gap_ge_0.10": gap >= 0.10,
        "materiality_ratio_lt_0.75_SR0.10": (SR == 0.10 and ratio < 0.75),
        "se": SE, "gN": g, "wall_clock_seconds": time.time() - t0,
    }


def _make_prereg(N: int) -> PreReg:
    return PreReg(
        run_id="wo003c", created_utc="2026-11-26T00:00:00Z",
        hypothesis="sample-V over-deflation", metric="sharpe",
        minimum_effect=0.0, alpha=ALPHA, n_candidates_tried=1, n_trials=N,
        seeds={"data": 1, "permutation": 2, "split": 3},
        split_scheme="none", bars={"ci_level": 0.95},
        code_version={"git_sha": "wo003c", "dirty": False},
    )


def cell_passfrac(SR: float, N: int, R: int, seed: int) -> dict:
    """Conjunction PASS fraction (evaluate_sharpe + verdict), run_b4 style."""
    mu = SR * SIGMA
    rng = np.random.default_rng(seed)
    passes = 0
    t0 = time.time()
    for _ in range(R):
        series = rng.normal(mu, SIGMA, size=(N, N_OBS))
        srs = _srs(series)
        srs = np.where(np.isfinite(srs), srs, 0.0)
        idx = int(np.argmax(srs))
        r_max = series[idx]
        p = _make_prereg(N)
        res = evaluate_sharpe(r_max, p, trial_sharpes=list(srs), n_boot=99)
        if verdict(p, res)["verdict"] == "PASS":
            passes += 1
    return {
        "cell": f"passfrac_SR{SR}_N{N}", "SR": SR, "N": N, "R": R,
        "pass_fraction": passes / R, "conjunction": "evaluate_sharpe + verdict",
        "wall_clock_seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Cell registry
# ---------------------------------------------------------------------------
def build_cells(scale: float) -> list[tuple[str, dict]]:
    cells: list[tuple[str, dict]] = []
    for N in GRID:
        cells.append((f"ind_rho0_N{N}",
                      {"fn": "ind", "N": N, "R": int(R_RHO0 * scale), "seed": MASTER + N}))
    for N in RHO_N_GRID:
        for rho in RHO_GRID:
            cells.append((f"rho_N{N}_rho{rho}",
                          {"fn": "rho", "N": N, "rho": rho,
                           "R": int(R_RHO_GT0 * scale), "seed": MASTER + 10000 + N * 10 + int(rho * 100)}))
    for SR in POWER_SR_GRID:
        for N in POWER_N_GRID:
            cells.append((f"power_SR{SR}_N{N}",
                          {"fn": "power", "SR": SR, "N": N,
                           "R": int(R_POWER * scale), "seed": MASTER + 20000 + int(SR * 1000) + N}))
    for SR in POWER_SR_GRID:
        for N in PASSFRAC_N:
            cells.append((f"passfrac_SR{SR}_N{N}",
                          {"fn": "passfrac", "SR": SR, "N": N,
                           "R": int(R_PASSFRAC * scale), "seed": MASTER + 30000 + int(SR * 1000) + N}))
    cells.append(("bridge_N10", {"fn": "ind", "N": 10, "R": int(R_BRIDGE * scale),
                                 "seed": MASTER + 40000}))
    return cells


def _run_one(cell: dict) -> dict:
    fn = cell["fn"]
    if fn == "ind":
        return cell_ind_rho0(cell["N"], cell["R"], cell["seed"])
    if fn == "rho":
        return cell_rho(cell["N"], cell["rho"], cell["R"], cell["seed"])
    if fn == "power":
        return cell_power(cell["SR"], cell["N"], cell["R"], cell["seed"])
    if fn == "passfrac":
        return cell_passfrac(cell["SR"], cell["N"], cell["R"], cell["seed"])
    raise ValueError(fn)


def _ckpt(name: str, obj: dict) -> str:
    path = os.path.join(RESULTS_DIR, f"checkpoint_wo003c_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2)
    return path


# ---------------------------------------------------------------------------
# A4 floor + A5 ladder (applied to measured results)
# ---------------------------------------------------------------------------
def apply_a4_floor(ind_results: dict) -> dict:
    """Smallest grid N s.t. every grid N'>=N has ratio_lwr99 >= 0.5.

    Recomputes the one-sided 99% lower bound from the raw paired counts
    (count_a, count_b, count_ab, R) so it is correct regardless of any stored
    per-cell value.
    """
    by_n = {}
    for r in ind_results:
        N, R = r["N"], r["R"]
        pa, pb, pab = r["count_a"] / R, r["count_b"] / R, r["count_ab"] / R
        cov_ab = pab - pa * pb
        if pa > 0 and pb > 0:
            var = ((1.0 - pa) / (R * pa) + (1.0 - pb) / (R * pb)
                   - 2.0 * cov_ab / (R * pa * pb))
            lwr = math.exp(math.log(pa / pb) - Z_99 * math.sqrt(var)) if var > 0 else float("nan")
        else:
            lwr = float("nan")
        by_n[N] = {"ratio": pa / pb if pb > 0 else float("nan"), "lwr99": lwr}
    qualifying = []
    for N in GRID:
        ok = all(by_n[n]["lwr99"] >= 0.5 for n in GRID if n >= N)
        if ok:
            qualifying.append(N)
    floor = min(qualifying) if qualifying else None
    below = [n for n in GRID if floor is not None and n < floor]
    gap_to_below = (floor - max(below)) if (floor is not None and below) else None
    return {"floor_N": floor, "gap_to_next_below": gap_to_below, "per_N": by_n}


def apply_a5_ladder(floor: dict, power_results: list) -> dict:
    fires_floor = floor["floor_N"] is not None
    material_ns = [r["N"] for r in power_results
                   if r["materiality_gap_ge_0.10"] or r["materiality_ratio_lt_0.75_SR0.10"]]
    fires_power = bool(material_ns)
    applies_below = None
    if fires_power:
        applies_below = max(material_ns)
    return {
        "fires_floor": fires_floor, "floor_N": floor["floor_N"],
        "fires_power": fires_power, "power_material_N": sorted(set(material_ns)),
        "warning_applies_below_N": floor["floor_N"] if fires_floor else applies_below,
        "ships": fires_floor or fires_power,
        "direction": "under sample-V at small N a PASS is conservative and a FAIL is weak evidence",
        "docs_only": not (fires_floor or fires_power),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(out_dir=RESULTS_DIR, scale=1.0, cells_filter=None, workers=6,
        unbound=False, repo=".", pilot=False, assemble_only=False):
    start_time = datetime.now(timezone.utc)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    _assert_psr_equals_dsr()

    start_sha, start_dirty, start_untracked = enforce_clean_tree(allow_unbound=unbound, repo=repo)
    start_runner_hash = runner_sha256(repo)

    cells = build_cells(scale)
    if cells_filter:
        cells = [c for c in cells if c[0] in cells_filter]
    print(f"[wo003c] {len(cells)} cells, scale={scale}, workers={workers}")

    results: dict[str, dict] = {}
    if assemble_only:
        for name, _spec in cells:
            path = os.path.join(out_dir, f"checkpoint_wo003c_{name}.json")
            with open(path) as f:
                results[name] = json.load(f)
            print(f"  [{name}] loaded")
    elif workers > 1 and len(cells) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, spec): name for name, spec in cells}
            for fut in as_completed(futs):
                name = futs[fut]
                res = fut.result()
                results[name] = res
                _ckpt(name, res)
                print(f"  [{name}] done ({res.get('wall_clock_seconds', 0):.1f}s) "
                      f"{_brief(res)}")
    else:
        for name, spec in cells:
            res = _run_one(spec)
            results[name] = res
            _ckpt(name, res)
            print(f"  [{name}] done ({res.get('wall_clock_seconds', 0):.1f}s) {_brief(res)}")

    # --- derived: decomposition, A4 floor, A5 ladder, convergence ---
    # Recompute each ind cell's lwr99 from raw counts (idempotent; fixes any
    # stale per-cell value carried in a checkpoint from an earlier code path).
    for r in results.values():
        if "count_a" in r and r.get("R"):
            R = r["R"]
            pa, pb, pab = r["count_a"] / R, r["count_b"] / R, r["count_ab"] / R
            cov_ab = pab - pa * pb
            if pa > 0 and pb > 0:
                var = ((1.0 - pa) / (R * pa) + (1.0 - pb) / (R * pb)
                       - 2.0 * cov_ab / (R * pa * pb))
                r["ratio_a_over_b_lwr99"] = (math.exp(math.log(pa / pb) - Z_99 * math.sqrt(var))
                                             if var > 0 else float("nan"))
    ind = [results[f"ind_rho0_N{N}"] for N in GRID if f"ind_rho0_N{N}" in results]
    decomp = {}
    for r in ind:
        N = r["N"]
        b, a = r["exceed_b_nullTheo"], r["exceed_a_sampleV"]
        decomp[N] = {"b_over_a": (b / a) if a > 0 else float("nan")}
    floor = apply_a4_floor(ind) if len(ind) == len(GRID) else {"floor_N": None, "note": "partial grid"}
    power = [v for k, v in results.items() if k.startswith("power_")]
    ladder = apply_a5_ladder(floor, power)

    # convergence doubling: N=10 at 2M vs bridge N=10 at 400k
    conv = None
    if "ind_rho0_N10" in results and "bridge_N10" in results:
        a = results["ind_rho0_N10"]["exceed_a_sampleV"]
        b = results["bridge_N10"]["exceed_a_sampleV"]
        se_a = math.sqrt(a * (1 - a) / results["ind_rho0_N10"]["R"])
        se_b = math.sqrt(b * (1 - b) / results["bridge_N10"]["R"])
        z = (a - b) / math.sqrt(se_a ** 2 + se_b ** 2) if (se_a + se_b) > 0 else float("nan")
        conv = {"N10_2M": a, "N10_400k": b, "z": z, "agree_2sigma": abs(z) < 2.0}

    end_time = datetime.now(timezone.utc)
    wall = time.time() - t0
    end_sha, end_dirty, end_untracked = git_info(repo)
    if end_sha != start_sha:
        raise RuntimeError(f"HEAD changed during run: {start_sha} -> {end_sha}")
    if end_dirty != start_dirty or end_untracked != start_untracked:
        raise RuntimeError("Tree cleanliness changed during run")

    final = {
        "git_sha": start_sha, "dirty": start_dirty, "untracked_files": start_untracked,
        "runner_sha256": start_runner_hash, "command_line": " ".join(sys.argv),
        "python_version": sys.version, "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "pip_freeze_hash": hashlib.sha256(
            __import__("subprocess").run([sys.executable, "-m", "pip", "freeze"],
                                         capture_output=True, text=True).stdout.encode("utf-8")).hexdigest(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "start_time_utc": start_time.strftime("%Y%m%dT%H%M%SZ"),
        "end_time_utc": end_time.strftime("%Y%m%dT%H%M%SZ"),
        "wall_clock_seconds": wall, "scale": scale, "pilot": pilot,
        "settings": {"n_obs": N_OBS, "sigma": SIGMA, "alpha": ALPHA,
                     "dsr_thresh": DSR_THRESH, "z95": Z_95, "grid": GRID,
                     "counts": {"rho0": R_RHO0, "rho_gt0": R_RHO_GT0, "power": R_POWER,
                                "bridge": R_BRIDGE, "passfrac": R_PASSFRAC}},
        "cells": results, "decomposition_b_over_a": decomp,
        "A4_floor": floor, "A5_ladder": ladder, "convergence_N10": conv,
        "binding": binding_fields(repo, extra={"start_sha": start_sha}),
        "binding_status": "UNBOUND" if unbound else "BOUND",
    }
    ts = end_time.strftime("%Y%m%dT%H%M%SZ")
    suffix = "_pilot" if pilot else ""
    path = os.path.join(out_dir, f"wo003c{suffix}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final, f, sort_keys=True, indent=2)
    print(f"\nArtifact: {path}")
    print(f"Binding: {final['binding_status']}  git_sha={start_sha}  wall={wall:.1f}s")
    with open(path) as f:
        written = json.load(f)
    ok, reason = verify_binding(written, repo)
    print(f"Binding verification: {'PASS' if ok else 'FAIL'} - {reason}")
    return final, path


def _brief(r: dict) -> str:
    if "exceed_a_sampleV" in r:
        return (f"a={r['exceed_a_sampleV']:.5f} b={r['exceed_b_nullTheo']:.5f} "
                f"d={r['exceed_d_loo']:.5f} corr={r['mech_corr_sr_max_sample_std']:+.3f}")
    if "exceed_sampleV" in r:
        return f"sampleV={r['exceed_sampleV']:.5f} oracleV={r['exceed_oracleV']:.5f}"
    if "power_theoreticalV" in r:
        return f"theo={r['power_theoreticalV']:.3f} samp={r['power_sampleV']:.3f} gap={r['gap_theo_minus_sample']:+.3f}"
    if "pass_fraction" in r:
        return f"pass={r['pass_fraction']:.3f}"
    return ""


def main():
    ap = argparse.ArgumentParser(description="WO-003c Phase B simulation driver")
    ap.add_argument("--pilot", action="store_true", help="2000 reps/cell for timing")
    ap.add_argument("--scale", type=float, default=None, help="count multiplier")
    ap.add_argument("--cells", nargs="*", default=None, help="subset of cell names")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--unbound", action="store_true")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip simulation; rebuild artifact from existing checkpoints")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args()
    scale = args.scale if args.scale is not None else (0.001 if args.pilot else 1.0)
    run(out_dir=args.out_dir, scale=scale, cells_filter=args.cells,
        workers=args.workers, unbound=args.unbound, pilot=args.pilot,
        assemble_only=args.assemble_only)


if __name__ == "__main__":
    main()