"""Worked example for Parapix (synthetic data only).

Planted edge: true Sharpe = 0.5 (mu=0.005 per period, sigma=0.01).
N=20 candidate strategies: 19 null (mu=0) + 1 planted edge.
Seed: 20261126 (master). n_obs=1000 per strategy. Runs on CPU, no network.

Usage: python examples/worked_example.py
"""

import time

import numpy as np

from parapix.evaluate import evaluate_sharpe
from parapix.prereg import PreReg
from parapix.sim import iid_noise, injected_edge
from parapix.verdict import verdict

MASTER_SEED = 20261126
N_NULL = 19
N_CANDIDATES = 20
N_OBS = 1000
SIGMA = 0.01
PLANTED_MU = 0.005
PLANTED_TRUE_SHARPE = PLANTED_MU / SIGMA


def build_candidates(seed=MASTER_SEED, n_null=N_NULL, n_obs=N_OBS, sigma=SIGMA, planted_mu=PLANTED_MU):
    ss = np.random.SeedSequence(seed)
    seeds = ss.spawn(n_null)
    nulls = [iid_noise(n_obs, int(s.generate_state(1)[0]), sigma) for s in seeds]
    planted = injected_edge(n_obs, seed + 999, sigma, planted_mu)
    return nulls, planted


def make_prereg(n_trials, seed=MASTER_SEED):
    return PreReg(
        run_id="worked-example",
        created_utc="2026-01-01T00:00:00Z",
        hypothesis="planted edge vs null best-of-N",
        metric="sharpe",
        minimum_effect=0.0,
        alpha=0.05,
        n_candidates_tried=N_CANDIDATES,
        n_trials=n_trials,
        seeds={"data": seed, "permutation": seed + 1, "split": seed + 2},
        split_scheme="single_series",
        bars={"ci_level": 0.95},
        code_version={"git_sha": "local", "dirty": False},
    )


def run_example(seed=MASTER_SEED, n_obs=N_OBS, n_boot=1999, verbose=True):
    t0 = time.time()
    nulls, planted = build_candidates(seed=seed, n_obs=n_obs)

    sharpes = [float(r.mean() / r.std(ddof=1)) for r in nulls]
    best_idx = int(np.argmax(sharpes))
    best_null = nulls[best_idx]
    trap_raw_sharpe = sharpes[best_idx]

    if verbose:
        print("=" * 72)
        print("Parapix worked example (synthetic data)")
        print(f"planted true Sharpe = {PLANTED_TRUE_SHARPE}")
        print(f"N candidates = {N_CANDIDATES} ({N_NULL} null + 1 planted), n_obs = {n_obs}")
        print("=" * 72)
        print()
        print("(a) THE TRAP")
        print(f"    best-of-{N_NULL} nulls raw Sharpe = {trap_raw_sharpe:.4f}")
        print("    naive read: a positive raw Sharpe among the best of many nulls")
        print("    looks like an edge, even though every series is pure noise.")
        print()

    # prereg written before the selected candidate is evaluated
    prereg_trap = make_prereg(N_CANDIDATES, seed=seed)
    res_trap = evaluate_sharpe(best_null, prereg_trap, n_boot=n_boot)
    v_trap = verdict(prereg_trap, res_trap)
    if verbose:
        print("(b) THE CORRECTION")
        print(f"    prereg n_trials = {prereg_trap.n_trials} (written before evaluation)")
        print(f"    corrected_p = {res_trap.corrected_p:.4f}, DSR = {res_trap.dsr:.4f}, SR0 = {res_trap.sr0:.4f}")
        print(f"    verdict = {v_trap['verdict']}  (null best must FAIL)")
        print()

    prereg_real = make_prereg(N_CANDIDATES, seed=seed)
    res_real = evaluate_sharpe(planted, prereg_real, n_boot=n_boot)
    v_real = verdict(prereg_real, res_real)
    planted_raw_sharpe = float(planted.mean() / planted.std(ddof=1))
    if verbose:
        print("(c) THE REAL ONE")
        print(f"    planted raw Sharpe = {planted_raw_sharpe:.4f} (true {PLANTED_TRUE_SHARPE})")
        print(f"    corrected_p = {res_real.corrected_p:.4f}, DSR = {res_real.dsr:.4f}, SR0 = {res_real.sr0:.4f}")
        print(f"    verdict = {v_real['verdict']}  (planted edge must PASS)")
        print()

    # sample-V rule at small N -> WO-003c warning
    small_n_trials = 5
    prereg_warn = make_prereg(small_n_trials, seed=seed)
    trial_sharpes = [float(r.mean() / r.std(ddof=1)) for r in nulls[:small_n_trials]]
    res_warn = evaluate_sharpe(nulls[0], prereg_warn, trial_sharpes=trial_sharpes, n_boot=min(n_boot, 99))
    v_warn = verdict(prereg_warn, res_warn)
    if verbose:
        print("(d) THE WARNING")
        print(f"    sample-V rule, n_trials = {small_n_trials} (small N)")
        print(f"    verdict = {v_warn['verdict']}")
        print(f"    warnings = {v_warn['warnings']}")
        print()
        print(f"elapsed = {time.time() - t0:.1f}s")

    return {
        "trap_raw_sharpe": trap_raw_sharpe,
        "trap_verdict": v_trap["verdict"],
        "planted_raw_sharpe": planted_raw_sharpe,
        "planted_verdict": v_real["verdict"],
        "warning_verdict": v_warn["verdict"],
        "warning_present": bool(v_warn["warnings"]),
        "trap_result": res_trap,
        "planted_result": res_real,
        "warning_result": res_warn,
    }


def main():
    run_example()


if __name__ == "__main__":
    main()
