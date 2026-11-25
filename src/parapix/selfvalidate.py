import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy

from .sim import iid_noise, ar1


def _git_info(repo):
    sha = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    if not sha:
        sha = "UNKNOWN"
    dirty_out = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout.strip()
    return sha, bool(dirty_out)


def run(n_strategies, n_obs, phi, alpha, seed):
    rng = np.random.default_rng(seed)
    iid_rejections = 0
    ar1_rejections = 0
    from scipy import stats
    for _ in range(n_strategies):
        r_iid = iid_noise(n_obs, int(rng.integers(0, 2**31)), 1.0)
        _, p_iid = stats.ttest_1samp(r_iid, 0.0)
        if p_iid < alpha:
            iid_rejections += 1
        r_ar1 = ar1(n_obs, int(rng.integers(0, 2**31)), 1.0, phi)
        _, p_ar1 = stats.ttest_1samp(r_ar1, 0.0)
        if p_ar1 < alpha:
            ar1_rejections += 1
    repo = "."
    try:
        git_sha, dirty = _git_info(repo)
    except Exception:
        git_sha, dirty = "UNKNOWN", False
    result = {
        "false_positive_rate_iid": iid_rejections / n_strategies,
        "false_positive_rate_ar1": ar1_rejections / n_strategies,
        "n_strategies": n_strategies,
        "n_obs": n_obs,
        "phi": phi,
        "alpha": alpha,
        "seed": seed,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "python_version": __import__("sys").version,
        "git_sha": git_sha,
        "dirty": dirty,
    }
    return result


def write_result(result, out_dir="results"):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"{out_dir}/selfvalidation_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, indent=2, ensure_ascii=False)
    return path
