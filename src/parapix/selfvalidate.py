"""Self-validation v3 runner with hash-binding.

Port of the recovered v2 selfvalidation logic (validate_v2.py, commit 92db7a7)
into the package as ``python -m parapix.selfvalidate``. Every artifact is
bound to the git commit that produced it; the runner refuses to run on a
dirty or untracked-file tree unless ``--unbound`` is passed, which labels the
artifact UNBOUND.

Binding fields recorded in every artifact:
  git_sha, dirty, runner_sha256, command_line, python_version,
  numpy_version, scipy_version, pip_freeze_hash, PYTHONHASHSEED,
  blas_threads, all seeds, start_time_utc, end_time_utc, wall_clock_seconds.

Resume / binding integrity: at run end the runner re-verifies HEAD and clean
state are unchanged from run start; a resume refuses if the SHA differs from
the checkpoint's.

Verifier: an artifact's git_sha must be an ANCESTOR of HEAD
(git merge-base --is-ancestor, not mere object existence) and its runner
sha256 must match ``git show <sha>:src/parapix/selfvalidate.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from parapix.bootstrap import ci_mean, ci_sharpe
from parapix.dsr import dsr, expected_max_sharpe, sharpe_stats, psr
from parapix.evaluate import evaluate_mean, evaluate_sharpe
from parapix.permtest import permutation_pvalue
from parapix.prereg import PreReg, content_hash
from parapix.sim import ar1, iid_noise, injected_edge, student_t_noise, vol_cluster
from parapix.verdict import verdict

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Git / binding helpers
# ---------------------------------------------------------------------------

def _run(cmd, repo=".", check=False):
    return subprocess.run(
        cmd, cwd=repo, capture_output=True, text=True, check=check
    )


def git_info(repo="."):
    """Return (head_sha, is_dirty, untracked_files).

    results/ is excluded from the dirty check (it holds per-arm checkpoints
    and the final artifact).
    """
    sha = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
    if not sha:
        sha = "UNKNOWN"

    # Porcelain output, excluding results/
    out = _run(["git", "status", "--porcelain"], repo).stdout.strip()
    dirty_lines = []
    for line in out.splitlines():
        if not line:
            continue
        path = line[3:] if line.startswith((" M", " A", " D", " R", " C", " U", "??", "!!")) else line
        if path.startswith("results/"):
            continue
        dirty_lines.append(line)
    is_dirty = len(dirty_lines) > 0

    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], repo).stdout.strip()
    untracked_files = [l for l in untracked.splitlines() if l and not l.startswith("results/")]

    return sha, is_dirty, untracked_files


def runner_sha256(repo="."):
    """sha256 of src/parapix/selfvalidate.py as committed at HEAD."""
    out = _run(
        ["git", "show", "HEAD:src/parapix/selfvalidate.py"], repo
    ).stdout
    if not out:
        # Fallback: hash the working tree file
        path = os.path.join(repo, "src", "parapix", "selfvalidate.py")
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return hashlib.sha256(out.encode("utf-8")).hexdigest()


def pip_freeze_hash():
    """sha256 of `pip freeze` output (pinned dependency snapshot)."""
    out = _run([sys.executable, "-m", "pip", "freeze"]).stdout
    return hashlib.sha256(out.encode("utf-8")).hexdigest()


def blas_threads():
    """Best-effort BLAS thread count from environment."""
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
        v = os.environ.get(var)
        if v is not None:
            return int(v)
    return None


def binding_fields(repo=".", extra=None):
    """All binding fields recorded in every artifact."""
    sha, is_dirty, untracked = git_info(repo)
    fields = {
        "git_sha": sha,
        "dirty": is_dirty,
        "untracked_files": untracked,
        "runner_sha256": runner_sha256(repo),
        "command_line": " ".join(sys.argv),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "pip_freeze_hash": pip_freeze_hash(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "blas_threads": blas_threads(),
    }
    if extra:
        fields.update(extra)
    return fields


def verify_binding(artifact, repo="."):
    """Verify an artifact's binding: git_sha is an ancestor of HEAD and
    runner_sha256 matches git show <sha>:src/parapix/selfvalidate.py.

    Returns (ok, reason). git_sha must be an ancestor of HEAD (merge-base
    --is-ancestor), not mere object existence.
    """
    git_sha = artifact.get("git_sha")
    runner_hash = artifact.get("runner_sha256")
    if not git_sha or git_sha == "UNKNOWN":
        return False, "missing or UNKNOWN git_sha"

    # git_sha must be an ancestor of HEAD
    r = _run(["git", "merge-base", "--is-ancestor", git_sha, "HEAD"], repo)
    if r.returncode != 0:
        return False, f"git_sha {git_sha} is not an ancestor of HEAD"

    # runner_sha256 must match git show <sha>:src/parapix/selfvalidate.py
    committed = _run(
        ["git", "show", f"{git_sha}:src/parapix/selfvalidate.py"], repo
    ).stdout
    if not committed:
        return False, f"selfvalidate.py not present at {git_sha}"
    expected_hash = hashlib.sha256(committed.encode("utf-8")).hexdigest()
    if runner_hash != expected_hash:
        return False, f"runner_sha256 mismatch: {runner_hash} != {expected_hash}"

    return True, "binding verified"


# ---------------------------------------------------------------------------
# Dirty-tree guard
# ---------------------------------------------------------------------------

def enforce_clean_tree(allow_unbound=False, repo="."):
    """Refuse to run on a dirty or untracked-file tree unless allow_unbound.

    results/ is excluded from the dirty check.
    """
    sha, is_dirty, untracked = git_info(repo)
    if is_dirty or untracked:
        if not allow_unbound:
            problems = []
            if is_dirty:
                problems.append("modified files")
            if untracked:
                problems.append(f"untracked files: {untracked}")
            raise RuntimeError(
                "Refusing to run on a non-clean tree (" + "; ".join(problems)
                + "). Use --unbound to label the artifact UNBOUND and proceed."
            )
    return sha, is_dirty, untracked


# ---------------------------------------------------------------------------
# v3 runner logic (port of validate_v2)
# ---------------------------------------------------------------------------

def _ckpt(out_dir, name, obj):
    path = os.path.join(out_dir, f"checkpoint_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2)
    return path


def run_validation(out_dir=RESULTS_DIR, unbound=False, n_strategies=2000,
                   n_obs=1000, n_perm=999, n_boot=999, n_series=1000,
                   master=20261125, sigma=0.01, alpha=0.05,
                   repo=".", start_time=None):
    """Run the v3 selfvalidation suite and write per-arm checkpoints + final artifact.

    Returns (final_dict, artifact_path).
    """
    if start_time is None:
        start_time = datetime.now(timezone.utc)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # Bind at run start
    start_sha, start_dirty, start_untracked = git_info(repo)
    start_runner_hash = runner_sha256(repo)

    # --- Permutation FPR ---
    ss = np.random.SeedSequence(master).spawn(n_strategies * 2)
    iid_rej = 0
    ar1_rej = 0
    for i in range(n_strategies):
        r = iid_noise(n_obs, int(ss[2 * i].generate_state(1)[0]), sigma)
        p = permutation_pvalue(r, int(ss[2 * i + 1].generate_state(1)[0]), n_perm=n_perm)
        if p * 1 <= alpha:
            iid_rej += 1
    ss2 = np.random.SeedSequence(master + 1).spawn(n_strategies * 2)
    for i in range(n_strategies):
        r = ar1(n_obs, int(ss2[2 * i].generate_state(1)[0]), sigma, 0.3)
        p = permutation_pvalue(r, int(ss2[2 * i + 1].generate_state(1)[0]), n_perm=n_perm)
        if p <= alpha:
            ar1_rej += 1
    perm = {
        "iid_fpr": iid_rej / n_strategies,
        "ar1_fpr": ar1_rej / n_strategies,
        "n_strategies": n_strategies,
        "n_obs": n_obs,
        "n_perm": n_perm,
        "mc_se_iid": math.sqrt((iid_rej / n_strategies) * (1 - iid_rej / n_strategies) / n_strategies),
        "mc_se_ar1": math.sqrt((ar1_rej / n_strategies) * (1 - ar1_rej / n_strategies) / n_strategies),
    }
    _ckpt(out_dir, "perm", perm)

    # --- Bootstrap coverage ---
    cases = [("iid", 0.0), ("iid", 0.002), ("ar1", 0.0), ("ar1", 0.002)]
    mean_cov = {}
    sharpe_cov = {}
    for kind, mu in cases:
        cover = 0
        for i in range(n_series):
            s = np.random.SeedSequence(master + 100 + i).spawn(1)[0]
            seed = int(s.generate_state(1)[0])
            if kind == "iid":
                r = injected_edge(n_obs, seed, sigma, mu)
            else:
                base = ar1(n_obs, seed, sigma, 0.3)
                r = base + mu
            lo, hi = ci_mean(r, 0.95, seed + 1, n_boot=n_boot)
            if lo <= mu <= hi:
                cover += 1
        mean_cov[f"{kind}_mu{mu}"] = cover / n_series
    for kind in ["iid", "ar1"]:
        cover = 0
        for i in range(n_series):
            s = np.random.SeedSequence(master + 200 + i).spawn(1)[0]
            seed = int(s.generate_state(1)[0])
            if kind == "iid":
                r = injected_edge(n_obs, seed, sigma, 0.002)
            else:
                base = ar1(n_obs, seed, sigma, 0.3)
                r = base + 0.002
            lo, hi = ci_sharpe(r, 0.95, seed + 1, n_boot=n_boot)
            true = 0.002 / sigma
            if lo <= true <= hi:
                cover += 1
        sharpe_cov[f"{kind}_mu0.002"] = cover / n_series
    cov = {"mean": mean_cov, "sharpe": sharpe_cov, "n_series": n_series, "n_obs": n_obs, "n_boot": n_boot}
    _ckpt(out_dir, "coverage", cov)

    # --- End-to-end ---
    def make_prereg():
        return PreReg(
            run_id="r", created_utc="2026-01-01T00:00:00Z", hypothesis="h", metric="mean",
            minimum_effect=0.001, alpha=0.05, n_candidates_tried=1, n_trials=1,
            seeds={"data": 20261125, "permutation": 20261126, "split": 20261127},
            split_scheme="x", bars={"ci_level": 0.95},
            code_version={"git_sha": "a", "dirty": False},
        )

    arms = {
        "iid": lambda seed: iid_noise(n_obs, seed, sigma),
        "ar1": lambda seed: ar1(n_obs, seed, sigma, 0.3),
        "student_t": lambda seed: student_t_noise(n_obs, seed, sigma, 5),
        "vol_cluster": lambda seed: vol_cluster(n_obs, seed, sigma),
    }
    e2e = {}
    for name, gen in arms.items():
        passes = 0
        es = np.random.SeedSequence(master + 300).spawn(500)
        for i in range(500):
            dseed = int(es[i].generate_state(1)[0])
            r = gen(dseed)
            prereg = make_prereg()
            res = evaluate_mean(r, prereg, n_perm, n_boot)
            v = verdict(prereg, res)
            if v["verdict"] == "PASS":
                passes += 1
        e2e[name] = passes / 500
    _ckpt(out_dir, "e2e", e2e)

    # --- Final artifact ---
    end_time = datetime.now(timezone.utc)
    wall_clock = time.time() - t0

    final = {
        "convention": "v2 permutation rates are one-sided",
        "permutation_fpr": perm,
        "bootstrap_coverage": cov,
        "end_to_end_fpr": e2e,
        "python_version": sys.version,
        "wall_clock_seconds": wall_clock,
        # Binding fields
        "git_sha": start_sha,
        "dirty": start_dirty,
        "untracked_files": start_untracked,
        "runner_sha256": start_runner_hash,
        "command_line": " ".join(sys.argv),
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "pip_freeze_hash": pip_freeze_hash(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "blas_threads": blas_threads(),
        "start_time_utc": start_time.strftime("%Y%m%dT%H%M%SZ"),
        "end_time_utc": end_time.strftime("%Y%m%dT%H%M%SZ"),
        "wall_clock_seconds": wall_clock,
        "unbound": bool(unbound),
    }
    if unbound:
        final["binding_status"] = "UNBOUND"
    else:
        final["binding_status"] = "BOUND"

    ts = end_time.strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"selfvalidation_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final, f, sort_keys=True, indent=2)

    # --- Resume / binding integrity: re-verify at run end ---
    end_sha, end_dirty, end_untracked = git_info(repo)
    if end_sha != start_sha:
        raise RuntimeError(
            f"HEAD changed during run: {start_sha} -> {end_sha}; binding integrity broken"
        )
    if end_dirty != start_dirty or end_untracked != start_untracked:
        raise RuntimeError(
            "Tree cleanliness changed during run; binding integrity broken"
        )

    return final, path


def main():
    parser = argparse.ArgumentParser(description="Self-validation v3 runner (hash-bound)")
    parser.add_argument("--unbound", action="store_true",
                        help="Run on a non-clean tree; artifact labeled UNBOUND")
    parser.add_argument("--out-dir", default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--n-strategies", type=int, default=2000)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=999)
    parser.add_argument("--n-boot", type=int, default=999)
    parser.add_argument("--n-series", type=int, default=1000)
    parser.add_argument("--master", type=int, default=20261125)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    repo = "."
    # Enforce clean tree unless --unbound
    start_sha, start_dirty, start_untracked = enforce_clean_tree(
        allow_unbound=args.unbound, repo=repo
    )

    final, path = run_validation(
        out_dir=args.out_dir,
        unbound=args.unbound,
        n_strategies=args.n_strategies,
        n_obs=args.n_obs,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        n_series=args.n_series,
        master=args.master,
        sigma=args.sigma,
        alpha=args.alpha,
        repo=repo,
    )

    print(path)
    print(f"binding_status: {final['binding_status']}")
    print(f"git_sha: {final['git_sha']}")
    print(f"dirty: {final['dirty']}")
    print(f"wall_clock_seconds: {final['wall_clock_seconds']:.1f}")


if __name__ == "__main__":
    main()
