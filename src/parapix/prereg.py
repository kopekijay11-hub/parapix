import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class PreReg:
    run_id: str
    created_utc: str
    hypothesis: str
    metric: str
    minimum_effect: float
    alpha: float
    n_candidates_tried: int
    n_trials: int
    seeds: dict
    split_scheme: str
    bars: dict
    code_version: dict
    search_space: str = ""


def _canonical(prereg):
    d = asdict(prereg)
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def to_json(prereg):
    return json.dumps(asdict(prereg), sort_keys=True, ensure_ascii=False, indent=2)


def from_json(text):
    d = json.loads(text)
    return PreReg(**d)


def validate(prereg):
    if not isinstance(prereg.hypothesis, str) or not prereg.hypothesis.strip():
        raise ValueError("hypothesis must be non-empty")
    if not isinstance(prereg.metric, str) or not prereg.metric.strip():
        raise ValueError("metric must be non-empty")
    if not (0 < prereg.alpha < 1):
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if prereg.n_candidates_tried < 1:
        raise ValueError("n_candidates_tried must be >= 1")
    if prereg.n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if not isinstance(prereg.seeds, dict):
        raise ValueError("seeds must be a dict")
    for k in ("data", "permutation", "split"):
        if k not in prereg.seeds or not isinstance(prereg.seeds[k], int):
            raise ValueError(f"seeds must contain int {k}")
    if not isinstance(prereg.bars, dict):
        raise ValueError("bars must be a dict")
    if "ci_level" not in prereg.bars:
        raise ValueError("bars must contain ci_level")
    ci_level = prereg.bars["ci_level"]
    if not isinstance(ci_level, (int, float)) or isinstance(ci_level, bool):
        raise ValueError("bars.ci_level must be a number")
    if not (0 < ci_level < 1):
        raise ValueError("bars.ci_level must satisfy 0 < ci_level < 1")
    if not isinstance(prereg.code_version, dict):
        raise ValueError("code_version must be a dict")
    if "git_sha" not in prereg.code_version or not isinstance(prereg.code_version["git_sha"], str):
        raise ValueError("code_version.git_sha must be a str")
    if "dirty" not in prereg.code_version or not isinstance(prereg.code_version["dirty"], bool):
        raise ValueError("code_version.dirty must be a bool")
    if not isinstance(prereg.search_space, str):
        raise ValueError("search_space must be a str")
    try:
        datetime.fromisoformat(prereg.created_utc.replace("Z", "+00:00"))
    except Exception:
        raise ValueError("created_utc must be ISO 8601 UTC")
    if not isinstance(prereg.run_id, str) or not prereg.run_id.strip():
        raise ValueError("run_id must be non-empty")
    if not isinstance(prereg.split_scheme, str) or not prereg.split_scheme.strip():
        raise ValueError("split_scheme must be non-empty")



def data_fingerprint(returns):
    import numpy as np

    x = np.asarray(returns, dtype=np.float64).reshape(-1)
    h = hashlib.sha256()
    h.update(str(x.shape[0]).encode("ascii"))
    h.update(b":")
    h.update(x.tobytes())
    return h.hexdigest()


def content_hash(prereg):
    return hashlib.sha256(_canonical(prereg).encode("utf-8")).hexdigest()


def check_prereg_precedes_results(prereg_path, results_path, repo):
    import subprocess
    def first_commit_time(path):
        out = subprocess.run(
            ["git", "-C", repo, "log", "--diff-filter=A", "--follow", "--format=%ct", "--", path],
            capture_output=True, text=True,
        )
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        if not lines:
            raise ValueError(f"{path} is uncommitted")
        return int(lines[-1])
    t_pre = first_commit_time(prereg_path)
    t_res = first_commit_time(results_path)
    if t_res <= t_pre:
        raise ValueError(
            f"results committed before or at same time as prereg: prereg={t_pre} results={t_res}"
        )
