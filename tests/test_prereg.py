import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone

import pytest

from parallax.prereg import PreReg, to_json, from_json, validate, content_hash, check_prereg_precedes_results


def make_prereg(**kw):
    base = dict(
        run_id="r1", created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis="h", metric="m", minimum_effect=0.1, alpha=0.05,
        n_candidates_tried=1, n_trials=1, seeds={"data": 1, "permutation": 2, "split": 3},
        split_scheme="s", bars={"ci_level": 0.95}, code_version={"git_sha": "abc", "dirty": False},
    )
    base.update(kw)
    return PreReg(**base)


def test_rejects_missing_field():
    with pytest.raises(TypeError):
        PreReg(run_id="r1", created_utc="2026-01-01T00:00:00+00:00", hypothesis="h", metric="m",
               minimum_effect=0.1, alpha=0.05, n_candidates_tried=1, n_trials=1,
               seeds={"data": 1, "permutation": 2, "split": 3}, split_scheme="s",
               bars={"ci_level": 0.95})


def test_rejects_alpha_le_0():
    with pytest.raises(ValueError):
        validate(make_prereg(alpha=0.0))


def test_rejects_alpha_ge_1():
    with pytest.raises(ValueError):
        validate(make_prereg(alpha=1.0))


def test_rejects_n_candidates():
    with pytest.raises(ValueError):
        validate(make_prereg(n_candidates_tried=0))


def test_rejects_empty_hypothesis():
    with pytest.raises(ValueError):
        validate(make_prereg(hypothesis=""))


def test_roundtrip():
    p = make_prereg()
    validate(p)
    p2 = from_json(to_json(p))
    assert asdict_eq(p, p2)


def asdict_eq(a, b):
    import dataclasses
    return dataclasses.asdict(a) == dataclasses.asdict(b)


def test_hash_changes():
    p = make_prereg()
    h1 = content_hash(p)
    p2 = make_prereg(hypothesis="h2")
    assert content_hash(p2) != h1


def test_precedence_fail():
    with tempfile.TemporaryDirectory() as d:
        repo = os.path.join(d, "repo")
        os.makedirs(repo)
        subprocess.run(["git", "-C", repo, "init"], check=True, capture_output=True)
        subprocess.run(["git", "-C", repo, "config", "user.email", "t@t.com"], check=True)
        subprocess.run(["git", "-C", repo, "config", "user.name", "t"], check=True)
        pre = os.path.join(repo, "pre.json")
        res = os.path.join(repo, "res.json")
        with open(pre, "w") as f:
            f.write(to_json(make_prereg()))
        with open(res, "w") as f:
            f.write("{}")
        env = dict(os.environ, GIT_AUTHOR_DATE="2026-01-01T00:00:00+00:00",
                   GIT_COMMITTER_DATE="2026-01-01T00:00:00+00:00")
        subprocess.run(["git", "-C", repo, "add", "res.json"], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-m", "res"], check=True, env=env, capture_output=True)
        env2 = dict(os.environ, GIT_AUTHOR_DATE="2026-01-01T00:01:00+00:00",
                    GIT_COMMITTER_DATE="2026-01-01T00:01:00+00:00")
        subprocess.run(["git", "-C", repo, "add", "pre.json"], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-m", "pre"], check=True, env=env2, capture_output=True)
        with pytest.raises(ValueError):
            check_prereg_precedes_results("pre.json", "res.json", repo)


def test_precedence_pass():
    with tempfile.TemporaryDirectory() as d:
        repo = os.path.join(d, "repo")
        os.makedirs(repo)
        subprocess.run(["git", "-C", repo, "init"], check=True, capture_output=True)
        subprocess.run(["git", "-C", repo, "config", "user.email", "t@t.com"], check=True)
        subprocess.run(["git", "-C", repo, "config", "user.name", "t"], check=True)
        pre = os.path.join(repo, "pre.json")
        res = os.path.join(repo, "res.json")
        with open(pre, "w") as f:
            f.write(to_json(make_prereg()))
        with open(res, "w") as f:
            f.write("{}")
        env = dict(os.environ, GIT_AUTHOR_DATE="2026-01-01T00:00:00+00:00",
                   GIT_COMMITTER_DATE="2026-01-01T00:00:00+00:00")
        subprocess.run(["git", "-C", repo, "add", "pre.json"], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-m", "pre"], check=True, env=env, capture_output=True)
        env2 = dict(os.environ, GIT_AUTHOR_DATE="2026-01-01T00:01:00+00:00",
                    GIT_COMMITTER_DATE="2026-01-01T00:01:00+00:00")
        subprocess.run(["git", "-C", repo, "add", "res.json"], check=True)
        subprocess.run(["git", "-C", repo, "commit", "-m", "res"], check=True, env=env2, capture_output=True)
        check_prereg_precedes_results("pre.json", "res.json", repo)
