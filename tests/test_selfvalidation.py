import json
import math

from parapix.selfvalidate import (
    binding_fields,
    git_info,
    runner_sha256,
    verify_binding,
)


def test_git_info_returns_head_and_dirty():
    sha, dirty, untracked = git_info()
    assert len(sha) == 40 or sha == "UNKNOWN"
    assert isinstance(dirty, bool)
    assert isinstance(untracked, list)


def test_runner_sha256_matches_committed_file():
    import hashlib
    h = runner_sha256()
    assert len(h) == 64
    out = __import__("subprocess").run(
        ["git", "show", "HEAD:src/parapix/selfvalidate.py"],
        capture_output=True, text=True,
    ).stdout
    expected = hashlib.sha256(out.encode("utf-8")).hexdigest()
    assert h == expected


def test_binding_fields_present():
    fields = binding_fields()
    for key in ["git_sha", "dirty", "runner_sha256", "command_line",
                "python_version", "numpy_version", "scipy_version",
                "pip_freeze_hash", "PYTHONHASHSEED", "blas_threads"]:
        assert key in fields, key
    assert fields["dirty"] is False
    assert fields["git_sha"] != "UNKNOWN"


def test_verify_binding_passes_on_current_tree():
    fields = binding_fields()
    artifact = {
        "git_sha": fields["git_sha"],
        "runner_sha256": fields["runner_sha256"],
    }
    ok, reason = verify_binding(artifact)
    assert ok is True, reason


def test_verify_binding_fails_on_tampered_runner_hash():
    fields = binding_fields()
    artifact = {
        "git_sha": fields["git_sha"],
        "runner_sha256": "0" * 64,
    }
    ok, reason = verify_binding(artifact)
    assert ok is False
    assert "mismatch" in reason
