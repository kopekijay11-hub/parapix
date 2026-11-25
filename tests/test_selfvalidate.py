import json
import math
import os
import subprocess

import numpy as np
import pytest

from parapix.selfvalidate import (
    binding_fields,
    enforce_clean_tree,
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
    h = runner_sha256()
    assert len(h) == 64
    # Recompute independently
    import hashlib
    out = subprocess.run(
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
    # Build a synthetic artifact with the current tree's binding
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
        "runner_sha256": "0" * 64,  # tampered
    }
    ok, reason = verify_binding(artifact)
    assert ok is False
    assert "mismatch" in reason


def test_verify_binding_fails_on_unknown_sha():
    artifact = {"git_sha": "0" * 40, "runner_sha256": "0" * 64}
    ok, reason = verify_binding(artifact)
    assert ok is False


def test_enforce_clean_tree_passes_on_clean_tree():
    # Should not raise on a clean tree
    sha, dirty, untracked = enforce_clean_tree(allow_unbound=False)
    assert dirty is False
    assert untracked == []


def test_enforce_clean_tree_refuses_dirty_tree(tmp_path, monkeypatch):
    # Create a dirty file outside results/ and verify refusal
    dirty_file = os.path.join(tmp_path, "dirty_marker.txt")
    with open(dirty_file, "w") as f:
        f.write("dirty")

    # Monkeypatch git_info to simulate a dirty tree
    import parapix.selfvalidate as sv

    orig_git_info = sv.git_info
    sv.git_info = lambda repo=".": ("abc123" * 7, True, [])
    try:
        with pytest.raises(RuntimeError, match="Refusing to run"):
            sv.enforce_clean_tree(allow_unbound=False)
    finally:
        sv.git_info = orig_git_info


def test_enforce_clean_tree_allows_unbound(monkeypatch):
    import parapix.selfvalidate as sv

    orig_git_info = sv.git_info
    sv.git_info = lambda repo=".": ("abc123" * 7, True, ["junk.txt"])
    try:
        # Should not raise when allow_unbound=True
        sha, dirty, untracked = sv.enforce_clean_tree(allow_unbound=True)
        assert dirty is True
        assert untracked == ["junk.txt"]
    finally:
        sv.git_info = orig_git_info


def test_artifact_contains_all_binding_fields():
    # Minimal check: binding_fields() returns all required keys
    fields = binding_fields()
    required = ["git_sha", "dirty", "untracked_files", "runner_sha256",
                "command_line", "python_version", "numpy_version",
                "scipy_version", "pip_freeze_hash", "PYTHONHASHSEED",
                "blas_threads"]
    for key in required:
        assert key in fields, f"missing binding field: {key}"
