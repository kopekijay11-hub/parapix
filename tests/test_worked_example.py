import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("worked_example", ROOT / "examples" / "worked_example.py")
worked = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worked)


def test_worked_example_core_path():
    result = worked.run_example(seed=20261126, n_obs=300, n_boot=99, verbose=False)
    assert result["trap_verdict"] == "FAIL"
    assert result["planted_verdict"] == "PASS"
    assert result["warning_present"] is True
