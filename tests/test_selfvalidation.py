import json, glob, os
from parallax.selfvalidate import run

def test_deterministic():
    a = run(5, 50, 0.3, 0.05, 1)
    b = run(5, 50, 0.3, 0.05, 1)
    assert a == b

def test_keys():
    r = run(5, 50, 0.3, 0.05, 1)
    required = ["false_positive_rate_iid", "false_positive_rate_ar1", "n_strategies", "n_obs",
                "phi", "alpha", "seed", "numpy_version", "scipy_version", "pandas_version",
                "python_version", "git_sha", "dirty"]
    for k in required:
        assert k in r
