from parapix.prereg import PreReg
from parapix.verdict import verdict

def base_prereg():
    return PreReg(run_id="r", created_utc="2026-01-01T00:00:00+00:00", hypothesis="h", metric="m",
                  minimum_effect=0.1, alpha=0.05, n_candidates_tried=1, n_trials=1,
                  seeds={"data": 1, "permutation": 2, "split": 3}, split_scheme="s",
                  bars={"ci_level": 0.95}, code_version={"git_sha": "abc", "dirty": False})

def test_pass():
    r = verdict(base_prereg(), {"estimate": 0.2, "ci_low": 0.11, "ci_high": 0.3, "ci_level": 0.95, "n_obs": 10, "corrected_p": 0.01})
    assert r["verdict"] == "PASS"

def test_fail():
    r = verdict(base_prereg(), {"estimate": 0.2, "ci_low": 0.05, "ci_high": 0.3, "ci_level": 0.95, "n_obs": 10, "corrected_p": 0.01})
    assert r["verdict"] == "FAIL"

def test_straddle():
    r = verdict(base_prereg(), {"estimate": 0.1, "ci_low": 0.05, "ci_high": 0.15, "ci_level": 0.95, "n_obs": 10, "corrected_p": 0.01})
    assert "CI straddles threshold (inconclusive)" in r["reasons"]

def test_exact_equality():
    r = verdict(base_prereg(), {"estimate": 0.2, "ci_low": 0.1, "ci_high": 0.3, "ci_level": 0.95, "n_obs": 10, "corrected_p": 0.01})
    assert r["verdict"] == "FAIL"

def test_corrected_p_missing():
    r = verdict(base_prereg(), {"estimate": 0.2, "ci_low": 0.11, "ci_high": 0.3, "ci_level": 0.95, "n_obs": 10, "corrected_p": None})
    assert r["verdict"] == "FAIL"

def test_purity():
    p = base_prereg()
    res = {"estimate": 0.2, "ci_low": 0.11, "ci_high": 0.3, "ci_level": 0.95, "n_obs": 10, "corrected_p": 0.01}
    import copy
    p0 = copy.deepcopy(p)
    r0 = verdict(p, res)
    r1 = verdict(p, res)
    assert r0 == r1
    assert p == p0
