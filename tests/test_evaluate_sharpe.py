import math

import numpy as np
import pytest
from scipy.stats import norm

from parapix.dsr import expected_max_sharpe
from parapix.evaluate import evaluate_sharpe
from parapix.prereg import PreReg
from parapix.verdict import verdict


def _make_prereg(metric="sharpe", n_trials=1, minimum_effect=0.0, alpha=0.05):
    return PreReg(
        run_id="r",
        created_utc="2026-01-01T00:00:00Z",
        hypothesis="h",
        metric=metric,
        minimum_effect=minimum_effect,
        alpha=alpha,
        n_candidates_tried=1,
        n_trials=n_trials,
        seeds={"data": 1, "permutation": 2, "split": 3},
        split_scheme="x",
        bars={"ci_level": 0.95},
        code_version={"git_sha": "a", "dirty": False},
    )


def test_evaluate_sharpe_known_pass_n1():
    # Strong injected Sharpe, N=1: reduces to raw Sharpe evaluation, should PASS
    rng = np.random.default_rng(7)
    mu = 0.05
    sigma = 0.01
    r = rng.normal(mu, sigma, size=5000)  # true SR = 5.0
    p = _make_prereg(metric="sharpe", n_trials=1, minimum_effect=0.0)
    res = evaluate_sharpe(r, p)
    assert res["sr0"] == 0.0
    assert res["dsr"] > 0.99  # very strong signal
    v = verdict(p, res)
    assert v["verdict"] == "PASS"


def test_evaluate_sharpe_known_fail_noise():
    # Pure noise, N=1: DSR ≈ 0.5, corrected_p ≈ 0.5 > alpha → FAIL
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.01, size=5000)
    p = _make_prereg(metric="sharpe", n_trials=1, minimum_effect=0.0)
    res = evaluate_sharpe(r, p)
    assert res["sr0"] == 0.0
    assert res["dsr"] < 0.95  # noise should not pass
    v = verdict(p, res)
    assert v["verdict"] == "FAIL"


def test_corrected_p_equals_one_minus_dsr():
    # corrected_p = 1 - DSR on a fixed case
    rng = np.random.default_rng(11)
    r = rng.normal(0.02, 0.01, size=2000)
    p = _make_prereg(metric="sharpe", n_trials=5, minimum_effect=0.0)
    res = evaluate_sharpe(r, p)
    assert abs(res["corrected_p"] - (1.0 - res["dsr"])) < 1e-12


def test_ci_shift_n1_vs_n_gt1():
    # Same series, N>1 vs N=1: estimate and CI differ by exactly SR0
    # Use mean=0.001, sigma=0.01 so sample SR ~ 0.1 (not saturated).
    rng = np.random.default_rng(13)
    r = rng.normal(0.001, 0.01, size=100)

    p_n1 = _make_prereg(metric="sharpe", n_trials=1, minimum_effect=0.0)
    res_n1 = evaluate_sharpe(r, p_n1)

    p_n10 = _make_prereg(metric="sharpe", n_trials=10, minimum_effect=0.0)
    res_n10 = evaluate_sharpe(r, p_n10)

    sr0 = res_n10["sr0"]
    assert sr0 > 0.0  # N=10: SR0 > 0
    assert res_n10["estimate"] == pytest.approx(res_n1["estimate"] - sr0, abs=1e-12)
    assert res_n10["ci_low"] == pytest.approx(res_n1["ci_low"] - sr0, abs=1e-12)
    assert res_n10["ci_high"] == pytest.approx(res_n1["ci_high"] - sr0, abs=1e-12)
    assert res_n10["dsr"] < res_n1["dsr"]  # deflation lowers DSR


def test_trial_sharpes_length_mismatch():
    p = _make_prereg(metric="sharpe", n_trials=5)
    r = np.random.default_rng(1).normal(0.0, 0.01, size=1000)
    with pytest.raises(ValueError, match="trial_sharpes length must equal prereg.n_trials"):
        evaluate_sharpe(r, p, trial_sharpes=[0.0, 0.0])  # length 2 != 5


def test_trial_sharpes_sample_rule():
    p = _make_prereg(metric="sharpe", n_trials=5)
    r = np.random.default_rng(2).normal(0.0, 0.01, size=1000)
    ts = [0.01, 0.02, -0.01, 0.0, 0.015]
    res = evaluate_sharpe(r, p, trial_sharpes=ts)
    assert res["trial_sr_rule"] == "sample"
    expected_std = float(np.std(np.asarray(ts), ddof=1))
    assert res["trial_sr_std"] == pytest.approx(expected_std, rel=1e-12)
    sr0_expected = expected_max_sharpe(5, expected_std)
    assert res["sr0"] == pytest.approx(sr0_expected, rel=1e-12)


def test_null_theoretical_rule():
    p = _make_prereg(metric="sharpe", n_trials=10)
    r = np.random.default_rng(4).normal(0.0, 0.01, size=2000)
    res = evaluate_sharpe(r, p)
    assert res["trial_sr_rule"] == "null_theoretical"
    expected_std = math.sqrt(1.0 / (2000 - 1))
    assert res["trial_sr_std"] == pytest.approx(expected_std, rel=1e-12)


def test_provenance_fields():
    p = _make_prereg(metric="sharpe", n_trials=10)
    r = np.random.default_rng(5).normal(0.0, 0.01, size=2000)
    res = evaluate_sharpe(r, p)
    for key in ["dsr", "sr0", "sr_raw", "ci_raw_low", "ci_raw_high",
                "trial_sr_std", "trial_sr_rule", "p_method"]:
        assert key in res, key
    assert res["p_method"] == "deflated_sharpe"


def test_nan_dsr_fails_verdict():
    # All-identical returns → sd=0 → sharpe_stats sr=NaN → dsr=NaN → corrected_p=NaN → verdict FAIL
    p = _make_prereg(metric="sharpe", n_trials=10)
    r = np.ones(1000) * 0.01  # zero dispersion
    res = evaluate_sharpe(r, p)
    assert math.isnan(res["corrected_p"]) or math.isnan(res["dsr"])
    v = verdict(p, res)
    assert v["verdict"] == "FAIL"
    assert "corrected_p missing or exceeds alpha" in v["reasons"]
