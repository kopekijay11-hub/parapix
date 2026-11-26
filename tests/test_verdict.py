from parapix.evaluate import Results
from parapix.prereg import PreReg, content_hash
from parapix.verdict import verdict


def base_prereg():
    return PreReg(
        run_id="r",
        created_utc="2026-01-01T00:00:00+00:00",
        hypothesis="h",
        metric="m",
        minimum_effect=0.1,
        alpha=0.05,
        n_candidates_tried=1,
        n_trials=1,
        seeds={"data": 1, "permutation": 2, "split": 3},
        split_scheme="s",
        bars={"ci_level": 0.95},
        code_version={"git_sha": "abc", "dirty": False},
    )


def make_results(**kw):
    base = dict(
        estimate=0.2,
        ci_low=0.11,
        ci_high=0.3,
        ci_level=0.95,
        n_obs=10,
        corrected_p=0.01,
        p_method="signflip_permutation_bonferroni",
        prereg_hash=content_hash(base_prereg()),
    )
    base.update(kw)
    return Results(**base)


def test_pass():
    r = verdict(base_prereg(), make_results())
    assert r["verdict"] == "PASS"


def test_fail():
    r = verdict(base_prereg(), make_results(ci_low=0.05))
    assert r["verdict"] == "FAIL"


def test_straddle():
    r = verdict(base_prereg(), make_results(estimate=0.1, ci_low=0.05, ci_high=0.15))
    assert "CI straddles threshold (inconclusive)" in r["reasons"]


def test_exact_equality():
    r = verdict(base_prereg(), make_results(ci_low=0.1))
    assert r["verdict"] == "FAIL"


def test_corrected_p_missing():
    r = verdict(base_prereg(), make_results(corrected_p=None))
    assert r["verdict"] == "FAIL"


def test_purity():
    p = base_prereg()
    res = make_results()
    import copy
    p0 = copy.deepcopy(p)
    r0 = verdict(p, res)
    r1 = verdict(p, res)
    assert r0 == r1
    assert p == p0


def test_rejects_dict():
    p = base_prereg()
    res = {"estimate": 0.2, "ci_low": 0.11, "ci_high": 0.3, "ci_level": 0.95, "n_obs": 10, "corrected_p": 0.01}
    try:
        verdict(p, res)
        assert False, "should have raised TypeError"
    except TypeError as e:
        assert "Results object" in str(e)


def test_hash_match():
    p = base_prereg()
    res = make_results()
    r = verdict(p, res)
    assert r["verdict"] == "PASS"
    assert r["prereg_hash"] == content_hash(p)
    assert r["git_sha"] == p.code_version["git_sha"]


def test_hash_mismatch():
    p = base_prereg()
    # Create results with wrong hash
    res = make_results(prereg_hash="wrong_hash")
    r = verdict(p, res)
    assert r["verdict"] == "FAIL"
    assert "prereg_hash mismatch or missing" in r["reasons"]
    assert r["prereg_hash"] == content_hash(p)
    assert r["git_sha"] == p.code_version["git_sha"]


def test_hash_missing():
    p = base_prereg()
    # Create results without prereg_hash (empty string)
    res = make_results(prereg_hash="")
    r = verdict(p, res)
    assert r["verdict"] == "FAIL"
    assert "prereg_hash mismatch or missing" in r["reasons"]


def test_sharpe_raw_ci_condition():
    """Pinning test: raw ci_low > min_effect but deflated ci_low < min_effect.
    Under new semantics, CI condition passes (uses raw CI); verdict hinges on corrected_p.
    """
    import numpy as np
    from parapix.evaluate import evaluate_sharpe

    # Construct returns where raw Sharpe CI is well above minimum_effect
    # but deflated CI (shifted by SR0) falls below.
    # Use sample trial_sharpes with high variance to get large SR0.
    rng = np.random.default_rng(42)
    r = rng.normal(0.015, 0.01, size=500)  # true SR ~ 1.5

    # High-variance trial sharpes to get large SR0
    trial_sharpes = [0.5, -0.3, 0.8, 0.1, -0.2, 0.6, -0.4, 0.3, 0.0, 0.2]  # std ~ 0.35

    p = PreReg(
        run_id="r",
        created_utc="2026-01-01T00:00:00+00:00",
        hypothesis="h",
        metric="sharpe",
        minimum_effect=1.0,  # min_effect between raw ci_low (~1.42) and deflated ci_low (~0.80)
        alpha=0.05,
        n_candidates_tried=1,
        n_trials=10,
        seeds={"data": 1, "permutation": 2, "split": 3},
        split_scheme="s",
        bars={"ci_level": 0.95},
        code_version={"git_sha": "abc", "dirty": False},
    )

    res = evaluate_sharpe(r, p, trial_sharpes=trial_sharpes)

    # Verify the pinning condition: raw ci_low > min_effect > deflated ci_low
    assert res.ci_raw_low > p.minimum_effect, f"raw ci_low {res.ci_raw_low} not > min_effect {p.minimum_effect}"
    assert res.ci_low < p.minimum_effect, f"deflated ci_low {res.ci_low} not < min_effect {p.minimum_effect}"

    # Under NEW semantics: verdict uses raw CI for CI condition -> PASS on CI
    # The verdict then hinges on corrected_p alone
    v = verdict(p, res)
    # CI condition passes (raw ci_low > min_effect), so only corrected_p matters
    # With strong signal, corrected_p should be small -> PASS
    assert v["verdict"] == "PASS", f"Expected PASS, got {v['verdict']}: {v['reasons']}"