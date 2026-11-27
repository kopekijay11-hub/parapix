import numpy as np

from parapix.evaluate import Results, evaluate_mean
from parapix.prereg import PreReg, content_hash, data_fingerprint
from parapix.verdict import verdict


def base_prereg(**kw):
    base = dict(
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
    base.update(kw)
    return PreReg(**base)


def make_results(prereg=None, **kw):
    if prereg is None:
        prereg = base_prereg()
    base = dict(
        estimate=0.2,
        ci_low=0.11,
        ci_high=0.3,
        ci_level=0.95,
        n_obs=10,
        corrected_p=0.01,
        p_method="signflip_permutation_bonferroni",
        prereg_hash=content_hash(prereg),
    )
    base.update(kw)
    return Results(**base)


def test_fingerprint_identical_for_identical_data():
    data = [1.0, 2.0, 3.0, 4.0]
    assert data_fingerprint(data) == data_fingerprint(list(data))


def test_fingerprint_differs_for_different_data():
    assert data_fingerprint([1.0, 2.0, 3.0]) != data_fingerprint([1.0, 2.0, 4.0])


def test_search_space_declared_pass():
    p = base_prereg(search_space="19 null + 1 planted")
    r = verdict(p, make_results(prereg=p))
    assert r["verdict"] == "PASS"
    assert r["search_space"] == "search space: 19 null + 1 planted"


def test_search_space_undeclared_pass():
    p = base_prereg()
    r = verdict(p, make_results(prereg=p))
    assert r["verdict"] == "PASS"
    assert r["search_space"] == "search space: (not declared — declared N may understate the real search)"


def test_search_space_absent_on_fail():
    p = base_prereg()
    r = verdict(p, make_results(prereg=p, ci_low=0.05))
    assert r["verdict"] == "FAIL"
    assert "search_space" not in r


def test_look_counter_warns_on_second_prereg_hash():
    returns = [0.1, 0.2, 0.3, 0.4]
    fp = data_fingerprint(returns)
    p1 = base_prereg(hypothesis="h1")
    p2 = base_prereg(hypothesis="h2")
    res1 = make_results(prereg=p1, data_fingerprint=fp)
    res2 = make_results(prereg=p2, data_fingerprint=fp)
    r = verdict(p2, res2, prior_results=[res1])
    assert any("evaluated under 2 prereg hashes" in w for w in r["warnings"])


def test_look_counter_silent_on_same_prereg_hash():
    returns = [0.1, 0.2, 0.3, 0.4]
    fp = data_fingerprint(returns)
    p = base_prereg()
    res = make_results(prereg=p, data_fingerprint=fp)
    r = verdict(p, res, prior_results=[res])
    assert not any("evaluated under" in w for w in r["warnings"])


def test_parapix_version_present_in_results():
    r = make_results(parapix_version="0.3.0")
    assert r.parapix_version == "0.3.0"


def test_parapix_version_echoed_by_verdict():
    p = base_prereg()
    res = make_results(prereg=p, parapix_version="0.3.0")
    r = verdict(p, res)
    assert r["parapix_version"] == "0.3.0"


def test_evaluate_mean_stamps_fingerprint_and_version():
    p = base_prereg()
    res = evaluate_mean([0.1]*20, p, n_perm=1000, n_boot=10)
    assert res.data_fingerprint == data_fingerprint([0.1]*20)
    assert res.parapix_version is not None


def test_verdict_outcomes_unchanged():
    p = base_prereg()
    assert verdict(p, make_results(prereg=p))["verdict"] == "PASS"
    assert verdict(p, make_results(prereg=p, ci_low=0.05))["verdict"] == "FAIL"
