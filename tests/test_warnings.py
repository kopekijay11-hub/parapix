"""WO-003c B4: additive Results.warnings + verdict pass-through."""

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
        estimate=0.2, ci_low=0.11, ci_high=0.3, ci_level=0.95, n_obs=10,
        corrected_p=0.01, p_method="signflip_permutation_bonferroni",
        prereg_hash=content_hash(base_prereg()),
    )
    base.update(kw)
    return Results(**base)


def test_warnings_default_empty_tuple():
    r = make_results()
    assert r.warnings == ()


def test_verdict_passes_warnings_on_pass():
    p = base_prereg()
    res = make_results(warnings=("small-N sample-V: PASS conservative",))
    v = verdict(p, res)
    assert v["verdict"] == "PASS"
    assert v["warnings"] == ["small-N sample-V: PASS conservative"]


def test_verdict_passes_warnings_on_fail():
    p = base_prereg()
    res = make_results(ci_low=0.05, warnings=("w1", "w2"))
    v = verdict(p, res)
    assert v["verdict"] == "FAIL"
    assert v["warnings"] == ["w1", "w2"]


def test_verdict_passes_warnings_on_hash_mismatch():
    p = base_prereg()
    res = make_results(prereg_hash="wrong", warnings=("w",))
    v = verdict(p, res)
    assert v["verdict"] == "FAIL"
    assert v["warnings"] == ["w"]


def test_verdict_empty_warnings_by_default():
    v = verdict(base_prereg(), make_results())
    assert v["warnings"] == []


def test_warnings_do_not_change_verdict():
    # Same numeric inputs, with and without warnings, must give the same verdict.
    p = base_prereg()
    v0 = verdict(p, make_results())
    v1 = verdict(p, make_results(warnings=("a", "b")))
    assert v0["verdict"] == v1["verdict"]
    assert v0["reasons"] == v1["reasons"]