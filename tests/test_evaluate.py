import numpy as np
from parapix.evaluate import evaluate_mean
from parapix.prereg import PreReg

def make_prereg():
    return PreReg(
        run_id="r", created_utc="2026-01-01T00:00:00Z", hypothesis="h", metric="mean",
        minimum_effect=0.001, alpha=0.05, n_candidates_tried=1, n_trials=1,
        seeds={"data":1,"permutation":2,"split":3}, split_scheme="x",
        bars={"ci_level":0.95}, code_version={"git_sha":"a","dirty":False})

def test_resolution_refusal():
    p = make_prereg()
    try:
        evaluate_mean(np.zeros(100), p, 10, 10)
        raised=False
    except ValueError:
        raised=True
    assert raised

def test_end_to_end():
    p = make_prereg()
    x = np.ones(1000)*0.1
    res = evaluate_mean(x, p, 999, 100)
    from parapix.verdict import verdict
    v = verdict(p, res)
    assert v["verdict"] in ("PASS","FAIL")


def test_results_roundtrip():
    """Results -> serialized/attribute form -> equal Results."""
    p = make_prereg()
    x = np.ones(1000)*0.1
    res = evaluate_mean(x, p, 999, 100)
    
    # Serialize via attribute access (dataclass asdict-like)
    attrs = {
        "estimate": res.estimate,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
        "ci_level": res.ci_level,
        "n_obs": res.n_obs,
        "corrected_p": res.corrected_p,
        "p_method": res.p_method,
        "prereg_hash": res.prereg_hash,
        "p_raw": res.p_raw,
        "data_fingerprint": res.data_fingerprint,
        "parapix_version": res.parapix_version,
    }
    
    from parapix.evaluate import Results
    res2 = Results(**attrs)
    
    assert res == res2
    assert res.estimate == res2.estimate
    assert res.ci_low == res2.ci_low
    assert res.ci_high == res2.ci_high
    assert res.ci_level == res2.ci_level
    assert res.n_obs == res2.n_obs
    assert res.corrected_p == res2.corrected_p
    assert res.p_method == res2.p_method
    assert res.prereg_hash == res2.prereg_hash
    assert res.p_raw == res2.p_raw
    assert res.data_fingerprint == res2.data_fingerprint
    assert res.parapix_version == res2.parapix_version
