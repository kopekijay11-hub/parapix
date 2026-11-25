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
