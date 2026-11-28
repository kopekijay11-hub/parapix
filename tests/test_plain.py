"""Tests for the plain-language verdict layer (WO-011)."""

import dataclasses
import json

import numpy as np

from parapix.evaluate import Results, evaluate_sharpe
from parapix.plain import (
    PLAIN_VERDICT_VERSION,
    THRESHOLDS,
    VerdictState,
    plain_verdict,
    render,
)
from parapix.prereg import PreReg, content_hash


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
        n_obs=50,
        corrected_p=0.005,
        p_method="signflip_permutation_bonferroni",
        prereg_hash=content_hash(prereg),
    )
    base.update(kw)
    return Results(**base)


def _serialize(res):
    return json.dumps(dataclasses.asdict(res), sort_keys=True, default=str)


# ---------- golden / render tests for all four states ----------

def test_render_strong():
    p = base_prereg()
    res = make_results(prereg=p, corrected_p=0.005, n_obs=50)
    assert plain_verdict(p, res) == VerdictState.STRONG
    md = render(p, res)
    assert "Verdict: STRONG" in md
    assert f"threshold_version: {PLAIN_VERDICT_VERSION}" in md
    assert "State is a plain-language reading" in md
    assert content_hash(p) in md


def test_render_promising():
    p = base_prereg()
    res = make_results(prereg=p, corrected_p=0.03, n_obs=50)
    assert plain_verdict(p, res) == VerdictState.PROMISING
    md = render(p, res)
    assert "Verdict: PROMISING" in md
    assert f"PLAIN_VERDICT_VERSION={PLAIN_VERDICT_VERSION}" in md


def test_render_no_evidence():
    p = base_prereg()
    res = make_results(prereg=p, corrected_p=0.1, n_obs=50)
    assert plain_verdict(p, res) == VerdictState.NO_EVIDENCE
    md = render(p, res)
    assert "Verdict: NO_EVIDENCE" in md


def test_render_insufficient_data_thin_n():
    p = base_prereg()
    res = make_results(prereg=p, n_obs=20, corrected_p=0.005)
    assert plain_verdict(p, res) == VerdictState.INSUFFICIENT_DATA
    md = render(p, res)
    assert "Verdict: INSUFFICIENT_DATA" in md


# ---------- boundary tests ----------

def test_boundary_n_obs_29_30():
    p = base_prereg()
    r29 = make_results(prereg=p, n_obs=29, corrected_p=0.005)
    r30 = make_results(prereg=p, n_obs=30, corrected_p=0.005)
    assert plain_verdict(p, r29) == VerdictState.INSUFFICIENT_DATA
    assert plain_verdict(p, r30) == VerdictState.STRONG


def test_boundary_corrected_p_strong_band():
    p = base_prereg()
    alpha = p.alpha
    eps = 1e-12
    factor = THRESHOLDS.STRONG_CORRECTED_P_FACTOR
    # at 0.2*alpha (inclusive -> STRONG)
    r_at = make_results(prereg=p, corrected_p=factor * alpha, n_obs=50)
    assert plain_verdict(p, r_at) == VerdictState.STRONG
    # just above -> PROMISING
    r_above = make_results(prereg=p, corrected_p=factor * alpha + eps, n_obs=50)
    assert plain_verdict(p, r_above) == VerdictState.PROMISING
    # just below stays STRONG
    r_below = make_results(prereg=p, corrected_p=factor * alpha - eps, n_obs=50)
    assert plain_verdict(p, r_below) == VerdictState.STRONG


def test_boundary_corrected_p_alpha_band():
    p = base_prereg()
    alpha = p.alpha
    eps = 1e-12
    r_at = make_results(prereg=p, corrected_p=alpha, n_obs=50)
    assert plain_verdict(p, r_at) == VerdictState.PROMISING
    r_above = make_results(prereg=p, corrected_p=alpha + eps, n_obs=50)
    assert plain_verdict(p, r_above) == VerdictState.NO_EVIDENCE
    r_below = make_results(prereg=p, corrected_p=alpha - 1e-9, n_obs=50)
    assert plain_verdict(p, r_below) in (VerdictState.PROMISING, VerdictState.STRONG)


# ---------- override test using real sample-V warning ----------

def test_override_sample_v_real_warning():
    p = base_prereg(n_trials=5)
    rng = np.random.default_rng(42)
    r = rng.normal(0.0, 0.01, size=1000)
    trial_sharpes = [float(x) for x in rng.normal(0.0, 0.1, size=5)]
    res = evaluate_sharpe(r, p, trial_sharpes=trial_sharpes, n_boot=99)
    # real warning text present
    assert any(w.startswith("small-N sample-V") for w in res.warnings)
    # override fires even though corrected_p may be small
    assert plain_verdict(p, res) == VerdictState.INSUFFICIENT_DATA


# ---------- byte-identity ----------

def test_byte_identity_results_unchanged():
    p = base_prereg()
    res = make_results(prereg=p)
    before = _serialize(res)
    _ = plain_verdict(p, res)
    _ = render(p, res)
    after = _serialize(res)
    assert before == after


# ---------- determinism ----------

def test_render_determinism():
    p = base_prereg()
    res = make_results(prereg=p, corrected_p=0.03)
    a = render(p, res)
    b = render(p, res)
    assert a == b
    assert a.encode() == b.encode()


# ---------- coexistence: both disagreement directions ----------

def test_coexistence_strong_state_verdict_fail():
    p = base_prereg()
    # STRONG state (corrected_p <= 0.2*alpha) but verdict FAIL (ci_low <= min_effect)
    res = make_results(prereg=p, corrected_p=0.005, ci_low=0.05, n_obs=50)
    assert plain_verdict(p, res) == VerdictState.STRONG
    from parapix.verdict import verdict
    v = verdict(p, res)
    assert v["verdict"] == "FAIL"


def test_coexistence_middle_state_verdict_pass():
    p = base_prereg()
    # PROMISING (middle band) state but verdict PASS
    res = make_results(prereg=p, corrected_p=0.03, ci_low=0.11, n_obs=50)
    assert plain_verdict(p, res) == VerdictState.PROMISING
    from parapix.verdict import verdict
    v = verdict(p, res)
    assert v["verdict"] == "PASS"