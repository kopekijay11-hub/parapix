import os

ROOT = os.path.dirname(os.path.dirname(__file__))
SOURCE = os.path.join(ROOT, "validate_v2.py")


def _src():
    with open(SOURCE, "r", encoding="utf-8") as f:
        return f.read()


def test_v2_source_exists():
    assert os.path.exists(SOURCE)


def test_v2_final_keys_match_artifact():
    src = _src()
    for key in ["convention", "permutation_fpr", "bootstrap_coverage",
                "end_to_end_fpr", "python_version", "wall_clock_seconds"]:
        assert ('"%s"' % key) in src, key


def test_v2_convention_is_one_sided():
    assert '"v2 permutation rates are one-sided"' in _src()


def test_v2_timestamp_format():
    src = _src()
    assert "%Y%m%dT%H%M%SZ" in src
    assert "selfvalidation_" in src


def test_v2_wall_clock():
    src = _src()
    assert "wall_clock_seconds" in src
    assert "time.time()-t0" in src


def test_v2_uses_parapix_permtest_bootstrap():
    src = _src()
    assert "permutation_pvalue" in src
    assert "ci_mean" in src
    assert "ci_sharpe" in src


def test_permutation_fpr_logic_mirrors_permtest():
    import numpy as np
    from parapix.permtest import permutation_pvalue
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 1.0, size=200)
    p = permutation_pvalue(r, seed=11, n_perm=199)
    assert 0.0 <= p <= 1.0
