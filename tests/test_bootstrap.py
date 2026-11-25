import numpy as np
from parapix.bootstrap import ci_mean, ci_sharpe, _auto_block_length

def test_block_int():
    x = np.random.default_rng(0).normal(size=1000)
    bf, bi = _auto_block_length(x)
    assert isinstance(bi, int) and 1 <= bi <= len(x)

def test_ci_mean_deterministic():
    x = np.arange(100, dtype=float)
    a = ci_mean(x, 0.95, 1)
    b = ci_mean(x, 0.95, 1)
    assert a == b

def test_ci_sharpe_finite():
    x = np.arange(1, 101, dtype=float)
    lo, hi = ci_sharpe(x, 0.95, 1)
    assert np.isfinite(lo) and np.isfinite(hi)
