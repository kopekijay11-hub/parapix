import numpy as np
import pytest
from parapix.bootstrap import ci_mean, ci_sharpe, _auto_block_length


def test_block_length_uses_doubled_bandwidth():
    # Behavioral check that the truncation bandwidth is M = 2*m_hat (the first
    # lag where k_n consecutive autocorrelations fall below the band), not m_hat.
    # arch.optimal_block_length's "circular" column implements the same M = 2*m_hat
    # rule (arch/bootstrap/base.py:135). If parapix used M = m_hat instead, the
    # value would not match arch's circular column.
    arch = pytest.importorskip("arch.bootstrap")
    from arch.bootstrap import optimal_block_length
    n = 1000
    rng = np.random.default_rng(11)
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + eps[t]
    bf, _ = _auto_block_length(x)
    ref = float(optimal_block_length(x)["circular"].iloc[0])
    assert bf == pytest.approx(ref, rel=1e-9)


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
