import numpy as np


def _auto_block_length(returns):
    x = np.asarray(returns, dtype=float)
    n = x.shape[0]
    eps = x - x.mean()
    b_max = int(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))
    kn = max(5, int(np.log10(n)))
    m_max = int(np.ceil(np.sqrt(n))) + kn
    cv = 2.0 * np.sqrt(np.log10(n) / n)
    acv = np.zeros(m_max + 1)
    abs_acorr = np.zeros(m_max + 1)
    opt_m = None
    for i in range(m_max + 1):
        if i == 0:
            cross_prod = float(eps @ eps)
            v1 = float(eps @ eps)
            v2 = float(eps @ eps)
        else:
            cross_prod = float(eps[i:] @ eps[: n - i])
            v1 = float(eps[i + 1 :] @ eps[i + 1 :])
            v2 = float(eps[: -(i + 1)] @ eps[: -(i + 1)])
        acv[i] = cross_prod / n
        denom = np.sqrt(v1 * v2)
        abs_acorr[i] = abs(cross_prod) / denom if denom > 0 else 0.0
        if i >= kn:
            if opt_m is None and np.all(abs_acorr[i - kn : i] < cv):
                opt_m = i - kn
    m = 2 * max(opt_m, 1) if opt_m is not None else m_max
    m = min(m, m_max)
    g = 0.0
    lr_acv = acv[0]
    for k in range(1, m + 1):
        lam = 1.0 if k / m <= 0.5 else 2.0 * (1.0 - k / m)
        g += 2.0 * lam * k * acv[k]
        lr_acv += 2.0 * lam * acv[k]
    d_cb = (4.0 / 3.0) * lr_acv * lr_acv
    b_float = ((2.0 * g * g) / d_cb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_float = min(b_float, float(b_max))
    b_int = max(1, int(np.floor(b_float)))
    return b_float, b_int


def ci_mean(returns, level, seed, n_boot=1999, block_length=None):
    returns = np.asarray(returns, dtype=float)
    n = returns.shape[0]
    if block_length is None:
        _, bl = _auto_block_length(returns)
    else:
        bl = int(block_length)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        nblocks = (n + bl - 1) // bl
        starts = rng.integers(0, n, size=nblocks)
        idx = np.concatenate([(s + np.arange(bl)) % n for s in starts])[:n]
        boots[b] = returns[idx].mean()
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    return float(np.percentile(boots, lo * 100)), float(np.percentile(boots, hi * 100))


def ci_sharpe(returns, level, seed, n_boot=1999, block_length=None):
    returns = np.asarray(returns, dtype=float)
    n = returns.shape[0]
    if block_length is None:
        _, bl = _auto_block_length(returns)
    else:
        bl = int(block_length)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        nblocks = (n + bl - 1) // bl
        starts = rng.integers(0, n, size=nblocks)
        idx = np.concatenate([(s + np.arange(bl)) % n for s in starts])[:n]
        samp = returns[idx]
        sd = samp.std(ddof=1)
        boots[b] = samp.mean() / sd if sd > 0 else np.nan
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    return float(np.percentile(boots, lo * 100)), float(np.percentile(boots, hi * 100))
