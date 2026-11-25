import numpy as np


def permutation_pvalue(returns, seed, n_perm=9999):
    """One-sided sign-flip permutation test of a positive mean.

    Null: returns are symmetric about zero, signs independent of magnitudes.
    p = (1 + #{mean_perm >= mean_obs}) / (n_perm + 1).
    Sign flips do not preserve dependence; this buys robustness to non-normality, not autocorrelation.
    Rejection under skew can reflect asymmetry, not only a nonzero mean.
    """
    returns = np.asarray(returns, dtype=float)
    n = returns.shape[0]
    rng = np.random.default_rng(seed)
    obs = float(returns.mean())
    count = 0
    chunk = 1024
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
        perm_means = signs @ returns / n
        count += int(np.sum(perm_means >= obs))
    return (1 + count) / (n_perm + 1)
