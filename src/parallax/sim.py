import numpy as np


def iid_noise(n, seed, sigma):
    """Zero-mean iid noise. Units: sigma is std dev per period in return units."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=n)


def injected_edge(n, seed, sigma, mu):
    """Mean mu per period. Units: mu and sigma in return units per period."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=n)


def ar1(n, seed, sigma, phi):
    """Zero-mean AR(1) noise with autocorrelation phi. Units: sigma is innovation std dev per period."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    out = np.empty(n, dtype=float)
    out[0] = eps[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out
