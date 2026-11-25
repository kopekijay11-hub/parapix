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


def student_t_noise(n, seed, sigma, df):
    """Zero-mean Student-t noise, df > 2, scaled so std dev is sigma (scale factor sigma * sqrt((df-2)/df))."""
    if df <= 2:
        raise ValueError("df must be > 2")
    rng = np.random.default_rng(seed)
    scale = sigma * np.sqrt((df - 2.0) / df)
    return rng.standard_t(df, size=n) * scale


def vol_cluster(n, seed, sigma, alpha_g=0.08, beta_g=0.90):
    """Zero-mean GARCH(1,1)-type volatility clustering. eps_t = sigma_t * z_t, sigma2_t = omega + alpha_g * eps2_{t-1} + beta_g * sigma2_{t-1}, omega = sigma^2 * (1 - alpha_g - beta_g), z iid normal. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=n)
    omega = sigma * sigma * (1.0 - alpha_g - beta_g)
    sigma2 = np.empty(n, dtype=float)
    eps = np.empty(n, dtype=float)
    sigma2[0] = sigma * sigma
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    for i in range(1, n):
        sigma2[i] = omega + alpha_g * eps[i - 1] * eps[i - 1] + beta_g * sigma2[i - 1]
        eps[i] = np.sqrt(sigma2[i]) * z[i]
    return eps
