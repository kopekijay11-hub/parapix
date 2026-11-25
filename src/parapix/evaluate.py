from parapix.bootstrap import ci_mean
from parapix.permtest import permutation_pvalue


def evaluate_mean(returns, prereg, n_perm, n_boot):
    if n_perm < prereg.n_candidates_tried / prereg.alpha - 1:
        raise ValueError("resolution: n_perm too small for Bonferroni")
    import numpy as np
    estimate = float(np.asarray(returns, dtype=float).mean())
    ci_low, ci_high = ci_mean(returns, prereg.bars["ci_level"], prereg.seeds["permutation"], n_boot=n_boot)
    p_raw = permutation_pvalue(returns, prereg.seeds["permutation"], n_perm=n_perm)
    corrected_p = min(1.0, p_raw * prereg.n_candidates_tried)
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": prereg.bars["ci_level"],
        "n_obs": len(returns),
        "corrected_p": corrected_p,
        "p_raw": p_raw,
        "p_method": "signflip_permutation_bonferroni",
    }
