import math
from dataclasses import dataclass
from typing import Literal, Optional

from parapix.bootstrap import ci_mean, ci_sharpe
from parapix.dsr import dsr, expected_max_sharpe, sharpe_stats
from parapix.permtest import permutation_pvalue
from parapix.prereg import PreReg, content_hash


@dataclass(frozen=True)
class Results:
    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_obs: int
    corrected_p: float
    p_method: Literal["signflip_permutation_bonferroni", "deflated_sharpe"]
    prereg_hash: str
    p_raw: Optional[float] = None
    sr_raw: Optional[float] = None
    sr0: Optional[float] = None
    dsr: Optional[float] = None
    ci_raw_low: Optional[float] = None
    ci_raw_high: Optional[float] = None
    trial_sr_std: Optional[float] = None
    trial_sr_rule: Optional[Literal["sample", "null_theoretical"]] = None
    skew: Optional[float] = None
    kurtosis: Optional[float] = None


def evaluate_mean(returns, prereg: PreReg, n_perm, n_boot):
    if n_perm < prereg.n_candidates_tried / prereg.alpha - 1:
        raise ValueError("resolution: n_perm too small for Bonferroni")
    import numpy as np

    estimate = float(np.asarray(returns, dtype=float).mean())
    ci_low, ci_high = ci_mean(
        returns, prereg.bars["ci_level"], prereg.seeds["permutation"], n_boot=n_boot
    )
    p_raw = permutation_pvalue(returns, prereg.seeds["permutation"], n_perm=n_perm)
    corrected_p = min(1.0, p_raw * prereg.n_candidates_tried)
    prereg_hash = content_hash(prereg)
    return Results(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=prereg.bars["ci_level"],
        n_obs=len(returns),
        corrected_p=corrected_p,
        p_method="signflip_permutation_bonferroni",
        prereg_hash=prereg_hash,
        p_raw=p_raw,
    )


def evaluate_sharpe(returns, prereg: PreReg, trial_sharpes=None):
    """Deflated Sharpe Ratio (DSR) evaluation.

    N comes from prereg.n_trials — never a free caller parameter.
    If trial_sharpes is given, its length must equal prereg.n_trials
    (else ValueError) and V = variance across them (ddof=1, N>1).
    If not given, V = 1 / (n_obs - 1) (null-theoretical) and
    trial_sr_rule = "null_theoretical"; if given, "sample".
    N=1: SR0 = 0 regardless.

    estimate = SR - SR0; ci_low / ci_high = ci_sharpe bounds each shifted by
    SR0 (SR0 treated as a constant). minimum_effect is in per-period Sharpe
    units on the NET-of-selection effect. At N=1 this reduces to the raw
    Sharpe evaluation.

    corrected_p = 1 - DSR (one-sided p-like quantity against H0: SR <= SR0;
    the SE is Wald, evaluated at the observed SR).

    Raw SR, raw CI, SR0, DSR all ride in provenance:
    {dsr, sr0, sr_raw, ci_raw_low, ci_raw_high, trial_sr_std,
    trial_sr_rule, p_method: "deflated_sharpe"}.

    ONE CORRECTION PER FAMILY: evaluate_mean's Bonferroni (n_candidates_tried)
    governs the mean path; DSR (n_trials) governs the Sharpe path. A prereg's
    metric selects exactly one path; nothing applies both.
    """
    import numpy as np

    x = np.asarray(returns, dtype=float)
    n_obs = x.shape[0]
    n_trials = prereg.n_trials
    level = prereg.bars["ci_level"]

    stats = sharpe_stats(x)
    sr_raw = stats["sr"]
    skew = stats["skew"]
    kurt = stats["kurtosis"]

    if trial_sharpes is not None:
        ts = np.asarray(trial_sharpes, dtype=float)
        if ts.shape[0] != n_trials:
            raise ValueError("trial_sharpes length must equal prereg.n_trials")
        if n_trials > 1:
            v = float(np.var(ts, ddof=1))
        else:
            v = 0.0
        trial_sr_std = math.sqrt(v) if v > 0 else 0.0
        trial_sr_rule = "sample"
    else:
        trial_sr_std = math.sqrt(1.0 / (n_obs - 1)) if n_obs > 1 else 0.0
        trial_sr_rule = "null_theoretical"

    sr0 = 0.0 if n_trials == 1 else expected_max_sharpe(n_trials, trial_sr_std)

    estimate = sr_raw - sr0
    try:
        ci_raw_low, ci_raw_high = ci_sharpe(
            x, level, prereg.seeds["permutation"]
        )
    except (ValueError, FloatingPointError):
        ci_raw_low = float("nan")
        ci_raw_high = float("nan")
    ci_low = ci_raw_low - sr0
    ci_high = ci_raw_high - sr0

    d = dsr(sr_raw, n_obs, skew, kurt, n_trials, trial_sr_std)
    corrected_p = 1.0 - d if not math.isnan(d) else float("nan")

    prereg_hash = content_hash(prereg)
    return Results(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=level,
        n_obs=n_obs,
        corrected_p=corrected_p,
        p_method="deflated_sharpe",
        prereg_hash=prereg_hash,
        sr_raw=sr_raw,
        sr0=sr0,
        dsr=d,
        ci_raw_low=ci_raw_low,
        ci_raw_high=ci_raw_high,
        trial_sr_std=trial_sr_std,
        trial_sr_rule=trial_sr_rule,
        skew=skew,
        kurtosis=kurt,
    )
