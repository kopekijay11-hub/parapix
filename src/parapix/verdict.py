from parapix.evaluate import Results
from parapix.prereg import PreReg, content_hash


def verdict(prereg: PreReg, results: Results) -> dict:
    if not isinstance(results, Results):
        raise TypeError("verdict() requires a Results object, not dict")

    expected_hash = content_hash(prereg)
    if results.prereg_hash != expected_hash:
        return {
            "verdict": "FAIL",
            "reasons": ["prereg_hash mismatch or missing"],
            "prereg_hash": expected_hash,
            "git_sha": prereg.code_version["git_sha"],
            "warnings": list(results.warnings),
        }

    git_sha = prereg.code_version["git_sha"]

    if results.p_method == "deflated_sharpe":
        ci_low = results.ci_raw_low
        ci_high = results.ci_raw_high
    else:
        ci_low = results.ci_low
        ci_high = results.ci_high

    reasons = []
    min_effect = prereg.minimum_effect
    straddles = (
        ci_low is not None
        and ci_high is not None
        and ci_low <= min_effect <= ci_high
    )
    if straddles:
        reasons.append("CI straddles threshold (inconclusive)")
    elif not (ci_low is not None and ci_low > min_effect):
        reasons.append("ci_low not strictly greater than minimum_effect")
    corrected_p = results.corrected_p
    alpha = prereg.alpha
    if corrected_p is None or not (corrected_p <= alpha):
        reasons.append("corrected_p missing or exceeds alpha")
    ci_level = results.ci_level
    bars_level = prereg.bars.get("ci_level")
    if bars_level is None or ci_level is None or not (ci_level >= bars_level):
        reasons.append("ci_level below declared bar")
    if reasons:
        return {"verdict": "FAIL", "reasons": reasons, "prereg_hash": expected_hash, "git_sha": git_sha, "warnings": list(results.warnings)}
    return {"verdict": "PASS", "reasons": [], "prereg_hash": expected_hash, "git_sha": git_sha, "warnings": list(results.warnings)}