def verdict(prereg, results):
    reasons = []
    ci_low = results.get("ci_low")
    ci_high = results.get("ci_high")
    min_effect = prereg.minimum_effect
    straddles = (ci_low is not None and ci_high is not None
                 and ci_low <= min_effect <= ci_high)
    if straddles:
        reasons.append("CI straddles threshold (inconclusive)")
    elif not (ci_low is not None and ci_low > min_effect):
        reasons.append("ci_low not strictly greater than minimum_effect")
    corrected_p = results.get("corrected_p")
    alpha = prereg.alpha
    if corrected_p is None or not (corrected_p <= alpha):
        reasons.append("corrected_p missing or exceeds alpha")
    ci_level = results.get("ci_level")
    bars_level = prereg.bars.get("ci_level")
    if bars_level is None or ci_level is None or not (ci_level >= bars_level):
        reasons.append("ci_level below declared bar")
    if reasons:
        return {"verdict": "FAIL", "reasons": reasons}
    return {"verdict": "PASS", "reasons": []}
