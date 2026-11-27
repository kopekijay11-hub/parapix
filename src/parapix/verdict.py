from typing import Any, Iterable, Optional

from parapix.evaluate import Results
from parapix.prereg import PreReg, content_hash


def _prior_attr(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _look_counter_warnings(results: Results, prior_results: Optional[Iterable[Any]]) -> list[str]:
    current_fp = getattr(results, "data_fingerprint", None)
    if not current_fp:
        return []
    hashes = {results.prereg_hash}
    for item in prior_results or ():
        if _prior_attr(item, "data_fingerprint") == current_fp:
            ph = _prior_attr(item, "prereg_hash")
            if ph is not None:
                hashes.add(ph)
    if len(hashes) > 1:
        return [f"this data has been evaluated under {len(hashes)} prereg hashes"]
    return []


def verdict(prereg: PreReg, results: Results, prior_results: Optional[Iterable[Any]] = None) -> dict:
    if not isinstance(results, Results):
        raise TypeError("verdict() requires a Results object, not dict")

    expected_hash = content_hash(prereg)
    git_sha = prereg.code_version["git_sha"]
    base_warnings = list(results.warnings) + _look_counter_warnings(results, prior_results)
    parapix_version = getattr(results, "parapix_version", None)

    if results.prereg_hash != expected_hash:
        return {
            "verdict": "FAIL",
            "reasons": ["prereg_hash mismatch or missing"],
            "prereg_hash": expected_hash,
            "git_sha": git_sha,
            "warnings": base_warnings,
            "parapix_version": parapix_version,
        }

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
        return {
            "verdict": "FAIL",
            "reasons": reasons,
            "prereg_hash": expected_hash,
            "git_sha": git_sha,
            "warnings": base_warnings,
            "parapix_version": parapix_version,
        }

    out = {
        "verdict": "PASS",
        "reasons": [],
        "prereg_hash": expected_hash,
        "git_sha": git_sha,
        "warnings": base_warnings,
        "parapix_version": parapix_version,
    }
    search_space = getattr(prereg, "search_space", "")
    if isinstance(search_space, str) and search_space.strip():
        out["search_space"] = f"search space: {search_space}"
    else:
        out["search_space"] = "search space: (not declared — declared N may understate the real search)"
    return out
