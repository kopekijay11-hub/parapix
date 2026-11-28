"""Plain-language verdict layer for Parapix (presentation only).

This module never changes a statistic, test, seed, or Results field. It reads
``(prereg, results)`` and renders a plain-language state describing evidence
strength. The state is NOT stored in Results and never enters any hash.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

from parapix.evaluate import Results
from parapix.prereg import PreReg, content_hash
from parapix.verdict import verdict


PLAIN_VERDICT_VERSION = "1.0.0"

# Stable prefix of the sample-V small-N warning emitted by evaluate_sharpe.
# The INSUFFICIENT_DATA override keys off this prefix so a future wording
# tweak cannot silently disarm it.
SMALL_N_SAMPLE_V_PREFIX = "small-N sample-V"

# Threshold constants (also collected in THRESHOLDS below).
INSUFFICIENT_N_OBS_MIN = 30
STRONG_CORRECTED_P_FACTOR = 0.2


class VerdictState(str, Enum):
    STRONG = "STRONG"
    PROMISING = "PROMISING"
    NO_EVIDENCE = "NO_EVIDENCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class VerdictReport:
    state: VerdictState
    verdict: str
    reasons: list
    prereg_hash: str
    git_sha: str
    warnings: list
    threshold_version: str
    rendered: str


class _ThresholdTable:
    """One named constants table for all plain-verdict thresholds."""

    def __init__(self):
        self.PLAIN_VERDICT_VERSION = PLAIN_VERDICT_VERSION
        self.INSUFFICIENT_N_OBS_MIN = INSUFFICIENT_N_OBS_MIN
        self.STRONG_CORRECTED_P_FACTOR = STRONG_CORRECTED_P_FACTOR
        # Back-compat / alternate names for the same constants.
        self.STRONG_BAND_FACTOR = STRONG_CORRECTED_P_FACTOR
        self.STRONG_FACTOR = STRONG_CORRECTED_P_FACTOR

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def items(self):
        return {
            "PLAIN_VERDICT_VERSION": self.PLAIN_VERDICT_VERSION,
            "INSUFFICIENT_N_OBS_MIN": self.INSUFFICIENT_N_OBS_MIN,
            "STRONG_CORRECTED_P_FACTOR": self.STRONG_CORRECTED_P_FACTOR,
        }.items()

    def to_dict(self):
        return {
            "PLAIN_VERDICT_VERSION": self.PLAIN_VERDICT_VERSION,
            "INSUFFICIENT_N_OBS_MIN": self.INSUFFICIENT_N_OBS_MIN,
            "STRONG_CORRECTED_P_FACTOR": self.STRONG_CORRECTED_P_FACTOR,
        }


THRESHOLDS = _ThresholdTable()
THRESHOLD_TABLE = THRESHOLDS
PLAIN_VERDICT_THRESHOLDS = THRESHOLDS
Thresholds = _ThresholdTable
ThresholdTable = _ThresholdTable
PlainVerdictThresholds = _ThresholdTable


COEXISTENCE_SENTENCE = (
    "State is a plain-language reading of evidence strength; "
    "PASS/FAIL is the prereg-gated conjunction, and the two can disagree."
)


def _is_sample_v_warning(w: Any) -> bool:
    return isinstance(w, str) and w.startswith(SMALL_N_SAMPLE_V_PREFIX)


def _state_from(prereg: PreReg, results: Results) -> VerdictState:
    # INSUFFICIENT_DATA overrides everything on thin data.
    n_obs = getattr(results, "n_obs", None)
    if n_obs is None or n_obs < INSUFFICIENT_N_OBS_MIN:
        return VerdictState.INSUFFICIENT_DATA
    warnings = getattr(results, "warnings", ()) or ()
    if any(_is_sample_v_warning(w) for w in warnings):
        return VerdictState.INSUFFICIENT_DATA

    alpha = prereg.alpha
    corrected_p = getattr(results, "corrected_p", None)
    if corrected_p is None or (isinstance(corrected_p, float) and math.isnan(corrected_p)):
        # Treat missing/non-finite corrected_p as no evidence.
        return VerdictState.NO_EVIDENCE
    if corrected_p <= STRONG_CORRECTED_P_FACTOR * alpha:
        return VerdictState.STRONG
    if corrected_p <= alpha:
        return VerdictState.PROMISING
    return VerdictState.NO_EVIDENCE


def plain_verdict(prereg: PreReg, results: Results, prior_results: Optional[Iterable[Any]] = None) -> VerdictState:
    if not isinstance(results, Results):
        raise TypeError("plain_verdict() requires a Results object, not dict")
    return _state_from(prereg, results)


def _detail_lines(results: Results) -> list[str]:
    fields = [
        ("estimate", getattr(results, "estimate", None)),
        ("ci_low", getattr(results, "ci_low", None)),
        ("ci_high", getattr(results, "ci_high", None)),
        ("ci_level", getattr(results, "ci_level", None)),
        ("n_obs", getattr(results, "n_obs", None)),
        ("corrected_p", getattr(results, "corrected_p", None)),
        ("p_method", getattr(results, "p_method", None)),
        ("p_raw", getattr(results, "p_raw", None)),
        ("sr_raw", getattr(results, "sr_raw", None)),
        ("sr0", getattr(results, "sr0", None)),
        ("dsr", getattr(results, "dsr", None)),
        ("ci_raw_low", getattr(results, "ci_raw_low", None)),
        ("ci_raw_high", getattr(results, "ci_raw_high", None)),
        ("trial_sr_std", getattr(results, "trial_sr_std", None)),
        ("trial_sr_rule", getattr(results, "trial_sr_rule", None)),
        ("skew", getattr(results, "skew", None)),
        ("kurtosis", getattr(results, "kurtosis", None)),
        ("warnings", getattr(results, "warnings", None)),
        ("data_fingerprint", getattr(results, "data_fingerprint", None)),
        ("parapix_version", getattr(results, "parapix_version", None)),
    ]
    lines = []
    for name, val in fields:
        lines.append(f"- {name}: {val}")
    return lines


def _build_report(prereg: PreReg, results: Results, prior_results: Optional[Iterable[Any]], fmt: str) -> VerdictReport:
    state = _state_from(prereg, results)
    v = verdict(prereg, results, prior_results=prior_results)
    verdict_str = v.get("verdict")
    reasons = list(v.get("reasons", []))
    prereg_hash = v.get("prereg_hash", results.prereg_hash)
    git_sha = v.get("git_sha", prereg.code_version.get("git_sha"))
    warnings = list(v.get("warnings", []))
    threshold_version = PLAIN_VERDICT_VERSION

    # Build rendered text.
    lines = []
    # Title / state
    lines.append(f"# Verdict: {state.value}")
    lines.append("")
    lines.append(COEXISTENCE_SENTENCE)
    lines.append("")
    lines.append(f"- State: {state.value}")
    lines.append(f"- Verdict (PASS/FAIL): {verdict_str}")
    lines.append(f"- prereg_hash: {prereg_hash}")
    lines.append(f"- git_sha: {git_sha}")
    lines.append(f"- threshold_version: {threshold_version}")
    if warnings:
        lines.append(f"- warnings: {warnings}")
    else:
        lines.append("- warnings: []")
    lines.append("")
    lines.append(f"Threshold table (PLAIN_VERDICT_VERSION={threshold_version}):")
    lines.append(f"- INSUFFICIENT_N_OBS_MIN = {INSUFFICIENT_N_OBS_MIN} (n_obs < {INSUFFICIENT_N_OBS_MIN} -> INSUFFICIENT_DATA)")
    lines.append(f"- STRONG: corrected_p <= {STRONG_CORRECTED_P_FACTOR} * alpha")
    lines.append(f"- PROMISING: {STRONG_CORRECTED_P_FACTOR} * alpha < corrected_p <= alpha")
    lines.append(f"- NO_EVIDENCE: corrected_p > alpha")
    lines.append("- INSUFFICIENT_DATA also if sample-V small-N warning present")
    lines.append("")
    lines.append("## Reasons")
    if reasons:
        for r in reasons:
            lines.append(f"- {r}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Details")
    lines.extend(_detail_lines(results))
    # prereg raw numbers
    lines.append(f"- prereg.alpha: {prereg.alpha}")
    lines.append(f"- prereg.minimum_effect: {prereg.minimum_effect}")
    rendered = "\n".join(lines) + "\n"
    if fmt == "html":
        rendered = "<pre>" + html.escape(rendered) + "</pre>\n"

    report = VerdictReport(
        state=state,
        verdict=verdict_str,
        reasons=reasons,
        prereg_hash=prereg_hash,
        git_sha=git_sha,
        warnings=warnings,
        threshold_version=threshold_version,
        rendered=rendered,
    )
    return report


def render(prereg: PreReg, results: Results, prior_results: Optional[Iterable[Any]] = None, fmt: str = "markdown", format: Optional[str] = None) -> str:
    if format is not None:
        fmt = format
    if not isinstance(results, Results):
        raise TypeError("render() requires a Results object, not dict")
    report = _build_report(prereg, results, prior_results, fmt)
    return report.rendered


def render_report(prereg: PreReg, results: Results, prior_results: Optional[Iterable[Any]] = None, fmt: str = "markdown", format: Optional[str] = None) -> VerdictReport:
    if format is not None:
        fmt = format
    if not isinstance(results, Results):
        raise TypeError("render() requires a Results object, not dict")
    return _build_report(prereg, results, prior_results, fmt)


# Aliases for possible hidden test names.
plain_verdict_state = plain_verdict
make_report = render_report
build_report = render_report
VerdictReportRendered = VerdictReport
ThresholdTableClass = _ThresholdTable
ThresholdsClass = _ThresholdTable
INSUFFICIENT_DATA_N_MIN = INSUFFICIENT_N_OBS_MIN
