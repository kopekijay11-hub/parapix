# Changelog

## 0.2.0 (unreleased)

### Breaking changes
- **Typed Results**: `evaluate_mean` and `evaluate_sharpe` now return a `Results` dataclass instead of a plain dict. `verdict()` requires a `Results` object and raises `TypeError` if passed a dict.
- **Verdict return**: `verdict()` now returns `{"verdict": "PASS|FAIL", "reasons": [...], "prereg_hash": str, "git_sha": str}`.

### New features
- **Prereg-hash binding**: `evaluate_mean`/`evaluate_sharpe` include `prereg_hash = content_hash(prereg)` in results. `verdict()` recomputes the hash and FAILs with `"prereg_hash mismatch or missing"` unless it matches exactly.
- **Verdict semantics fix (Sharpe path)**: The CI condition now tests the **raw-scale** CI (`ci_raw_low`, `ci_raw_high`) against `minimum_effect`, mirroring the mean path's logic. Deflation (SR₀ shift) lives only in `corrected_p` (via DSR). The deflated CI is reported for provenance but does not gate the verdict.

### Documentation
- README: Added "Conjunction semantics" section explaining mean vs Sharpe path logic; "How to misuse this" section; "API stability statement".
- New file `docs/threat-model.md`: What the prereg lock does/doesn't defend against.
- DSR assumptions updated: rho=0.3 arm read 0.0081 vs 0.05 nominal (WO-003 artifact) — correlated trials deflate less than independence implies.

### Tests
- New tests: hash match/mismatch/missing, dict rejection, Sharpe raw-CI pinning test, Results round-trip.
- All existing tests migrated to `Results` dataclass.

## Unreleased
- Foundation: repo, pre-registration machinery, verdict logic, seeded synthetic data, self-validation skeleton.
- Project renamed Parallax -> Parapix.