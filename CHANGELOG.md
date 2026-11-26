# Changelog

## 0.2.0 (unreleased)

### New features
- **Worked example (WO-007)**: `examples/worked_example.py` runs end-to-end on synthetic data — the best-of-N trap (FAIL), the prereg/DSR correction, a planted edge (PASS, true Sharpe 0.5), and the small-N sample-V warning. README "Worked example" section reproduces its output verbatim.

### Breaking changes
- **Typed Results**: `evaluate_mean` and `evaluate_sharpe` now return a `Results` dataclass instead of a plain dict. `verdict()` requires a `Results` object and raises `TypeError` if passed a dict.
- **Verdict return**: `verdict()` now returns `{"verdict": "PASS|FAIL", "reasons": [...], "prereg_hash": str, "git_sha": str}`.

### New features
- **Prereg-hash binding**: `evaluate_mean`/`evaluate_sharpe` include `prereg_hash = content_hash(prereg)` in results. `verdict()` recomputes the hash and FAILs with `"prereg_hash mismatch or missing"` unless it matches exactly.
- **Verdict semantics fix (Sharpe path)**: The CI condition now tests the **raw-scale** CI (`ci_raw_low`, `ci_raw_high`) against `minimum_effect`, mirroring the mean path's logic. Deflation (SR₀ shift) lives only in `corrected_p` (via DSR). The deflated CI is reported for provenance but does not gate the verdict.
- **Additive warnings pass-through (WO-003c B4)**: `Results` gains a `warnings: tuple = ()` field and `verdict()` echoes `list(results.warnings)` under a new `"warnings"` key on every return path (PASS, FAIL, hash-mismatch). Purely additive: the PASS/FAIL logic and all existing keys are unchanged; `warnings` never alters the verdict. This carries the WO-003c small-N sample-V caution (A5 ladder) from the evaluation layer to the caller.

### Documentation
- README: Added "Conjunction semantics" section explaining mean vs Sharpe path logic; "How to misuse this" section; "API stability statement".
- New file `docs/threat-model.md`: What the prereg lock does/doesn't defend against.
- DSR assumptions updated: rho=0.3 arm read 0.0081 vs 0.05 nominal (WO-003 artifact) — correlated trials deflate less than independence implies.
- **WO-003c Phase B (sample-V over-deflation)**: bound sweep (R=2M/cell, n_obs=1000, artifact `results/wo003c_20261126T135642Z.json`). Sample-V exceedance is ~13× below null-theoretical at N=10 (a/b=0.075), rising to 0.71 at N=250. A4 floor = N=250 (paired one-sided 99% bound on a/b ≥ 0.5); A5 warning ladder fires on the floor branch → a small-N sample-V caution ships (applies below N=250; a PASS is conservative, a FAIL is weak evidence). Power is saturated at the registered SR grid (no power-branch firing). B1 re-derivation matches the library to ≤1e-6; B2 non-normality correction ≈3.5e-5 abs / 0.11% of se.

### Tests
- New tests: hash match/mismatch/missing, dict rejection, Sharpe raw-CI pinning test, Results round-trip.
- All existing tests migrated to `Results` dataclass.

### CI
- GitHub Actions workflow (`.github/workflows/ci.yml`): suite runs on every push/PR; `tests-arch` job installs `arch==9.0.0` so the block-length comparison test runs and passes.

## Unreleased
- Foundation: repo, pre-registration machinery, verdict logic, seeded synthetic data, self-validation skeleton.
- Project renamed Parallax -> Parapix.