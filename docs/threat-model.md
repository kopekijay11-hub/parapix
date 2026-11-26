# Threat Model: Pre-registration Lock

## What the prereg lock defends against

The pre-registration mechanism in Parapix provides **binding** between a `PreReg` and its `Results` via a SHA-256 content hash (`prereg_hash`). This defends against:

1. **Accidental drift**: A user modifies the pre-registration after seeing results (e.g., tweaking `minimum_effect`, `alpha`, `n_candidates_tried`, or `n_trials` to make a FAIL into a PASS). The hash mismatch causes `verdict()` to return FAIL with reason `"prereg_hash mismatch or missing"`.

2. **Result substitution**: Swapping results from one pre-registration into another. The hash binds each result to exactly one pre-registration.

3. **Silent parameter changes**: Any field change in the `PreReg` (hypothesis, metric, seeds, bars, code_version, etc.) changes the hash.

## What the prereg lock does NOT defend against

1. **Committer-date backdating**: Git commit timestamps (author/committer dates) are user-controlled. A determined user can write a pre-registration, backdate its commit, then write results, and the `check_prereg_precedes_results` check will pass. The lock stops *accidental* drift, not *intentional* history rewriting.

2. **Post-hoc pre-registration**: If a user runs experiments, picks the best outcome, and *then* writes a pre-registration matching that outcome (with matching `n_candidates_tried`/`n_trials` honestly reported), the hash will match and the verdict will reflect the reported parameters. The system cannot distinguish this from a genuine pre-commitment.

3. **Undercounting multiplicity**: `n_candidates_tried` (mean path) and `n_trials` (Sharpe path) are self-reported integers. If a user tries 100 strategies but reports 1, the Bonferroni/DSR correction is defeated. This is the fundamental weak link — no cryptographic binding can fix self-reported parameters.

4. **Non-exchangeable data**: The permutation test assumes exchangeability under the null. Feeding autocorrelated, volatility-clustered, or otherwise dependent returns inflates the false-positive rate. The prereg hash does not validate data properties.

5. **Code substitution**: `code_version.git_sha` is recorded but not enforced by the library at runtime. A user can report any SHA.

## Future hardening (not built)

- **External timestamping**: Anchoring the pre-registration hash to a public transparency log (e.g., RFC 3161 timestamp, blockchain anchor, or Sigstore) would make backdating detectable. This is explicitly **not** implemented in v0.x; the current design assumes a trusted committer or external process for timestamping.

- **Parameter validation**: Cross-referencing `n_candidates_tried`/`n_trials` against experiment logs or a trial registry would close the self-reporting gap. Out of scope for the core library.

## Summary

| Threat | Defended? | Mechanism |
|--------|-----------|-----------|
| Accidental prereg drift | Yes | `prereg_hash` in Results, verified by `verdict()` |
| Result substitution | Yes | `prereg_hash` binds 1:1 |
| Committer-date backdating | No | Git dates are user-controlled |
| Post-hoc preregistration | No | Cannot distinguish from genuine pre-commitment |
| Undercounting multiplicity | No | Self-reported integers |
| Non-exchangeable data | No | Not validated |
| Code substitution | No | `git_sha` recorded, not enforced |