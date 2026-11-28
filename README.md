Parapix v0.2.0 — a toolkit for testing the statistical significance of trading strategy returns. It provides block-bootstrap confidence intervals for mean and Sharpe ratio (with iid and dependent data), a permutation test for the null of zero skill, and the Deflated Sharpe Ratio (DSR) for multiple-testing correction.

## v0.1 validation bars

Every number below comes from a bound artifact with its settings recorded.

| Bar | Target | Measured | Settings (bound artifact) |
|-----|--------|----------|---------------------------|
| False-positive rate, iid null, nominal 0.05 — DSR | within 2 SE of nominal | **0.05035** | n_series=20000, n_obs=1000, sigma=0.01, master_seed=20261125, Phase B rider artifact: `results/selfvalidation_phaseb_20261126T061103Z.json` |
| False-positive rate, iid null, nominal 0.05 — permutation test | within 2 SE of nominal | **0.0575** | n_series=2000, n_obs=1000, n_perm=999, iid, WO-002 artifact: `results/selfvalidation_20261125T081205Z.json` |
| CI coverage, iid, nominal 0.95 | >= 0.93 | **0.940–0.9485** (across all five methods) | n_series=2000, n_obs=1000, sigma=0.01, phi=0.3, n_boot=999, master_seed=20261125, artifact: same as above |
| Power floor: true Sharpe 0.5 at n_obs=1000 | PASSes at >= 0.99 | **1.000** | Power curve artifact: `results/selfvalidation_phaseb_power_20261126T064743Z.json` |
| Minimum detectable effect (MDE) | — | **0.10** (smallest grid SR with PASS >= 0.80) | iid, N=1, CI-rejection power (PASS fraction averaged across percentile, bootstrap-SE normal, studentized, BCa-block, HAC delta-method), n_obs=1000, sigma=0.01, n_series=1000 per grid point, SR grid {0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}, level=0.95, n_boot=999, master_seed=20261126 |

**Documented limitation (not a bar):** CI coverage under AR(1) phi=0.3 dependence is 0.92–0.93 at nominal 0.95 across all five methods (Phase B artifact, same as above). This is an expected limitation of block bootstrap with automatic block length at this n_obs; not a library defect.

## Power curve detail

CI-rejection power: PASS fraction per grid point, averaged across the five CI methods (percentile, bootstrap-SE normal, studentized, BCa-block, HAC delta-method):

| True Sharpe | PASS fraction |
|-------------|---------------|
| 0.00 | 0.0502 |
| 0.05 | 0.3642 |
| 0.10 | 0.8846 |
| 0.15 | 0.9986 |
| 0.20 | 1.0000 |
| 0.30 | 1.0000 |
| 0.50 | 1.0000 |

Monotone non-decreasing ✓. Size at SR 0.00 = 0.0502 (within 2 SE of 0.05) ✓. Power at SR 0.50 = 1.000 (>= 0.99) ✓. MDE = 0.10.

The power-floor bar itself rests on the WO-003 positive control (evaluate_sharpe, true SR 0.5, PASS 1.000).

## Scope

Parapix tests statistical significance, not backtest correctness; it cannot detect lookahead bias, survivorship bias, unrealistic fills, or omitted costs.

## Known gaps

- Timing permutation (circular shift of signal vs returns)
- Block sign-flip / wild bootstrap
- Walk-forward splitter (cut for v0.1)

## Deflated Sharpe Ratio (WO-003)

The Deflated Sharpe Ratio (DSR) corrects for two sources of performance inflation: selection bias across multiple trials and non-Normality of returns. It is a Probabilistic Sharpe Ratio (PSR) whose benchmark is SR₀, the expected maximum Sharpe of N independent zero-skill trials:

    SR₀ = √V · ((1 − γ_E) · Φ⁻¹(1 − 1/N) + γ_E · Φ⁻¹(1 − 1/(N·e)))
    DSR  = Φ((SR − SR₀) · √(n−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²))

where V is the variance across trials of estimated per-period Sharpes, γ_E ≈ 0.5772 is the Euler-Mascheroni constant, γ₃ is skewness, γ₄ is raw kurtosis (3 for a Normal), and n is the number of return observations.

Assumptions, plainly stated:
- Trials are independent draws from a common distribution. Correlated trials (e.g. a grid search over a smooth parameter surface) violate this; the effective N is smaller than the raw count. The rho=0.3 arm in the DSR calibration rider read 0.0081 vs the 0.05 nominal (WO-003 bound artifact) — correlated trials deflate **less** than independence implies, with the effective-N caveat.
- Returns are iid. The analytic SE understates the true SE under autocorrelation: at AR(1) φ=0.3 the variance inflation is ~1.857×, so the SE is understated by ~1.36× (first order). DSR against SR₀ is therefore not a size-calibrated FWER test under dependence.

Sources: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014), Eq. (1)-(2); Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier" (2012), Eq. (11). Implementation: src/parapix/dsr.py.

## Permutation test assumption

The sign-flip permutation test assumes exchangeability: under the null the returns are symmetric about zero and signs are independent of magnitudes. It is robust to non-normality, not to autocorrelation/dependence. Under serial dependence its false-positive rate inflates, because permuting the series destroys the dependence structure the null relies on; a nominal level (e.g. 5%) can therefore be exceeded under dependence. See src/parapix/permtest.py.

## Conjunction semantics (v0.2)

**Mean path:** The confidence interval is **uncorrected for multiplicity** (n_candidates_tried) and guards effect size; the Bonferroni correction lives entirely in `corrected_p`. A PASS requires both `ci_low > minimum_effect` AND `corrected_p <= alpha`.

**Sharpe path:** The deflation (SR₀ shift) lives entirely in `corrected_p` (via DSR). The CI condition tests the **raw-scale** interval (`ci_raw_low`, `ci_raw_high`) against `minimum_effect`. A PASS requires both `ci_raw_low > minimum_effect` AND `corrected_p <= alpha`. The deflated CI (`ci_low`, `ci_high`) is reported for provenance but does not gate the verdict.

## How to misuse this

- **Peeking before registering:** If you look at results before writing the pre-registration, the n_candidates_tried and n_trials you report are fictitious. The prereg hash binds results to a prereg, but it cannot detect that the prereg was written *after* the fact.
- **Undercounting n_candidates_tried / n_trials:** These are self-reported. If you tried 50 strategies but report 1, the Bonferroni/DSR correction is meaningless. This is the real weak link.
- **Feeding non-exchangeable returns to the permutation test:** The sign-flip test assumes exchangeability under the null. Autocorrelated returns, volatility clustering, or any dependence structure will inflate the false-positive rate.
- **Ignoring the effective-N caveat:** Correlated trials (e.g., a smooth grid search) have an effective N smaller than the raw count. Reporting the raw count overstates the correction.

## API stability statement

**v0.x:** The pre-registration / verdict contracts (PreReg, Results, verdict() return) are the stable core. Other surfaces (bootstrap internals, simulation helpers, self-validation runner) may change without deprecation cycles. All breaking changes are recorded in the CHANGELOG.

## WO-003b Phase B: Sharpe CI Under Dependence — Coverage Reconciliation

### (a) Corrected AR(1) Sharpe coverage (supersedes 0.925 figure)

The previously reported 0.925 AR(1) Sharpe coverage figure was measured against the wrong truth: the generator's `sigma` is the innovation std dev, but `ci_sharpe` computes Sharpe against the marginal std dev. The old truth basis (`true = mu / sigma`) overstated the true Sharpe by ~4.8% at phi=0.3.

Corrected basis: true Sharpe = mu * sqrt(1-phi^2) / sigma (marginal-sd basis). Fix committed in src/parapix/selfvalidate.py (commit 2ee7405: "WO-003b: fix AR(1) Sharpe truth to marginal-sd basis (validation defect)").

Bound artifact: results/selfvalidation_phaseb_20261126T061103Z.json
Settings: n_series=2000, n_obs=1000, sigma=0.01, phi=0.3, n_boot=999, master_seed=20261125, levels={0.80,0.90,0.95,0.99}, runner=tools/run_phase_b.py (commit d6583cb), git_sha=d6583cb, dirty=false.

Results at 0.95 level (sharpe_ar1_mu0.002, true=0.190788):
- percentile: 0.9275, bootstrap-SE normal: 0.9290, HAC delta: 0.9335, BCa-block: 0.9285, studentized: 0.9275
All methods under-cover vs 0.95 nominal under AR(1) dependence — EXPECTED LIMITATION, not a library defect. iid configs cover near nominal (0.941-0.9485 at 0.95).

### (b) Reconciled CI coverage table

Shared-draws design: one bootstrap resample set (n_boot=999) per series; percentile, bootstrap-SE normal, studentized, BCa-block, and HAC delta-method all derived from the same resample set. HAC delta-method is analytic (no resampling). Settings: n_series=2000, n_obs=1000, sigma=0.01, phi=0.3, n_boot=999, master_seed=20261125. Coverage at each level; full per-series data in the bound artifact.

Coverage at 0.95 level (true values: mean_iid=mu, mean_ar1=mu, sharpe_iid=mu/sigma, sharpe_ar1=mu*sqrt(1-phi^2)/sigma):

| arm | true | percentile | bootstrap_se_normal | studentized | bca_block | hac_delta |
|-----|------|------------|---------------------|-------------|-----------|-----------|
| mean_iid_mu0.0 | 0.000000 | 0.9440 | 0.9435 | 0.9415 | 0.9410 | 0.9400 |
| mean_iid_mu0.002 | 0.002000 | 0.9440 | 0.9435 | 0.9415 | 0.9410 | 0.9400 |
| mean_ar1_mu0.0 | 0.000000 | 0.9215 | 0.9225 | 0.9235 | 0.9205 | 0.9300 |
| mean_ar1_mu0.002 | 0.002000 | 0.9215 | 0.9225 | 0.9235 | 0.9205 | 0.9300 |
| sharpe_iid_mu0.0 | 0.000000 | 0.9440 | 0.9450 | 0.9425 | 0.9415 | 0.9405 |
| sharpe_iid_mu0.002 | 0.200000 | 0.9475 | 0.9485 | 0.9425 | 0.9455 | 0.9435 |
| sharpe_ar1_mu0.0 | 0.000000 | 0.9215 | 0.9245 | 0.9245 | 0.9220 | 0.9300 |
| sharpe_ar1_mu0.002 | 0.190788 | 0.9275 | 0.9290 | 0.9275 | 0.9285 | 0.9335 |

Mean interval width at 0.95 (same arm order):
0.001231 / 0.001231 / 0.001678 / 0.001678 / 0.123356 / 0.124579 / 0.160531 / 0.161494

McNemar discordant counts and studentized degenerate-resample fraction (0.0000 for all arms) are in the bound artifact. No LIBRARY DEFECT found; all under-coverage under dependence is classified as EXPECTED LIMITATION.

### (c) DSR small-N warning

DSR calibration rider: iid null + AR(1) null, N=1, n_series=20000, n_obs=1000, sigma=0.01, phi=0.3, trial_sr_std=1/sqrt(n_obs-1), trial_sr_rule=null_theoretical, bound artifact above.

Results (fraction DSR > 0.95 vs prediction):
- iid: 0.05035 (prediction 0.05) — size-calibrated under iid null
- AR(1): 0.11525 (prediction ~0.114) — over-rejects under dependence, consistent with variance inflation ~1.857x at phi=0.3 (SE understated ~1.36x)

Warning: the sample-V estimate requires N>1 and is unstable for small N. The null-theoretical V (1/(n_obs-1)) is the recommended default when N is small; DSR against SR0 is not a size-calibrated FWER test under dependence.

## WO-003c Phase B: Sample-V Over-Deflation

The DSR's trial-variance estimate V can be taken two ways: the **null-theoretical** V = 1/(n_obs−1) (a fixed constant), or the **sample** V = variance across the N trial Sharpes (ddof=1). The sample V is itself an estimate and, at small N, is a noisy, *deflated* estimate of the true trial-SR dispersion. This section measures the finite-sample consequence with a bound simulation.

Settings (bound artifact `results/wo003c_20261126T135642Z.json`, git_sha `cf7d51f`, dirty=false, BOUND): n_obs=1000, sigma=√(1/999)=0.03164, alpha=0.05, DSR>0.95 (z=1.644854), N grid {5,10,20,50,100,250}; R=2,000,000 per independent cell, 200,000 per rho/power cell, 400,000 return-series bridge, 10,000 per pass-fraction cell (n_boot=99, diagnostic); master_seed=20261126; 8 workers, 3.71 h wall. The driver `tools/sim_wo003c.py` calls the library `expected_max_sharpe`/`psr` and never re-derives the DSR (the `psr`/`dsr` linearity identity is asserted at start-up).

### (B1) Library formula pinned + independent re-derivation

`src/parapix/dsr.py` sha256 `d63517b0…`, scipy 1.18.1. `tools/verify_wo003c_b1b2.py` re-implements `expected_max_sharpe`/`psr`/`dsr` from the paper equations in a fresh process (own normal CDF/PPF, no `parapix.dsr` in the re-derivation path) and matches the library to **0.0 (≤1e-6) over 29,408 grid points**.

### (B2) Non-normality / estimated-statistic correction (direct, no dsr call)

Computed by direct threshold/root evaluation (no `dsr`, no Monte Carlo). The exact denominator (1 + ½·SR²) vs the linear (denominator = 1) threshold shifts the exceedance by ≈ **−3.5e-5 abs** at N=250, i.e. **≈0.11% of the per-trial SR std** se=0.03164 — matching the registered 3.8e-5 / 0.12% pin.

### (B3) Independent-trials sweep — sample-V over-deflation

Exceedance P(DSR>0.95) under the iid null; (a) sample-V, (b) null-theoretical, (d) leave-selected-out V; corr = corr(SR_max, sample std).

| N | (a) sample-V | (b) null-theo | (d) leave-out | (a)/(b) | corr(SR_max, std) |
|---|---|---|---|---|---|
| 5 | 0.001147 | 0.011499 | 0.029416 | 0.100 | +0.631 |
| 10 | 0.000490 | 0.006555 | 0.014908 | 0.075 | +0.626 |
| 20 | 0.000608 | 0.004014 | 0.007836 | 0.151 | +0.581 |
| 50 | 0.000795 | 0.002283 | 0.003554 | 0.348 | +0.490 |
| 100 | 0.000788 | 0.001560 | 0.002056 | 0.505 | +0.416 |
| 250 | 0.000661 | 0.000931 | 0.001138 | 0.711 | +0.323 |

(b) reproduces the Phase A closed form. (a)/(b) rises from ~0.08 at N=10 to ~0.71 at N=250: sample-V over-deflation is severe at small N and vanishes by N≈250. Decomposition (b)/(a) = [(b)/(c)]·[(c)/(a)]: the two factors are reported separately — (b)/(a) = 13.4 at N=10 (10.0/6.6/2.9/2.0/1.4 at N=5/20/50/100/250). (a) at N=5 (0.00115) sits marginally above the Phase A band [1e-4,1e-3]; all other (a) cells and the (d) N=10 cell (0.0149) are inside their registered ranges. The mechanism correlation is **higher than Phase A's 0.30–0.45 (N=10) / 0.15–0.25 (N=50) guess** (measured +0.63 / +0.49) but falls with N as predicted — reported as a falsified-high hand estimate, not a defect.

### (A2) Correlated trials (shared factor) — sample-V vs oracle-V

Sample-V exceedance vs the oracle comparator V=(1−rho)sigma² (the wrong benchmark for correlated trials):

| N | rho | sample-V | oracle-V |
|---|---|---|---|
| 10 | 0.3 | 0.00751 | 0.01412 |
| 10 | 0.6 | 0.02422 | 0.02782 |
| 10 | 0.9 | 0.04369 | 0.04378 |
| 50 | 0.3 | 0.00630 | 0.00871 |
| 50 | 0.6 | 0.02235 | 0.02396 |
| 50 | 0.9 | 0.04277 | 0.04284 |

All six sample-V cells are inside their Phase A ranges. For correlated trials the null-theoretical V is the wrong benchmark (oracle-V ≠ sample-V except as rho→1).

### (B5) Power cost and conjunction

DSR-only power (theoretical-V minus sample-V) at the registered SR grid {0.10,0.15,0.20} × N {10,50,100,250} is **saturated at ≈1.0 for both rules (gap ≈0)** at n_obs=1000 — the A3 hand-predictions (0.48/0.94/0.998) are falsified; the power branch of the warning ladder does not fire. The evaluate_sharpe conjunction pass-fraction is likewise 1.000 at N∈{10,50}, SR∈{0.10,0.15,0.20}.

### (B6) Convergence and budget

Return-series bridge: N=10 sample-V exceedance at 2M (0.0004895) vs 400k (0.00055), z=−1.50 — agree within 2σ. Counts/seeds are the registered A6 values; total wall 3.71 h on 8 workers.

### (A4) Floor rule and (A5) warning ladder

A4 floor (paired common-random-number draws, one-sided 99% lower bound on (a)/(b) ≥ 0.5): **floor N = 250** (lwr99 0.680 at N=250; 0.482 at N=100 is below 0.5), gap 150 to the next grid point below. A5 ladder: the floor branch fires → **a warnings entry ships, applying below N=250**; direction: *under sample-V at small N a PASS is conservative and a FAIL is weak evidence.* The power branch does not fire. The `Results.warnings` / `verdict()` pass-through (v0.2, B4) is the channel that carries this caution to callers.
## Reading the verdict (plain-language layer)

Parapix also ships a presentation-only layer (`parapix.plain`) that maps a
`(prereg, results)` pair to one of four evidence-strength states: `STRONG`,
`PROMISING`, `NO_EVIDENCE`, `INSUFFICIENT_DATA`. It never changes any
statistic, test, seed, or Results field and never enters a hash.

- `INSUFFICIENT_DATA` overrides every other state on thin data (n_obs < 30, or
  the sample-V small-N warning is present).
- Otherwise the state bands on `corrected_p` vs `prereg.alpha`:
  `STRONG` if `corrected_p <= 0.2 * alpha`, `PROMISING` if
  `0.2 * alpha < corrected_p <= alpha`, `NO_EVIDENCE` if
  `corrected_p > alpha`.
- All thresholds live in one table, `parapix.plain.THRESHOLDS`, versioned
  `PLAIN_VERDICT_VERSION = "1.0.0"`, printed in every rendered report.
- Raw numbers remain under a `## Details` section; the verdict never replaces them.
- State and PASS/FAIL can disagree: "State is a plain-language reading of
  evidence strength; PASS/FAIL is the prereg-gated conjunction, and the two
  can disagree."

Usage:

    from parapix.plain import plain_verdict, render
    state = plain_verdict(prereg, results)
    report = render(prereg, results)   # markdown, includes the threshold table

## Worked example

Run:

    python examples/worked_example.py

What it shows: a deterministic, synthetic (no network) end-to-end pass over Parapix —
the uncorrected best-of-N trap, the prereg/DSR correction that rejects it, a planted
edge that passes, and the small-N sample-V warning.

Output (verbatim from an actual run of the committed script):

```
========================================================================
Parapix worked example (synthetic data)
planted true Sharpe = 0.5
N candidates = 20 (19 null + 1 planted), n_obs = 1000
========================================================================

(a) THE TRAP
    best-of-19 nulls raw Sharpe = 0.0661
    naive read: a positive raw Sharpe among the best of many nulls
    looks like an edge, even though every series is pure noise.

(b) THE CORRECTION
    prereg n_trials = 20 (written before evaluation)
    corrected_p = 0.4247, DSR = 0.5753, SR0 = 0.0601
    verdict = FAIL  (null best must FAIL)

(c) THE REAL ONE
    planted raw Sharpe = 0.5257 (true 0.5)
    corrected_p = 0.0000, DSR = 1.0000, SR0 = 0.0601
    verdict = PASS  (planted edge must PASS)

(d) THE WARNING
    sample-V rule, n_trials = 5 (small N)
    verdict = FAIL
    warnings = ['small-N sample-V: PASS conservative', 'under sample-V at small N a PASS is conservative and a FAIL is weak evidence']

elapsed = 6.3s
```
