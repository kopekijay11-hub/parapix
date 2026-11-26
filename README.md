Parapix v0.1.0 — a toolkit for testing the statistical significance of trading strategy returns. It provides block-bootstrap confidence intervals for mean and Sharpe ratio (with iid and dependent data), a permutation test for the null of zero skill, and the Deflated Sharpe Ratio (DSR) for multiple-testing correction.

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
- Trials are independent draws from a common distribution. Correlated trials (e.g. a grid search over a smooth parameter surface) violate this; the effective N is smaller than the raw count.
- Returns are iid. The analytic SE understates the true SE under autocorrelation: at AR(1) φ=0.3 the variance inflation is ~1.857×, so the SE is understated by ~1.36× (first order). DSR against SR₀ is therefore not a size-calibrated FWER test under dependence.

Sources: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014), Eq. (1)-(2); Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier" (2012), Eq. (11). Implementation: src/parapix/dsr.py.

## Permutation test assumption

The sign-flip permutation test assumes exchangeability: under the null the returns are symmetric about zero and signs are independent of magnitudes. It is robust to non-normality, not to autocorrelation/dependence. Under serial dependence its false-positive rate inflates, because permuting the series destroys the dependence structure the null relies on; a nominal level (e.g. 5%) can therefore be exceeded under dependence. See src/parapix/permtest.py.

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