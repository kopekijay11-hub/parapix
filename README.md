Work in progress. No results yet.

WO-002 validation numbers are provenance-verified, not reproduction-verified;
v3 provides the first bound numbers.

## Scope

Parapix tests statistical significance, not backtest correctness; it cannot detect lookahead bias, survivorship bias, unrealistic fills, or omitted costs.

## Known gaps

- Timing permutation (circular shift of signal vs returns)
- Block sign-flip / wild bootstrap
- Walk-forward splitter (cut for v0.1)

## Deflated Sharpe Ratio (WO-003)

The Deflated Sharpe Ratio (DSR) corrects for two sources of performance
inflation: selection bias across multiple trials and non-Normality of returns.
It is a Probabilistic Sharpe Ratio (PSR) whose benchmark is SR₀, the expected
maximum Sharpe of N independent zero-skill trials:

    SR₀ = √V · ((1 − γ_E) · Φ⁻¹(1 − 1/N) + γ_E · Φ⁻¹(1 − 1/(N·e)))
    DSR  = Φ((SR − SR₀) · √(n−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²))

where V is the variance across trials of estimated per-period Sharpes,
γ_E ≈ 0.5772 is the Euler-Mascheroni constant, γ₃ is skewness, γ₄ is raw
(kurtosis; 3 for a Normal), and n is the number of return observations.

Assumptions, plainly stated:
- Trials are independent draws from a common distribution. Correlated trials
  (e.g. a grid search over a smooth parameter surface) violate this; the
  effective N is smaller than the raw count.
- Returns are iid. The analytic SE understates the true SE under
  autocorrelation: at AR(1) φ=0.3 the variance inflation is ~1.857×, so the
  SE is understated by ~1.36× (first order). DSR against SR₀ is therefore not
  a size-calibrated FWER test under dependence.

Sources: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014),
Eq. (1)-(2); Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier"
(2012), Eq. (11). Implementation: src/parapix/dsr.py.

## Permutation test assumption

The sign-flip permutation test assumes exchangeability: under the null the
returns are symmetric about zero and signs are independent of magnitudes. It is
robust to non-normality, not to autocorrelation/dependence. Under serial
dependence its false-positive rate inflates, because permuting the series
destroys the dependence structure the null relies on; a nominal level (e.g. 5%)
can therefore be exceeded under dependence. See src/parapix/permtest.py.
