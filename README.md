Work in progress. No results yet.

## Scope

Parapix tests statistical significance, not backtest correctness; it cannot detect lookahead bias, survivorship bias, unrealistic fills, or omitted costs.

## Known gaps

- Deflated Sharpe (WO-003)
- Timing permutation (circular shift of signal vs returns)
- Block sign-flip / wild bootstrap
- Walk-forward splitter (cut for v0.1)

## Permutation test assumption

The sign-flip permutation test assumes exchangeability: under the null the
returns are symmetric about zero and signs are independent of magnitudes. It is
robust to non-normality, not to autocorrelation/dependence. Under serial
dependence its false-positive rate inflates, because permuting the series
destroys the dependence structure the null relies on; a nominal level (e.g. 5%)
can therefore be exceeded under dependence. See src/parapix/permtest.py.
