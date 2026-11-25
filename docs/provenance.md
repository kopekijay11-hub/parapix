# Provenance — WO-002 / WO-003 Recovery

## v2 selfvalidation artifact (recovered)

- Artifact: `results/selfvalidation_20261125T081205Z.json`
- Artifact sha256: `1c85252c8856eba9de6f1559b8904a04646e04f44a5f8cfc55d4910f8c83b0dd`
- Recovered v2 source: `validate.py` (quarantined at `~/parapix-recovery-quarantine/validate.py`)
- Quarantine file sha256: `957334dee7f9806a3c2e8849df1eca5ca17baee8ce1ba20d4cca0219d380c90a`
- Recovery commit: `92db7a7` ("recovered, tier A")
- Annotated tag: `validate-v2-recovered` on `92db7a7`
- Committed as repo-root `validate_v2.py` (faithful copy; later relativized and deleted in WO-003 B1 port)

## WO-002 corroboration

The recovered v2 source reproduces every top-level key of the v2 artifact verbatim:
`bootstrap_coverage`, `convention`, `end_to_end_fpr`, `permutation_fpr`,
`python_version`, `wall_clock_seconds`. Config: `n_obs=1000`, `n_series=1000`,
`n_perm=999`, `n_boot=999`, `n_strategies=2000`. The artifact omits `git_sha`/
`dirty` (v2 style). Tier A match.

## WO-003 v3 port

- `validate_v2.py` deleted from the working tree; its bytes survive at tagged
  commit `92db7a7` and in the quarantine copy.
- v3 runner: `src/parapix/selfvalidate.py` (run via `python -m parapix.selfvalidate`)
- Every v3 artifact is hash-bound: `git_sha`, `dirty`, `runner_sha256`,
  `command_line`, `python_version`, `numpy_version`, `scipy_version`,
  `pip_freeze_hash`, `PYTHONHASHSEED`, `blas_threads`, seeds, start/end times,
  wall clock.
- Runner refuses to run on a dirty/untracked tree unless `--unbound` labels
  the artifact UNBOUND; `results/` is excluded from the dirty check.
- Verifier: `git_sha` must be an ancestor of HEAD (`merge-base --is-ancestor`)
  and `runner_sha256` must match `git show <sha>:src/parapix/selfvalidate.py`.

## Status

WO-002 validation numbers are provenance-verified, not reproduction-verified;
v3 provides the first bound numbers.
