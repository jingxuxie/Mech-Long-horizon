# Clean-copy reproduction check — 2026-09-15

The source and measurement tree pushed in commit `0fcc2027f363659e0c07143542b3567855dd1e78` was verified against the local Git index. Both have tree SHA:

```
6d89f21358df25cede8b0951625677180304f387
```

This checks all 38 tracked files, with Git-normalized line endings. The GitHub connector also confirmed that `main` points to that commit.

A direct `git clone` from the execution container failed because it could not resolve github.com. Instead, the verified local Git index was exported into a new clean directory with `git checkout-index`, and the complete `make reproduce` and `make paper` commands were run there. This is a clean-copy rerun in the same execution environment, not an independent external replication.

Results:

- 24 regression tests passed.
- All nine CSV result tables reproduced exactly after excluding measured runtime columns.
- All ten learned checkpoint JSON files reproduced exactly, including loss traces and parameters.
- The exact and perturbation assertion checks passed again.
- The manuscript built to 15 pages with no unresolved reference markers.
- The final PDF was rendered and visually checked; plots use raw result tables.

The original runtime records were retained rather than replaced with the rerun timings. Exact agreement in this environment is not a promise of bitwise agreement on a different PyTorch backend or hardware platform.
