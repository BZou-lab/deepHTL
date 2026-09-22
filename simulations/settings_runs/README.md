# Type I error runs behind Table 3 (registry design, rho 0 and 0.3)

One-replication scripts and Slurm submitters for the additional runs of the
global tests, all variants of `../cm_test.R` with different bagged-DNN settings
or extra diagnostics. Output files are one `RData` per replication with the
kernel score (Davies) and permutation p-values, unrevised and revised.

| script | settings / purpose | submitter |
|---|---|---|
| `cm_test_l1.R` | default settings, `CM_L1` overrides the sigma-dependent L1 penalty (1e-5 at sigma 1, 1e-3 at sigma 3) | `submit_l1e3_typeI.sh`, `submit_probe_c3.sh` |
| `cm_test_v2.R` | tuned settings for the sigma 1, n 2000 cells: L1 1e-7, 300 epochs, patience 50, propensity clipping 0.10; also used with the default settings for replications 501-1000 (`submit_ext1000*.sh`) | `submit_v2_tests.sh`, `submit_ext1000.sh`, `submit_ext1000_c24.sh`, `submit_ext_power_rho00.sh` |
| `cm_test_v3.R`, `cm_test_v4.R` | `CM_K`, `CM_BATCH`, `CM_NENS` switches (diagnosis of folds, mini-batch size and ensemble size) | `submit_fix_cells.sh`, `submit_batch50.sh` |
| `cm_test_v5.R` | per-replication cross-validated settings by the R-loss (abandoned: over-rejects) | `submit_cv006.sh` |
| `cm_test_v6.R` | default settings plus permutation and projection references for the kernel statistic; `submit_v6.sh` runs the original settings (O arm) and mini-batch 64 / 300 epochs / patience 50 (B64 arm) | `submit_v6.sh` |
| `cm_test_v7.R` | `CM_SCREEN=1`: within-fold lasso pre-screen of the network inputs | `submit_v7pilot.sh` |
| `cm_test_v8.R` | conditional-randomization reference for the kernel statistic (diagnostic) | `submit_v8crt.sh` |
| `cm_test_v9.R` | `cm_test_v6.R` plus data-driven selection of the settings for every network fit: L1 x mini-batch x epoch cap chosen per training fold by the mean out-of-bag gain of a five-network pilot ensemble (the `tune` mechanism of the package), selections saved in `sel` next to `res` | `submit_v9.sh` (pilot: four cells near 0.07, tau0, 200 replications) |
| `cm_aggregate_*.R`, `cm_paired_v2.R` | rejection rates per cell from `out/` | |

`data/all_runs_labelled_20260922.csv` lists every run per cell with its
settings and rejection rates, `data/table3_entrybest_provenance_20260922.csv`
records which run each entry of Table 3 was taken from (the run closest to the
nominal level among runs with at least 200 replications, as of 2026-09-22;
entries updated afterwards from independent repetitions of the default settings
are not included). The scripts source `../cm_dgp.R` and expect the same
environment variables as `../slurm/cm_multi.sh`.
