# Simulation code for the deepHTL paper

Self-contained R scripts that reproduce the simulation studies of the paper
(Section 3 and the Web Tables). Every table is an aggregate of one-replication
scripts run over a grid of eight configurations (`CM_CONFIG` in `cm_dgp.R`:
n in {1000, 2000} x p in {20, 40} x sigma in {1, 3}, n varying fastest) and
independent replications.

## Data-generating process (`cm_dgp.R`)

`gen_cm(n, p, sigma, tau_type)` draws from the outcome model of Section 3 with
`tau_type` in `tau0` (no effect), `tau3` (constant effect 3) and `S2` (the HTE
effect function); `S1` is the simpler effect function of Web Tables 4 and 5.
The covariate design is chosen with environment variables:

| design | environment |
|---|---|
| registry design (default): latent block correlation 0.3 on X1..X10, X3/X6/X7 binary | none (`CM_RHO` changes the block correlation) |
| benchmark design: X ~ N(0, I) | `CM_GAUSS=1` |
| benchmark covariates with a correlation block, no binaries (Web Table 7) | `CM_NOBIN=1 CM_RHO=0.5` |

`cm_dgp2.R` sources `cm_dgp.R` and adds optional simplifications of the nuisance
functions used in diagnostics only (`CM_FSIMPLE=1` linear f, `CM_ESIMPLE=1` linear
logit e, `CM_EMAIN=1` the original propensity plus main effects). With none of them
set it is identical to `cm_dgp.R`.

## Global tests, final settings (Table 3): `cm_test_v18r2.R`

`cm_test_v18r2.R` is the script behind Table 3 (type I error of the global tests at
rho = 0 and 0.3, unrevised and revised). One call is one replication of one grid cell:

```
Rscript cm_test_v18r2.R <cfg 1-8> <rep>
```

with the environment `TAU_TYPE=tau0|tau3|S2`, `CM_RHO=0|0.3`, `NPERM=2000`, `CM_K=5`,
`CM_CLIP=0.05`, `OUT_DIR` and the settings of `settings_pilot5.env`, which are the
final nuisance pipeline of the paper:

| variable | value | meaning |
|---|---|---|
| `CM_SCREEN=2` | hierarchical screen | within each training fold a lasso for Y (gaussian) and Z (binomial) on [X, X^2, X_i X_j], `lambda.min`, union of supports, and every covariate in a selected feature is kept (at least `CM_SCREEN_MIN`=5) |
| `CM_E_INPUT=raw` | propensity inputs | the propensity net takes the kept covariates only, the outcome nets (mu, mu*) the kept covariates plus the selected squares and products |
| `CM_MU_BATCH=128`, `CM_MU_EPOCH=500`, `CM_MU_PATIENCE=50` | outcome budget | mini-batch, epoch cap and patience of the outcome nets, while the propensity net and the stage-2 nets keep 256 / 120 / 20 |
| (default) | L1 | 1e-5 at sigma = 1, 1e-3 at sigma = 3 (`CM_L1` overrides) |

The script fits the nuisances once (stratified folds, propensity clipped at 0.05,
lambda-blended revised constant) and records, per replication, the product-form
kernel score test `p_score_u` / `p_score_r` (the analytic test of the paper, the
package's `kernel_score_test()`), the legacy sign-form statistic `p_davies_u` /
`p_davies_r`, the permutation test with the fold x arm shuffle `p_perm_u` / `p_perm_r`
(the permutation test of the paper) and with the fold x arm x e_hat-quintile shuffle
`p_permstr_u` / `p_permstr_r`, together with oracle nuisance diagnostics (`rmse_e`,
`rmse_mu`, `rmse_ms`, alignment scalars, kernel spectrum) and the per-observation
vectors needed to rebuild the statistics offline. The script was named
`cm_test_v18.R` (revision 2, `CM_E_INPUT`) during development, the copy here carries the
name used on the cluster. It sources `cm_dgp.R`, so the registry design is the default
and `CM_RHO` sets the block correlation.

To reproduce Table 3, run 1000 replications of every cell (cfg 1 to 8) for
`TAU_TYPE` in {tau0, tau3} and `CM_RHO` in {0, 0.3} with the settings above, then

```
Rscript cm_aggregate_v18.R <out dir(s) rho 0> <out dir(s) rho 0.3>
```

which prints the rejection rates at alpha = 0.05 in the layout of Table 3
(`summary_v18_table3.csv`, Anal. = `p_score_*`, Perm. = `p_perm_*`) and a long summary
with all recorded tests and the nuisance error per cell (`summary_v18_long.csv`).
`TAU_TYPE=S2` under the same settings gives the power rows.

`submit_v18fix.sh` is the Slurm submission used for the final run: it reads
`settings_pilot5.env`, exports it together with `CM_K=5 CM_CLIP=0.05 NPERM=2000
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1` and submits `cm_multi.sh` arrays (`RPT=1`, 1000
replications per cell, tau0 and tau3) for the cells listed in its header (cfg 3, 4, 5 and 8
at rho 0 and cfg 4, 6, 7 and 8 at rho 0.3, the cells whose earlier estimate was at or
above 0.065, the other cells are not resubmitted by this script). Run it from a directory that
contains `cm_test_v18r2.R`, `cm_dgp.R`, `settings_pilot5.env` and a copy of
`slurm/cm_multi.sh`, after editing the paths, module and library lines for your site.
`SMOKE=1` shrinks a call to n = 300, 50 permutations and two-network ensembles.

## Scripts (one replication per call)

| script | what it produces | paper |
|---|---|---|
| `cm_test_v18r2.R <cfg> <rep>` with `TAU_TYPE=tau0|tau3|S2` and `settings_pilot5.env` | product-form kernel score (Davies) and permutation p-values, unrevised and revised, final nuisance pipeline | Table 3 |
| `cm_test.R <cfg> <rep>` with `TAU_TYPE=tau0|tau3|S2` | the earlier pipeline (all covariates as network inputs, 256 / 120 / 20 for every net, sign-form statistic), kept for the record | superseded |
| `cm_varsel.R <cfg> <rep>` | covariate-specific screen, marginal and conditional p-values | Table 2, Web Tables 7 and 9 |
| `cm_est.R <cfg> <rep>` with `TAU_TYPE=S2|S1` (default S2, the HTE effect function; S1 = the simple effect function) | log-MSE of deepHTL, R-DNN and six competitors on identical data | Web Tables 5 and 8 |
| `cm_est_rl.R <cfg> <rep>` with `TAU_TYPE=S2|S1` | log-MSE of the Lasso and KRR R-learner variants (unrevised and revised) on the same data | Web Table 4 |
| `cm_est_xgb.R <cfg> <rep>` with `TAU_TYPE=S2|S1` | log-MSE of the XGBoost variant (unrevised and revised) on the same data, via `cvboost2.R` | Web Table 4 |
| `cm_aggregate_v18.R` | Table 3 and a long summary from `cm_test_v18r2.R` output | |
| `cm_aggregate.R` | summary CSV files from `out/` for the other scripts | |

`lib/` holds the estimator code exactly as used for the paper (`dnn.R` is the
bagged-DNN implementation of deepHTL and R-DNN; `lasso.R`, `xgboost.R`,
`kern.R` the alternative base learners). `cvboost2.R` re-implements
`rlearner::cvboost` on the current xgboost API. `slurm/` contains the array
job scripts and `launch_cm.sh`, the submission order used on our cluster; edit
the module and library lines for your site. `slurm/cm_multi.sh` runs several
replications per array task (env `SCRIPT`, `PREFIX`, `OUT_DIR`, `CFG`, `RPT`) for
clusters that cap the number of queued array tasks; it skips replications whose
output file already exists. `SMOKE=1` shrinks every script to a minutes-long
test run.

Dependencies: deepTL, MASS, glmnet, CompQuadForm, grf, ranger, dbarts, xgboost.

`settings_runs/` holds the earlier type I error runs of the global tests under
alternative bagged-DNN settings (L1 penalty, mini-batch size, epochs, per-fold tuning)
and the diagnostics that led to the final settings, with a per-entry provenance file.
