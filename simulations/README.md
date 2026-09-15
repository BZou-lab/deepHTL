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

## Scripts (one replication per call)

| script | what it produces | paper |
|---|---|---|
| `cm_test.R <cfg> <rep>` with `TAU_TYPE=tau0|tau3|S2` | kernel score (Davies) and permutation p-values, unrevised and revised | Table 1, Web Table 3 |
| `cm_varsel.R <cfg> <rep>` | covariate-specific screen, marginal and conditional p-values | Table 2, Web Tables 7 and 9 |
| `cm_est.R <cfg> <rep>` with `TAU_TYPE=S2|S1` (default S2, the HTE effect function; S1 = the simple effect function) | log-MSE of deepHTL, R-DNN and six competitors on identical data | Web Tables 5 and 8 |
| `cm_est_rl.R <cfg> <rep>` with `TAU_TYPE=S2|S1` | log-MSE of the Lasso and KRR R-learner variants (unrevised and revised) on the same data | Web Table 4 |
| `cm_est_xgb.R <cfg> <rep>` with `TAU_TYPE=S2|S1` | log-MSE of the XGBoost variant (unrevised and revised) on the same data, via `cvboost2.R` | Web Table 4 |
| `cm_aggregate.R` | summary CSV files from `out/` | |

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
