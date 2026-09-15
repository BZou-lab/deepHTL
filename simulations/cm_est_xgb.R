###############################################################################
# cm_est_xgb.R -- the XGBoost R-learner variant (R--XGBoost / Rev--XGBoost) on
# EXACTLY the data of cm_est.R / cm_est_rl.R (same seed formula and draw
# order). Split out because rlearner::cvboost is incompatible with the
# installed xgboost 3.2; weight_xgboost() is sourced unchanged from
# htdnn/xgboost.R and only its cvboost() call is redirected to cvboost2().
#   Rscript cm_est_xgb.R <cfg 1-8> <rep>    env: CM_GAUSS=0|1 SMOKE=0|1
###############################################################################
args <- commandArgs(trailingOnly = TRUE); cfg_id <- as.integer(args[1]); rep_id <- as.integer(args[2])
suppressPackageStartupMessages({ library(deepTL); library(MASS); library(glmnet); library(xgboost) })
source("lib/xgboost.R")
source("cvboost2.R"); source("cm_dgp.R")
cvboost <- cvboost2                      # weight_xgboost() looks cvboost up in the global env
SMOKE <- identical(Sys.getenv("SMOKE"), "1")
OUT_DIR <- Sys.getenv("OUT_DIR", "out/est_xgb")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cfg <- CM_CONFIG[cfg_id, ]; n <- cfg$n; p <- cfg$p; sigma <- cfg$sigma
if (SMOKE) n <- 300
cat(sprintf("cfg %d: n=%d p=%d sigma=%g | Scenario II | rep %d | gauss=%s\n", cfg_id, n, p, sigma, rep_id, CM_GAUSS))
set.seed(100000 * cfg_id + 1000 * rep_id + 11)          # identical draws to cm_est.R
d <- gen_cm(n, p, sigma, "S2"); x <- d$X; y <- d$Y; z <- d$Z
xt <- cm_X(n, p); tt <- cm_tau(xt, "S2")
logmse <- function(pred) log(mean((pred - tt)^2))
t0 <- Sys.time()
th <- predict.weight_xgboost(weight_xgboost(importTrt(x, y, z)), xt, "both")
df_est_rl <- data.frame(cfg_id = cfg_id, rep_id = rep_id, n = n, p = p, sigma = sigma,
                        method = c("R--XGBoost", "Rev--XGBoost"), logmse = c(logmse(th[, 1]), logmse(th[, 2])))
cat(sprintf("  XGBoost %5.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
print(df_est_rl[, c("method", "logmse")], row.names = FALSE)
save(df_est_rl, file = file.path(OUT_DIR, sprintf("est_rl_c%d_r%d.RData", cfg_id, rep_id)))
