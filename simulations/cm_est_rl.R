###############################################################################
# cm_est_rl.R -- the paper's R-learner base-learner variants (Lasso, XGBoost,
# KRR; unrevised "R--" and revised "Rev--") on EXACTLY the data of cm_est.R:
# same seed formula, same generator call order, so every row pairs with the
# deepHTL / R-DNN / competitor rows of est_c<cfg>_r<rep>.RData.
#   Rscript cm_est_rl.R <cfg 1-8> <rep>    env: CM_GAUSS=0|1 SMOKE=0|1
###############################################################################
args <- commandArgs(trailingOnly = TRUE); cfg_id <- as.integer(args[1]); rep_id <- as.integer(args[2])
suppressPackageStartupMessages({ library(deepTL); library(MASS); library(glmnet) })
source("lib/lasso.R"); source("lib/xgboost.R"); source("lib/kern.R"); source("cvboost2.R")
cvboost <- cvboost2                      # weight_xgboost() looks cvboost up in the global env
source("cm_dgp.R")
SMOKE <- identical(Sys.getenv("SMOKE"), "1")
OUT_DIR <- Sys.getenv("OUT_DIR", "out/est_rl")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cfg <- CM_CONFIG[cfg_id, ]; n <- cfg$n; p <- cfg$p; sigma <- cfg$sigma
if (SMOKE) n <- 300
cat(sprintf("cfg %d: n=%d p=%d sigma=%g | Scenario II | rep %d | gauss=%s\n", cfg_id, n, p, sigma, rep_id, CM_GAUSS))
set.seed(100000 * cfg_id + 1000 * rep_id + 11)          # identical draws to cm_est.R
d <- gen_cm(n, p, sigma, "S2"); x <- d$X; y <- d$Y; z <- d$Z
xt <- cm_X(n, p); tt <- cm_tau(xt, "S2")
logmse <- function(pred) log(mean((pred - tt)^2))
obj <- importTrt(x, y, z)
res <- list()
add <- function(m, pred) res[[length(res) + 1]] <<- data.frame(cfg_id = cfg_id, rep_id = rep_id, n = n, p = p, sigma = sigma, method = m, logmse = logmse(pred))
run <- function(label, fitter, predictor) {
  t0 <- Sys.time()
  th <- tryCatch(predictor(fitter(obj), xt, "both"), error = function(e) { cat("  ", label, "FAILED:", conditionMessage(e), "\n"); NULL })
  if (!is.null(th)) { add(paste0("R--", label), th[, 1]); add(paste0("Rev--", label), th[, 2]) }
  cat(sprintf("  %-8s %5.1f min\n", label, as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}
run("Lasso",   weight_lasso,   predict.weight_lasso)
run("XGBoost", weight_xgboost, predict.weight_xgboost)
run("KRR",     weight_kern,    predict.weight_kern)
df_est_rl <- do.call(rbind, res); print(df_est_rl[, c("method", "logmse")], row.names = FALSE)
save(df_est_rl, file = file.path(OUT_DIR, sprintf("est_rl_c%d_r%d.RData", cfg_id, rep_id)))
