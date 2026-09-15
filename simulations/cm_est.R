###############################################################################
# cm_est.R -- estimation accuracy on the correlated-mixed design (Scenario II):
# deepHTL (revised) and R-DNN (unrevised) via weight_dnn (htdnn/dnn.R, the
# paper's estimation code) plus the six competitors of sce_competitors_all.R,
# all on the SAME data; independent test set of size n. One replication.
#   Rscript cm_est.R <cfg 1-8> <rep>    env: SMOKE=0|1
###############################################################################
args <- commandArgs(trailingOnly = TRUE); cfg_id <- as.integer(args[1]); rep_id <- as.integer(args[2])
suppressPackageStartupMessages({ library(deepTL); library(MASS); library(grf); library(ranger); library(glmnet) })
source("lib/dnn.R")
source("cm_dgp.R")
SMOKE <- identical(Sys.getenv("SMOKE"), "1")
OUT_DIR <- Sys.getenv("OUT_DIR", "out/est")
TAU_TYPE <- Sys.getenv("TAU_TYPE", "S2")   # S2 = HTE effect function (paper), S1 = simple effect function; use a different OUT_DIR per tau type
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
has_dbarts <- requireNamespace("dbarts", quietly = TRUE) && !SMOKE
cfg <- CM_CONFIG[cfg_id, ]; n <- cfg$n; p <- cfg$p; sigma <- cfg$sigma
if (SMOKE) n <- 300
ntree <- if (SMOKE) 50 else 500
cat(sprintf("cfg %d: n=%d p=%d sigma=%g | tau=%s | rep %d\n", cfg_id, n, p, sigma, TAU_TYPE, rep_id))
set.seed(100000 * cfg_id + 1000 * rep_id + 11)
d <- gen_cm(n, p, sigma, TAU_TYPE); x <- d$X; y <- d$Y; z <- d$Z
xt <- cm_X(n, p); tt <- cm_tau(xt, TAU_TYPE)
logmse <- function(pred) log(mean((pred - tt)^2))
res <- list()
add <- function(m, pred) res[[length(res) + 1]] <<- data.frame(cfg_id = cfg_id, rep_id = rep_id, n = n, p = p, sigma = sigma, method = m, logmse = logmse(pred))

## deepHTL / R-DNN (weight_dnn defaults: 5 folds, l1 = 1e-3)
fit <- weight_dnn(importTrt(x, y, z), en_dnn_ctrl = cm_ctrl(1e-3))
th  <- predict.weight_dnn(fit, xt, "both")
add("R-DNN", th[, 1]); add("deepHTL", th[, 2])

## competitors (identical code to sce_competitors_all.R / mixed_est.R)
rf  <- function(xx, yy) ranger::ranger(y ~ ., data = data.frame(y = yy, xx), num.trees = ntree, min.node.size = 5)
prd <- function(mod, xx) predict(mod, data.frame(xx))$predictions
cf <- grf::causal_forest(x, y, z, num.trees = if (SMOKE) 200 else 2000, tune.parameters = "all")
add("causal-forest", predict(cf, xt)$predictions)
m1 <- rf(x[z == 1, ], y[z == 1]); m0 <- rf(x[z == 0, ], y[z == 0])
add("T-learner", prd(m1, xt) - prd(m0, xt))
ms <- rf(cbind(x, Z = z), y); add("S-learner", prd(ms, cbind(xt, Z = 1)) - prd(ms, cbind(xt, Z = 0)))
D1 <- y[z == 1] - prd(m0, x[z == 1, ]); D0 <- prd(m1, x[z == 0, ]) - y[z == 0]
g1 <- rf(x[z == 1, ], D1); g0 <- rf(x[z == 0, ], D0)
e_rf <- ranger::ranger(zz ~ ., data = data.frame(zz = z, x), num.trees = ntree)
e_te <- pmin(pmax(predict(e_rf, data.frame(xt))$predictions, 0.05), 0.95)
add("X-learner", e_te * prd(g0, xt) + (1 - e_te) * prd(g1, xt))
K <- 5; folds <- sample(rep(1:K, length.out = n)); psi <- numeric(n)
for (k in 1:K) { tr <- folds != k; te <- which(folds == k)
  a1 <- rf(x[tr & z == 1, ], y[tr & z == 1]); a0 <- rf(x[tr & z == 0, ], y[tr & z == 0])
  ae <- ranger::ranger(zz ~ ., data = data.frame(zz = z[tr], x[tr, ]), num.trees = ntree)
  eh <- pmin(pmax(predict(ae, data.frame(x[te, ]))$predictions, 0.05), 0.95)
  m1h <- prd(a1, x[te, ]); m0h <- prd(a0, x[te, ])
  psi[te] <- (z[te] - eh) / (eh * (1 - eh)) * (y[te] - (z[te] * m1h + (1 - z[te]) * m0h)) + m1h - m0h }
add("DR-learner", prd(rf(x, psi), xt))
if (has_dbarts) { b1 <- dbarts::bart(x[z == 1, ], y[z == 1], xt, verbose = FALSE); b0 <- dbarts::bart(x[z == 0, ], y[z == 0], xt, verbose = FALSE)
  add("BART-T", colMeans(b1$yhat.test) - colMeans(b0$yhat.test)) }
df_est <- do.call(rbind, res); print(df_est[order(df_est$logmse), c("method", "logmse")], row.names = FALSE)
save(df_est, file = file.path(OUT_DIR, sprintf("est_c%d_r%d.RData", cfg_id, rep_id)))
