###############################################################################
# cm_test.R -- global tests (kernel score / Davies + cross-fitted permutation)
# on the correlated-mixed design (cm_dgp.R), one replication of one grid cell.
#   Rscript cm_test.R <cfg 1-8> <rep>    env: TAU_TYPE=tau0|tau3|S2  NPERM=2000  SMOKE=0|1
# Pipeline identical to mixed_test.R (nuisances fitted once and shared by both
# tests, stratified folds, propensity clipped at 0.05, lambda-blended tau0);
# l1 follows the paper's permutation scripts (1e-5 at sigma 1, 1e-3 at sigma 3).
###############################################################################
args <- commandArgs(trailingOnly = TRUE); cfg_id <- as.integer(args[1]); rep_id <- as.integer(args[2])
suppressPackageStartupMessages({ library(deepTL); library(MASS); library(glmnet); library(CompQuadForm) })
source("cm_dgp.R")
TAU_TYPE <- Sys.getenv("TAU_TYPE", "S2"); B <- as.integer(Sys.getenv("NPERM", "2000"))
SMOKE <- identical(Sys.getenv("SMOKE"), "1")
OUT_DIR <- Sys.getenv("OUT_DIR", "out/test")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cfg <- CM_CONFIG[cfg_id, ]; n <- cfg$n; p <- cfg$p; sigma <- cfg$sigma; K <- 5
if (SMOKE) { n <- 300; B <- 50 }
ctrl <- cm_ctrl(cm_l1(sigma))
cat(sprintf("cfg %d: n=%d p=%d sigma=%g | tau=%s | rep %d | B=%d l1=%g\n", cfg_id, n, p, sigma, TAU_TYPE, rep_id, B, cm_l1(sigma)))
set.seed(100000 * cfg_id + 1000 * rep_id + 7)
d <- gen_cm(n, p, sigma, TAU_TYPE); X <- d$X; Y <- d$Y; Z <- d$Z
overlap <- mean(d$e < 0.05 | d$e > 0.95)
z_fac <- factor(ifelse(Z == 1, "A", "B"), levels = c("A", "B"))

## ---- stratified folds ----
folds <- integer(n)
for (lev in c(0, 1)) { ix <- which(Z == lev); folds[ix] <- sample(rep(1:K, length.out = length(ix))) }
fit_pred <- function(x, y, xnew, w = NULL) {
  obj <- if (is.null(w)) deepTL::importDnnet(x = x, y = y) else deepTL::importDnnet(x = x, y = y, w = w)
  mod <- do.call(deepTL::ensemble_dnnet, c(list(object = obj), ctrl))
  pk <- deepTL::predict(mod, xnew)
  if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])
}

## ---- stage 1: e_hat, mu_hat ----
e_hat <- mu_hat <- numeric(n)
for (k in 1:K) { tr <- folds != k; te <- folds == k
  e_hat[te]  <- fit_pred(X[tr, ], z_fac[tr], X[te, ])
  mu_hat[te] <- fit_pred(X[tr, ], Y[tr],     X[te, ]) }
e_hat <- pmin(pmax(e_hat, 0.05), 0.95); w <- (Z - e_hat)^2
Ytilde <- (Y - mu_hat) / (Z - e_hat)

## ---- revised constant effect (lambda blend, as in the paper's scripts) ----
tau0 <- sum(w * Ytilde) / sum(w)
beta1 <- tryCatch(stats::coef(stats::lm(Y ~ Z + e_hat))[2], error = function(e) 0); if (is.na(beta1)) beta1 <- 0
best <- list(score = Inf, lam = 1); fi <- sample(rep(1:2, length.out = n))
for (lam in seq(0, 1, length.out = 21)) { c_lam <- lam * tau0 + (1 - lam) * beta1; score <- 0
  for (kk in 1:2) { itr <- fi != kk; ite <- fi == kk
    xm <- tryCatch(glmnet::cv.glmnet(X[itr, ], Y[itr] - c_lam * Z[itr], alpha = 1), error = function(e) NULL)
    if (is.null(xm)) { score <- Inf; break }
    xh <- as.numeric(stats::predict(xm, X[ite, ], s = "lambda.min"))
    score <- score + stats::var((Y[ite] - c_lam * Z[ite] - xh) / (Z[ite] - e_hat[ite]), na.rm = TRUE) }
  if (score < best$score) best <- list(score = score, lam = lam) }
tau0_opt <- best$lam * tau0 + (1 - best$lam) * beta1
Ystar <- Y - tau0_opt * Z; ys0_hat <- numeric(n)
for (k in 1:K) { tr <- folds != k; te <- folds == k; ys0_hat[te] <- fit_pred(X[tr, ], Ystar[tr], X[te, ]) }
tildeYstar <- (Ystar - ys0_hat) / (Z - e_hat)

## ---- kernel score test (unrevised, revised) ----
resid_std <- function(yt) { r <- numeric(n); for (k in 1:K) { tr <- folds != k; te <- folds == k
    r[te] <- yt[te] - sum(w[tr] * yt[tr]) / sum(w[tr]) }
  s2 <- mean(w * r^2); r / sqrt(s2 / w) }
P <- diag(n) - tcrossprod(sqrt(w)) / sum(w)
Xs <- scale(X); Dm <- as.matrix(dist(Xs)); bw <- median(Dm[upper.tri(Dm)]); if (bw == 0) bw <- 1
Kp <- P %*% exp(-(Dm^2) / (2 * bw^2)) %*% P; Kp <- 0.5 * (Kp + t(Kp))
eig <- eigen(Kp, symmetric = TRUE, only.values = TRUE)$values; eig <- eig[eig > 1e-8]
p_dav <- function(s) { Q <- as.numeric(t(s) %*% Kp %*% s)
  pv <- tryCatch(CompQuadForm::davies(Q, lambda = eig)$Qq, error = function(e) NA_real_)
  if (is.na(pv) || !is.finite(pv) || pv < 0 || pv > 1) pv <- tryCatch(CompQuadForm::liu(Q, lambda = eig), error = function(e) NA_real_)
  pv }
p_davies_u <- p_dav(resid_std(Ytilde)); p_davies_r <- p_dav(resid_std(tildeYstar))

## ---- permutation test: cross-fitted stage-2 predictions, shuffle within fold and arm ----
pred_u <- pred_r <- numeric(n)
for (k in 1:K) { tr <- folds != k; te <- folds == k
  pred_u[te] <- fit_pred(X[tr, ], Ytilde[tr],     X[te, ], w = w[tr])
  pred_r[te] <- fit_pred(X[tr, ], tildeYstar[tr], X[te, ], w = w[tr]) }
obs_u <- sum(w * (Ytilde - pred_u)^2) / n; obs_r <- sum(w * (tildeYstar - pred_r)^2) / n
perm_u <- perm_r <- numeric(B)
for (b in 1:B) { pu <- pr <- numeric(n)
  for (k in 1:K) for (arm in c(0, 1)) { ix <- which(folds == k & Z == arm); sh <- sample(ix); pu[ix] <- pred_u[sh]; pr[ix] <- pred_r[sh] }
  perm_u[b] <- sum(w * (Ytilde - pu)^2) / n; perm_r[b] <- sum(w * (tildeYstar - pr)^2) / n }
p_perm_u <- (sum(perm_u <= obs_u) + 1) / (B + 1); p_perm_r <- (sum(perm_r <= obs_r) + 1) / (B + 1)

res <- data.frame(cfg_id = cfg_id, rep_id = rep_id, tau_type = TAU_TYPE, n = n, p = p, sigma = sigma, overlap_out = overlap,
                  tau0 = tau0, tau0_opt = tau0_opt, lam = best$lam,
                  p_davies_u = p_davies_u, p_davies_r = p_davies_r, p_perm_u = p_perm_u, p_perm_r = p_perm_r)
print(res, row.names = FALSE)
save(res, file = file.path(OUT_DIR, sprintf("test_%s_c%d_r%d.RData", TAU_TYPE, cfg_id, rep_id)))
