###############################################################################
# cm_test_v8.R -- 2026-09-22. cm_test_v7.R plus a CONDITIONAL RANDOMIZATION reference for the REVISED kernel statistic:
# the outcome residual R_i = Y*_i - mu*_hat(X_i) is held fixed and the treatment is redrawn Z*_i ~ Bernoulli(e_hat_i) B times;
# for each draw the statistic is rebuilt exactly as for the data (weights (Z*-e_hat)^2, fold-wise weighted centring,
# studentisation, projection P, quadratic form with K) and p_crt_r = (1 + #{Q* >= Q_r}) / (B + 1). Because
# sign(Z - e_hat) = 2Z - 1 under clipping, E[s_i | X] = (2 e_i - 1) b(X_i)/sigma is a mean shift the observed statistic carries
# whenever the nuisance residual b = mu - mu_hat is structured; the redrawn statistic carries the same shift (with e_hat in place
# of e), so the reference absorbs it instead of the Davies null. Diagnostic companion p_crtor_r redraws Z* from the TRUE
# propensity d$e (simulation only). Unrevised statistic is not CRT-calibrated (Y - mu_hat still contains tau0 Z).
###############################################################################
###############################################################################
args <- commandArgs(trailingOnly = TRUE); cfg_id <- as.integer(args[1]); rep_id <- as.integer(args[2])
suppressPackageStartupMessages({ library(deepTL); library(MASS); library(glmnet); library(CompQuadForm) })
source("cm_dgp.R")
TAU_TYPE <- Sys.getenv("TAU_TYPE", "S2"); B <- as.integer(Sys.getenv("NPERM", "2000"))
SMOKE <- identical(Sys.getenv("SMOKE"), "1")
OUT_DIR <- Sys.getenv("OUT_DIR", "/nas/longleaf/home/shuaiy/project/corrmix_design/out/test_v6")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cfg <- CM_CONFIG[cfg_id, ]; n <- cfg$n; p <- cfg$p; sigma <- cfg$sigma; K <- as.integer(Sys.getenv("CM_K", "5"))
L1   <- as.numeric(Sys.getenv("CM_L1", if (sigma == 1) "1e-5" else "1e-3"))
NEP  <- as.integer(Sys.getenv("CM_EPOCH", "120"))
PAT  <- as.integer(Sys.getenv("CM_PATIENCE", "20"))
CLIP <- as.numeric(Sys.getenv("CM_CLIP", "0.05"))
NBAT <- as.integer(Sys.getenv("CM_BATCH", "256"))
NENS <- as.integer(Sys.getenv("CM_NENS", "30"))
BK   <- as.integer(Sys.getenv("BKERN", "2000"))
SCREEN <- identical(Sys.getenv("CM_SCREEN", "0"), "1"); SCREEN_MIN <- as.integer(Sys.getenv("CM_SCREEN_MIN", "5"))
if (SMOKE) { n <- 300; B <- 50; BK <- 50 }
ctrl <- cm_ctrl(L1); if (!SMOKE) { ctrl$esCtrl$n.epoch <- NEP; ctrl$esCtrl$early.stop.det <- PAT
  ctrl$esCtrl$n.batch <- NBAT; ctrl$n.ensemble <- NENS }
cat(sprintf("cfg %d: n=%d p=%d sigma=%g | tau=%s | rep %d | B=%d BK=%d l1=%g\n", cfg_id, n, p, sigma, TAU_TYPE, rep_id, B, BK, L1))
cat(sprintf("epochs=%d patience=%d clip=%.2f K=%d batch=%d ensemble=%d screen=%d\n", ctrl$esCtrl$n.epoch, ctrl$esCtrl$early.stop.det, CLIP, K, ctrl$esCtrl$n.batch, ctrl$n.ensemble, SCREEN))
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

## ---- optional within-fold pre-screen of the nuisance inputs (union of lasso supports for Y and Z) ----
screen_cols <- function(xtr, ytr, ztr) {
  if (!SCREEN) return(seq_len(ncol(xtr)))
  cy <- tryCatch(as.numeric(stats::coef(glmnet::cv.glmnet(xtr, ytr, alpha = 1), s = "lambda.min"))[-1], error = function(e) rep(1, ncol(xtr)))
  cz <- tryCatch(as.numeric(stats::coef(glmnet::cv.glmnet(xtr, ztr, family = "binomial", alpha = 1), s = "lambda.min"))[-1], error = function(e) rep(1, ncol(xtr)))
  sc <- pmax(abs(cy) * apply(xtr, 2, stats::sd), abs(cz) * apply(xtr, 2, stats::sd))
  sel <- which(cy != 0 | cz != 0)
  if (length(sel) < SCREEN_MIN) sel <- order(sc, decreasing = TRUE)[seq_len(min(SCREEN_MIN, ncol(xtr)))]
  sort(sel) }
sels <- vector("list", K)

## ---- stage 1: e_hat, mu_hat ----
e_hat <- mu_hat <- numeric(n)
for (k in 1:K) { tr <- folds != k; te <- folds == k; sk <- screen_cols(X[tr, , drop = FALSE], Y[tr], Z[tr]); sels[[k]] <- sk
  e_hat[te]  <- fit_pred(X[tr, sk, drop = FALSE], z_fac[tr], X[te, sk, drop = FALSE])
  mu_hat[te] <- fit_pred(X[tr, sk, drop = FALSE], Y[tr],     X[te, sk, drop = FALSE]) }
n_sel <- mean(sapply(sels, length)); if (SCREEN) cat(sprintf("screen: mean %.1f of %d covariates kept per fold\n", n_sel, p))
e_hat <- pmin(pmax(e_hat, CLIP), 1 - CLIP); w <- (Z - e_hat)^2
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
for (k in 1:K) { tr <- folds != k; te <- folds == k; sk <- sels[[k]]; ys0_hat[te] <- fit_pred(X[tr, sk, drop = FALSE], Ystar[tr], X[te, sk, drop = FALSE]) }
tildeYstar <- (Ystar - ys0_hat) / (Z - e_hat)

## ---- kernel score test (unrevised, revised), Davies reference ----
resid_std <- function(yt) { r <- numeric(n); for (k in 1:K) { tr <- folds != k; te <- folds == k
    r[te] <- yt[te] - sum(w[tr] * yt[tr]) / sum(w[tr]) }
  s2 <- mean(w * r^2); r / sqrt(s2 / w) }
P <- diag(n) - tcrossprod(sqrt(w)) / sum(w)
Xs <- scale(X); Dm <- as.matrix(dist(Xs)); bw <- median(Dm[upper.tri(Dm)]); if (bw == 0) bw <- 1
Kmat <- exp(-(Dm^2) / (2 * bw^2)); Kp <- P %*% Kmat %*% P; Kp <- 0.5 * (Kp + t(Kp))
eig <- eigen(Kp, symmetric = TRUE, only.values = TRUE)$values; eig <- eig[eig > 1e-8]
p_dav_gen <- function(Q, lam) { pv <- tryCatch(CompQuadForm::davies(Q, lambda = lam)$Qq, error = function(e) NA_real_)
  if (is.na(pv) || !is.finite(pv) || pv < 0 || pv > 1) pv <- tryCatch(CompQuadForm::liu(Q, lambda = lam), error = function(e) NA_real_)
  pv }
qf <- function(M, s) as.numeric(crossprod(s, M %*% s))
s_u <- resid_std(Ytilde); s_r <- resid_std(tildeYstar)
Q_u <- qf(Kp, s_u); Q_r <- qf(Kp, s_r)
p_davies_u <- p_dav_gen(Q_u, eig); p_davies_r <- p_dav_gen(Q_r, eig)

## ---- score-form statistic: v = (Z - e_hat) * residual / sigma, null matrix M D K D M (as in v2) ----
dd <- Z - e_hat; Mm <- diag(n) - tcrossprod(dd) / sum(dd^2); A <- Mm %*% (dd * t(dd * Kmat)) %*% Mm; A <- 0.5 * (A + t(A))
eigA <- eigen(A, symmetric = TRUE, only.values = TRUE)$values; eigA <- eigA[eigA > 1e-8]
p_sc <- function(yt) { rc <- yt - sum(w * yt) / sum(w); v <- w * rc / sqrt(mean(w * rc^2)); p_dav_gen(qf(Kmat, v), eigA) }
p_score_u <- p_sc(Ytilde); p_score_r <- p_sc(tildeYstar); rm(A, Mm)

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

## ---- v6 additions (after every RNG-consuming step above): permutation references and the linear-projection variant ----
strata <- interaction(folds, Z, drop = TRUE)
perm_strat <- function() { idx <- seq_len(n)
  for (g in levels(strata)) { ix <- which(strata == g); if (length(ix) > 1) idx[ix] <- ix[sample.int(length(ix))] }
  idx }
U <- sqrt(w) * cbind(1, X); qrU <- qr(U); Qu <- qr.Q(qrU)[, seq_len(qrU$rank), drop = FALSE]
PB <- diag(n) - tcrossprod(Qu); KpB <- PB %*% Kmat %*% PB; KpB <- 0.5 * (KpB + t(KpB))
eigB <- eigen(KpB, symmetric = TRUE, only.values = TRUE)$values; eigB <- eigB[eigB > 1e-8]
QB_u <- qf(KpB, s_u); QB_r <- qf(KpB, s_r)
p_proj_u <- p_dav_gen(QB_u, eigB); p_proj_r <- p_dav_gen(QB_r, eigB)
cnt <- c(kperm_u = 0, kperm_r = 0, kperms_u = 0, kperms_r = 0, projk_u = 0, projk_r = 0)
for (b in seq_len(BK)) { i1 <- sample.int(n); i2 <- perm_strat()
  cnt["kperm_u"]  <- cnt["kperm_u"]  + (qf(Kp,  s_u[i1]) >= Q_u);  cnt["kperm_r"]  <- cnt["kperm_r"]  + (qf(Kp,  s_r[i1]) >= Q_r)
  cnt["kperms_u"] <- cnt["kperms_u"] + (qf(Kp,  s_u[i2]) >= Q_u);  cnt["kperms_r"] <- cnt["kperms_r"] + (qf(Kp,  s_r[i2]) >= Q_r)
  cnt["projk_u"]  <- cnt["projk_u"]  + (qf(KpB, s_u[i2]) >= QB_u); cnt["projk_r"]  <- cnt["projk_r"]  + (qf(KpB, s_r[i2]) >= QB_r) }
pk <- (cnt + 1) / (BK + 1)

## ---- v8: conditional randomization reference for the revised kernel statistic ----
crt_p <- function(R, e_draw, Q_obs, B = BK) {
  Zs <- matrix(stats::rbinom(n * B, 1, rep(e_draw, B)), n, B)     # redrawn treatments, columns = draws
  Ds <- Zs - e_hat; Ws <- Ds^2; Yt <- R / Ds                      # statistic rebuilt with e_hat, as on real data
  Rc <- Yt
  for (k in 1:K) { tr <- folds != k; te <- folds == k
    ck <- colSums(Ws[tr, , drop = FALSE] * Yt[tr, , drop = FALSE]) / colSums(Ws[tr, , drop = FALSE])
    Rc[te, ] <- Yt[te, , drop = FALSE] - matrix(ck, sum(te), B, byrow = TRUE) }
  s2 <- colMeans(Ws * Rc^2); S <- sqrt(Ws) * Rc / matrix(sqrt(s2), n, B, byrow = TRUE)
  a <- sqrt(Ws) / matrix(sqrt(colSums(Ws)), n, B, byrow = TRUE)
  PS <- S - a * matrix(colSums(a * S), n, B, byrow = TRUE)
  Qs <- colSums(PS * (Kmat %*% PS))
  c(p = (1 + sum(Qs >= Q_obs)) / (B + 1), mean = mean(Qs), sd = stats::sd(Qs)) }
R_rev <- Ystar - ys0_hat
stopifnot(abs(qf(Kp, resid_std(R_rev / (Z - e_hat))) - Q_r) < 1e-6)   # the pipeline's statistic is reproduced by the same formula
crt <- crt_p(R_rev, e_hat, Q_r); crt_or <- crt_p(R_rev, d$e, Q_r)

res <- data.frame(cfg_id = cfg_id, rep_id = rep_id, tau_type = TAU_TYPE, n = n, p = p, sigma = sigma, overlap_out = overlap,
                  l1 = L1, n_epoch = ctrl$esCtrl$n.epoch, patience = ctrl$esCtrl$early.stop.det, clip = CLIP, K = K,
                  n_batch = ctrl$esCtrl$n.batch, n_ens = ctrl$n.ensemble, bw = bw, rank_proj = qrU$rank, screen = SCREEN, n_sel = n_sel,
                  tau0 = tau0, tau0_opt = tau0_opt, lam = best$lam,
                  p_davies_u = p_davies_u, p_davies_r = p_davies_r, p_score_u = p_score_u, p_score_r = p_score_r,
                  p_perm_u = p_perm_u, p_perm_r = p_perm_r,
                  p_kperm_u = pk[["kperm_u"]], p_kperm_r = pk[["kperm_r"]], p_kperms_u = pk[["kperms_u"]], p_kperms_r = pk[["kperms_r"]],
                  p_proj_u = p_proj_u, p_proj_r = p_proj_r, p_projk_u = pk[["projk_u"]], p_projk_r = pk[["projk_r"]],
                  p_crt_r = crt[["p"]], p_crtor_r = crt_or[["p"]], Q_r = Q_r, crt_mean = crt[["mean"]], crt_sd = crt[["sd"]])
print(res, row.names = FALSE)
save(res, file = file.path(OUT_DIR, sprintf("test_%s_c%d_r%d.RData", TAU_TYPE, cfg_id, rep_id)))
