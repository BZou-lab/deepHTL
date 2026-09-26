###############################################################################
# cm_test_v15.R -- 2026-09-25. ORACLE-SWAP experiment for the type I inflation of the analytic kernel score test.
# Lines from "args <-" to the end of the stage-2 permutation test are cm_test_v6.R VERBATIM (same DGP call, folds,
# fit_pred, stage 1, lambda-blended tau0_opt, ys0_hat, resid_std, P, K, Davies code, product-form block, permutation
# test, same RNG order), so p_davies_u/r, p_score_u/r and p_perm_u/r reproduce v6 for the same seed. Dropped from v6:
# the post-permutation kperm / kperms / proj / projk block and its BKERN env (already measured in out/v6orig_*).
# Everything new is computed AFTER the v6 permutation test:
#   1. oracle nuisances from the DGP: e_true = d$e, mu_true = f + tau(X) e_true, mu_star_true = f + (tau(X) - tau0_opt) e_true
#      (tau(X) = 0 for tau0, 3 for tau3); delta_e = e_true - e_hat, delta_mu = mu_true - mu_hat, delta_ms = mu_star_true - ys0_hat;
#      rmse_e/mu/ms, alignment scalars c_e/c_mu/c_ms = mean((2 e_true - 1) delta_*), mean_de/dmu/dms, mean_de_dmu, mean_de2,
#      cor_de_a = cor(delta_e, 2 e_true - 1) (cor_dmu_a, cor_dms_a likewise).
#   2. swap versions of the analytic test, (mu source, e source) in {hh, th, ht, tt}, same folds, w / P / eigenvalues recomputed
#      per e source (e_true UNCLIPPED): p_dav_u_<mmee>, p_dav_r_<mmee>, z_u_<mmee>, z_r_<mmee> with z = (Q - sum lam) / sqrt(2 sum lam^2),
#      tau0_<mmee> = plain weighted mean of the swapped Ytilde (diagnostic only: the revised outcome keeps Ystar = Y - tau0_opt Z
#      because ys0_hat and mu_star_true are both defined at the (h,h) tau0_opt). chk_hh_u/r = p_dav_*_hh - p_davies_* (must be 0).
#   3. predicted mean shift of the (h,h) scores: m_u = sign(Z - e_hat) (delta_mu - tau(X) delta_e) / sigma_hat_u,
#      m_r = sign(Z - e_hat) delta_ms / sigma_hat_r (sigma_hat = sqrt(s2) of resid_std); ncp_u/r = (P m)' K (P m), sd_null =
#      sqrt(2 sum lam^2), z_pred_u/r = ncp / sd_null, sum_s_u/r = sum(s), sum_m_u/r = sum(m); exact decomposition
#      Q = ncp + q_cross + q_noise with q_cross = 2 m' P K P (s - m), q_noise = (s - m)' P K P (s - m); kp_ones = 1' P K P 1 / n.
#      sig_u/r_<mmee> = the studentizer sqrt(s2) of every swap (sigma_hat_u/r duplicate sig_u/r_hh).
#   4. kernel spectrum of P K P: lam1_share, lam1_sq_share, sum_lam, k_off_mean / k_off_sd (off-diagonal K), cos1 = |v1' 1| / sqrt(n)
#      for the top eigenvector v1, cos_sw1 = cos(sqrt(w), 1), q1_u/r = lam1 (v1' s)^2 (top-direction share of Q).
#   5. permutation test on the oracle (t,t) inputs: stage-2 refit on the oracle Ytilde / tildeYstar (2 extra fit_pred loops),
#      (dropped in v17)
#   6. candidate fix on the (h,h) scores: centered projection P_c = I - Qc Qc', Qc = orthonormal basis of cbind(sqrt(w), 1),
#      Davies on the eigenvalues of P_c K P_c: p_dav_u_cen, p_dav_r_cen, z_u_cen, z_r_cen, lam1_share_cen, rank_cen.
#   7. per-observation vectors saved next to res (list vec: X (n x p, stored so K can be rebuilt offline without regenerating the
#      DGP, about 0.6 MB per file at n 2000 p 40), Z, Y, e_true, mu_true, mu_star_true, e_hat (clipped), mu_hat, ys0_hat, folds, w,
#      s_u, s_r, pred_u, pred_r, eig, tau0, tau0_opt, bw, x_check = c(sum(X), sum(X^2))). If X is ever regenerated offline instead
#      (set.seed(100000 * cfg_id + 1000 * rep_id + 7); gen_cm(n, p, sigma, TAU_TYPE)), compare with
#      all.equal(vec$x_check, c(sum(X), sum(X^2)), tolerance = 1e-8), never identical(): the same seed reproduces X only to
#      rounding across machines / node types (sum(X) differed at the 1e-13 relative level between this Mac and the login node).
# env (all optional): CM_L1 CM_EPOCH CM_PATIENCE CM_CLIP CM_BATCH CM_NENS CM_K NPERM TAU_TYPE CM_RHO OUT_DIR SMOKE (n 300, B 50)
#   Rscript cm_test_v15.R <cfg 1-8> <rep>      output: OUT_DIR/test_<TAU_TYPE>_c<cfg>_r<rep>.RData holding res (1 row) and vec
# cm_test_v18.R -- 2026-09-26 (rev 2: CM_E_INPUT=raw|aug, propensity-net inputs). cm_test_v17.R plus CM_SCREEN=2: HIERARCHICAL screen (lasso for Y and Z on [X, X^2, X_i X_j], all p variables)
# and INTERACTION-AUGMENTED nuisance inputs (selected raw variables + the selected squares / products) for e, mu, mu*. Kernel test and
# stage-2 model still use the raw p covariates. CM_SCREEN=1 = the v7 main-effect screen, 0 = none. Records n_sel (raw vars) and n_feat.
# cm_test_v17.R -- 2026-09-26. cm_test_v16.R (v15 + pre-screen) with (a) SEPARATE DNN settings for the propensity model (CM_E_*) and the
# outcome models mu / mu* (CM_MU_*): HID (comma list), BATCH, EPOCH, PATIENCE, L1, LR, each defaulting to the global CM_* value;
# (b) the permutation test computed with BOTH the fold x arm shuffle (p_perm_*) and the fold x arm x e_hat-quintile shuffle
# (p_permstr_*); (c) the oracle (t,t) stage-2 refit of v15 dropped (no p_perm_*_tt columns). Product-form test p_score_*
# (orthogonal score, homoscedastic Davies) is computed as in v6/v15 and is the intended primary analytic test.
# cm_test_v16.R -- 2026-09-25 (night). cm_test_v15.R plus the OPTIONAL within-fold covariate pre-screen of cm_test_v7.R for the
# NUISANCE models (CM_SCREEN=1): in each training fold a lasso for Y ~ X (gaussian) and Z ~ X (binomial), lambda.min, and the
# union of their supports (at least CM_SCREEN_MIN, filled up by absolute standardized coefficient) is the input set of the
# bagged-DNN fits of e, mu and mu* on that fold. Kernel test and stage-2 models still use all p covariates. The screen consumes
# RNG (cv.glmnet folds) before stage 1, so seeds do NOT reproduce v15/v6. Records screen, n_sel. CM_SCREEN=0 == v15.
###############################################################################
t0 <- proc.time()[["elapsed"]]
args <- commandArgs(trailingOnly = TRUE); cfg_id <- as.integer(args[1]); rep_id <- as.integer(args[2])
suppressPackageStartupMessages({ library(deepTL); library(MASS); library(glmnet); library(CompQuadForm) })
source("cm_dgp.R")
TAU_TYPE <- Sys.getenv("TAU_TYPE", "S2"); B <- as.integer(Sys.getenv("NPERM", "2000"))
SMOKE <- identical(Sys.getenv("SMOKE"), "1")
OUT_DIR <- Sys.getenv("OUT_DIR", "/nas/longleaf/home/shuaiy/project/corrmix_design/out/test_v15")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
cfg <- CM_CONFIG[cfg_id, ]; n <- cfg$n; p <- cfg$p; sigma <- cfg$sigma; K <- as.integer(Sys.getenv("CM_K", "5"))
L1   <- as.numeric(Sys.getenv("CM_L1", if (sigma == 1) "1e-5" else "1e-3"))
NEP  <- as.integer(Sys.getenv("CM_EPOCH", "120"))
PAT  <- as.integer(Sys.getenv("CM_PATIENCE", "20"))
CLIP <- as.numeric(Sys.getenv("CM_CLIP", "0.05"))
NBAT <- as.integer(Sys.getenv("CM_BATCH", "256"))
NENS <- as.integer(Sys.getenv("CM_NENS", "30"))
SCREEN_MODE <- as.integer(Sys.getenv("CM_SCREEN", "0")); SCREEN <- SCREEN_MODE >= 1; SCREEN_MIN <- as.integer(Sys.getenv("CM_SCREEN_MIN", "5"))
LR <- as.numeric(Sys.getenv("CM_LR", "1e-3")); HID <- as.integer(strsplit(Sys.getenv("CM_HID", "128,64,32"), ",")[[1]])
gv <- function(m, key, def, f = as.numeric) { v <- Sys.getenv(paste0("CM_", m, "_", key), ""); if (v == "") def else f(v) }
mk_ctrl <- function(m) list(n.ensemble = if (SMOKE) 2 else NENS, verbose = FALSE,
  esCtrl = list(n.hidden = gv(m, "HID", HID, function(v) as.integer(strsplit(v, ",")[[1]])), n.batch = gv(m, "BATCH", NBAT, as.integer),
                n.epoch = if (SMOKE) 5 else gv(m, "EPOCH", NEP, as.integer), norm.x = TRUE, norm.y = TRUE, activate = "relu", accel = "rcpp",
                l1.reg = gv(m, "L1", L1), plot = FALSE, learning.rate.adaptive = "adam", learning.rate = gv(m, "LR", LR),
                early.stop.det = gv(m, "PATIENCE", PAT, as.integer)))
ctrl_e <- mk_ctrl("E"); ctrl_mu <- mk_ctrl("MU")
if (SMOKE) { n <- 300; B <- 50 }
ctrl <- cm_ctrl(L1); if (!SMOKE) { ctrl$esCtrl$n.epoch <- NEP; ctrl$esCtrl$early.stop.det <- PAT
  ctrl$esCtrl$n.batch <- NBAT; ctrl$n.ensemble <- NENS }
cat(sprintf("cfg %d: n=%d p=%d sigma=%g | tau=%s | rep %d | B=%d l1=%g\n", cfg_id, n, p, sigma, TAU_TYPE, rep_id, B, L1))
cat(sprintf("epochs=%d patience=%d clip=%.2f K=%d batch=%d ensemble=%d\n", ctrl$esCtrl$n.epoch, ctrl$esCtrl$early.stop.det, CLIP, K, ctrl$esCtrl$n.batch, ctrl$n.ensemble))
set.seed(100000 * cfg_id + 1000 * rep_id + 7)
d <- gen_cm(n, p, sigma, TAU_TYPE); X <- d$X; Y <- d$Y; Z <- d$Z
overlap <- mean(d$e < 0.05 | d$e > 0.95)
z_fac <- factor(ifelse(Z == 1, "A", "B"), levels = c("A", "B"))

## ---- stratified folds ----
folds <- integer(n)
for (lev in c(0, 1)) { ix <- which(Z == lev); folds[ix] <- sample(rep(1:K, length.out = length(ix))) }
fit_pred <- function(x, y, xnew, w = NULL, ct = ctrl) {
  obj <- if (is.null(w)) deepTL::importDnnet(x = x, y = y) else deepTL::importDnnet(x = x, y = y, w = w)
  mod <- do.call(deepTL::ensemble_dnnet, c(list(object = obj), ct))
  pk <- deepTL::predict(mod, xnew)
  if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])
}

## ---- optional within-fold pre-screen of the nuisance inputs (union of lasso supports for Y and Z), as in cm_test_v7.R ----
## hierarchical screen (CM_SCREEN=2): lasso on [X, X^2, X_i X_j]; returns raw variables involved and the selected engineered columns
pairs_all <- utils::combn(p, 2); Fall <- cbind(X, X^2, X[, pairs_all[1, ]] * X[, pairs_all[2, ]])
fvars_all <- c(lapply(1:p, function(j) j), lapply(1:p, function(j) j), lapply(seq_len(ncol(pairs_all)), function(k) pairs_all[, k]))
lasso_nz <- function(Ftr, ytr, fam) { cf <- tryCatch(as.numeric(stats::coef(glmnet::cv.glmnet(Ftr, ytr, family = fam, alpha = 1), s = "lambda.min"))[-1], error = function(e) rep(0, ncol(Ftr))); which(cf != 0) }
screen_h <- function(tr) { sf <- sort(unique(c(lasso_nz(Fall[tr, ], Y[tr], "gaussian"), lasso_nz(Fall[tr, ], Z[tr], "binomial"))))
  vars <- sort(unique(unlist(fvars_all[sf]))); if (length(vars) < SCREEN_MIN) vars <- sort(unique(c(vars, order(-abs(stats::cor(X[tr, ], Y[tr])))[seq_len(SCREEN_MIN)])))
  list(vars = vars, feats = sf) }
nuis_input <- function(sel, aug = TRUE) if (is.list(sel)) { if (aug) cbind(X[, sel$vars, drop = FALSE], Fall[, sel$feats, drop = FALSE]) else X[, sel$vars, drop = FALSE] } else X[, sel, drop = FALSE]
E_AUG <- Sys.getenv("CM_E_INPUT", "aug") == "aug"   # CM_E_INPUT=raw: propensity net gets the selected raw variables only (mu nets keep the augmented inputs)
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
for (k in 1:K) { tr <- folds != k; te <- folds == k; sk <- if (SCREEN_MODE == 2) screen_h(tr) else screen_cols(X[tr, , drop = FALSE], Y[tr], Z[tr]); sels[[k]] <- sk
  Xin <- nuis_input(sk); Xe <- nuis_input(sk, aug = E_AUG)
  e_hat[te]  <- fit_pred(Xe[tr, , drop = FALSE], z_fac[tr], Xe[te, , drop = FALSE], ct = ctrl_e)
  mu_hat[te] <- fit_pred(Xin[tr, , drop = FALSE], Y[tr],     Xin[te, , drop = FALSE], ct = ctrl_mu) }
n_sel <- mean(sapply(sels, function(s) if (is.list(s)) length(s$vars) else length(s))); n_feat <- mean(sapply(sels, function(s) ncol(nuis_input(s))))
if (SCREEN) cat(sprintf("screen mode %d: mean %.1f of %d covariates kept per fold, %.1f nuisance inputs\n", SCREEN_MODE, n_sel, p, n_feat))
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
for (k in 1:K) { tr <- folds != k; te <- folds == k; Xin <- nuis_input(sels[[k]]); ys0_hat[te] <- fit_pred(Xin[tr, , drop = FALSE], Ystar[tr], Xin[te, , drop = FALSE], ct = ctrl_mu) }
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
## ---- stratified permutation: shuffle within fold x arm x e_hat quintile (exchangeability of the pseudo-outcomes within strata) ----
qb <- cut(e_hat, breaks = unique(stats::quantile(e_hat, seq(0, 1, .2))), include.lowest = TRUE)
strat_fq <- interaction(folds, Z, qb, drop = TRUE); idx_by <- split(seq_len(n), strat_fq)
perms_u <- perms_r <- numeric(B)
for (b in 1:B) { sh <- integer(n); for (ix in idx_by) sh[ix] <- if (length(ix) > 1) ix[sample.int(length(ix))] else ix
  perms_u[b] <- sum(w * (Ytilde - pred_u[sh])^2) / n; perms_r[b] <- sum(w * (tildeYstar - pred_r[sh])^2) / n }
p_permstr_u <- (sum(perms_u <= obs_u) + 1) / (B + 1); p_permstr_r <- (sum(perms_r <= obs_r) + 1) / (B + 1); n_strata <- nlevels(strat_fq)

## =====================================================================================================================
## v15 additions. Every v6 quantity above is final here; the RNG is consumed again only by the (t,t) stage-2 refit in 5.
## =====================================================================================================================
secs_v6 <- proc.time()[["elapsed"]] - t0

## ---- 1. oracle nuisances from the DGP ----
tau0_true <- switch(TAU_TYPE, tau0 = 0, tau3 = 3, NA_real_)
tau_true <- d$tau; e_true <- d$e; f_true <- cm_f(X); a_true <- 2 * e_true - 1
mu_true <- f_true + tau_true * e_true                            # E[Y | X]
mu_star_true <- f_true + (tau_true - tau0_opt) * e_true          # E[Y - tau0_opt Z | X]
delta_e <- e_true - e_hat; delta_mu <- mu_true - mu_hat; delta_ms <- mu_star_true - ys0_hat
rmse <- function(v) sqrt(mean(v^2))

## ---- 2. swap versions of the analytic test: same folds, w / P / eigenvalues recomputed per e source ----
resid_std_w <- function(yt, ww) { r <- numeric(n); for (k in 1:K) { tr <- folds != k; te <- folds == k
    r[te] <- yt[te] - sum(ww[tr] * yt[tr]) / sum(ww[tr]) }
  s2 <- mean(ww * r^2); list(s = r / sqrt(s2 / ww), s2 = s2) }
kern_null <- function(ww) { Pw <- diag(n) - tcrossprod(sqrt(ww)) / sum(ww); Kw <- Pw %*% Kmat %*% Pw; Kw <- 0.5 * (Kw + t(Kw))
  ev <- eigen(Kw, symmetric = TRUE, only.values = TRUE)$values; list(Kp = Kw, eig = ev[ev > 1e-8]) }
zq <- function(Q, lam) (Q - sum(lam)) / sqrt(2 * sum(lam^2))
src_e <- list(h = e_hat, t = e_true); src_mu <- list(h = mu_hat, t = mu_true); src_ms <- list(h = ys0_hat, t = mu_star_true)
nullK <- lapply(src_e, function(ee) kern_null((Z - ee)^2))       # h: identical arithmetic to v6's P, Kp, eig
swap <- list()
for (ee in c("h", "t")) for (mm in c("h", "t")) { es <- src_e[[ee]]; ww <- (Z - es)^2
  Yt_s <- (Y - src_mu[[mm]]) / (Z - es); Ys_s <- (Ystar - src_ms[[mm]]) / (Z - es)
  su <- resid_std_w(Yt_s, ww); sr <- resid_std_w(Ys_s, ww); nk <- nullK[[ee]]
  Qu <- qf(nk$Kp, su$s); Qr <- qf(nk$Kp, sr$s)
  swap[[paste0(mm, ee)]] <- list(tau0 = sum(ww * Yt_s) / sum(ww), p_u = p_dav_gen(Qu, nk$eig), p_r = p_dav_gen(Qr, nk$eig),
                                 z_u = zq(Qu, nk$eig), z_r = zq(Qr, nk$eig), s2_u = su$s2, s2_r = sr$s2,
                                 Yt = Yt_s, Ys = Ys_s, w = ww) }
chk_hh_u <- swap$hh$p_u - p_davies_u; chk_hh_r <- swap$hh$p_r - p_davies_r
if (!isTRUE(chk_hh_u == 0) || !isTRUE(chk_hh_r == 0)) cat(sprintf("WARNING: (h,h) swap differs from v6: %g %g\n", chk_hh_u, chk_hh_r))
sd_null_t <- sqrt(2 * sum(nullK$t$eig^2))

## ---- 3. predicted mean shift of the (h,h) scores ----
sg_u <- sqrt(swap$hh$s2_u); sg_r <- sqrt(swap$hh$s2_r); sgn <- sign(Z - e_hat)
m_u <- sgn * (delta_mu - tau_true * delta_e) / sg_u; m_r <- sgn * delta_ms / sg_r
ncp_u <- qf(Kp, m_u); ncp_r <- qf(Kp, m_r); sd_null <- sqrt(2 * sum(eig^2))
q_cross_u <- 2 * as.numeric(crossprod(m_u, Kp %*% (s_u - m_u))); q_noise_u <- qf(Kp, s_u - m_u)   # Q_u = ncp_u + q_cross_u + q_noise_u
q_cross_r <- 2 * as.numeric(crossprod(m_r, Kp %*% (s_r - m_r))); q_noise_r <- qf(Kp, s_r - m_r)
kp_ones <- sum(Kp) / n                                            # 1' P K P 1 / n: K-energy left along the ones direction after P

## ---- 4. kernel spectrum of P K P ----
eV <- eigen(Kp, symmetric = TRUE); v1 <- eV$vectors[, 1]; rm(eV)
cos1 <- abs(sum(v1)) / sqrt(n); cos_sw1 <- abs(sum(sqrt(w))) / sqrt(sum(w) * n)
q1_u <- eig[1] * sum(v1 * s_u)^2; q1_r <- eig[1] * sum(v1 * s_r)^2
k_off <- Kmat[upper.tri(Kmat)]; k_off_mean <- mean(k_off); k_off_sd <- stats::sd(k_off); rm(k_off)

## ---- 5. (v15's oracle stage-2 refit dropped in v17) ----

## ---- 6. candidate fix on the (h,h) scores: centered projection removing span{sqrt(w), 1} ----
Uc <- cbind(sqrt(w), 1); qrC <- qr(Uc); Qc <- qr.Q(qrC)[, seq_len(qrC$rank), drop = FALSE]
Pc <- diag(n) - tcrossprod(Qc); Kpc <- Pc %*% Kmat %*% Pc; Kpc <- 0.5 * (Kpc + t(Kpc))
eigC <- eigen(Kpc, symmetric = TRUE, only.values = TRUE)$values; eigC <- eigC[eigC > 1e-8]
Qc_u <- qf(Kpc, s_u); Qc_r <- qf(Kpc, s_r)
p_dav_u_cen <- p_dav_gen(Qc_u, eigC); p_dav_r_cen <- p_dav_gen(Qc_r, eigC); z_u_cen <- zq(Qc_u, eigC); z_r_cen <- zq(Qc_r, eigC)
rm(Pc, Kpc, Uc)

## ---- 7. results ----
sw <- function(nm, what) swap[[nm]][[what]]
res <- data.frame(cfg_id = cfg_id, rep_id = rep_id, tau_type = TAU_TYPE, n = n, p = p, sigma = sigma, overlap_out = overlap,
                  l1 = L1, n_epoch = ctrl$esCtrl$n.epoch, patience = ctrl$esCtrl$early.stop.det, clip = CLIP, K = K,
                  n_batch = ctrl$esCtrl$n.batch, n_ens = ctrl$n.ensemble, bw = bw, screen = SCREEN_MODE, n_sel = n_sel, n_feat = n_feat, e_input = if (E_AUG) "aug" else "raw",
                  tau0 = tau0, tau0_opt = unname(tau0_opt), lam = best$lam, tau0_true = tau0_true,
                  p_davies_u = p_davies_u, p_davies_r = p_davies_r, p_score_u = p_score_u, p_score_r = p_score_r,
                  p_perm_u = p_perm_u, p_perm_r = p_perm_r,
                  # 1. nuisance error and alignment
                  rmse_e = rmse(delta_e), rmse_mu = rmse(delta_mu), rmse_ms = rmse(delta_ms),
                  c_e = mean(a_true * delta_e), c_mu = mean(a_true * delta_mu), c_ms = mean(a_true * delta_ms),
                  mean_de = mean(delta_e), mean_dmu = mean(delta_mu), mean_dms = mean(delta_ms),
                  mean_de_dmu = mean(delta_e * delta_mu), mean_de2 = mean(delta_e^2),
                  cor_de_a = stats::cor(delta_e, a_true), cor_dmu_a = stats::cor(delta_mu, a_true), cor_dms_a = stats::cor(delta_ms, a_true),
                  # 2. swaps (mm = mu source, ee = e source; h = fitted, t = oracle)
                  p_dav_u_hh = sw("hh", "p_u"), p_dav_r_hh = sw("hh", "p_r"), z_u_hh = sw("hh", "z_u"), z_r_hh = sw("hh", "z_r"), tau0_hh = sw("hh", "tau0"),
                  p_dav_u_th = sw("th", "p_u"), p_dav_r_th = sw("th", "p_r"), z_u_th = sw("th", "z_u"), z_r_th = sw("th", "z_r"), tau0_th = sw("th", "tau0"),
                  p_dav_u_ht = sw("ht", "p_u"), p_dav_r_ht = sw("ht", "p_r"), z_u_ht = sw("ht", "z_u"), z_r_ht = sw("ht", "z_r"), tau0_ht = sw("ht", "tau0"),
                  p_dav_u_tt = sw("tt", "p_u"), p_dav_r_tt = sw("tt", "p_r"), z_u_tt = sw("tt", "z_u"), z_r_tt = sw("tt", "z_r"), tau0_tt = sw("tt", "tau0"),
                  sig_u_hh = sqrt(sw("hh", "s2_u")), sig_r_hh = sqrt(sw("hh", "s2_r")), sig_u_th = sqrt(sw("th", "s2_u")), sig_r_th = sqrt(sw("th", "s2_r")),
                  sig_u_ht = sqrt(sw("ht", "s2_u")), sig_r_ht = sqrt(sw("ht", "s2_r")), sig_u_tt = sqrt(sw("tt", "s2_u")), sig_r_tt = sqrt(sw("tt", "s2_r")),
                  chk_hh_u = chk_hh_u, chk_hh_r = chk_hh_r, sd_null_t = sd_null_t, sum_lam_t = sum(nullK$t$eig),
                  # 3. predicted mean shift of the (h,h) scores
                  sigma_hat_u = sg_u, sigma_hat_r = sg_r, ncp_u = ncp_u, ncp_r = ncp_r, sd_null = sd_null, sum_lam = sum(eig),
                  z_pred_u = ncp_u / sd_null, z_pred_r = ncp_r / sd_null,
                  q_cross_u = q_cross_u, q_noise_u = q_noise_u, q_cross_r = q_cross_r, q_noise_r = q_noise_r, kp_ones = kp_ones,
                  sum_s_u = sum(s_u), sum_s_r = sum(s_r), sum_m_u = sum(m_u), sum_m_r = sum(m_r),
                  # 4. kernel spectrum
                  lam1_share = eig[1] / sum(eig), lam1_sq_share = eig[1]^2 / sum(eig^2), n_eig = length(eig),
                  k_off_mean = k_off_mean, k_off_sd = k_off_sd, cos1 = cos1, cos_sw1 = cos_sw1, q1_u = q1_u, q1_r = q1_r,
                  # 5. stratified permutation test and the per-model settings
                  p_permstr_u = p_permstr_u, p_permstr_r = p_permstr_r, n_strata = n_strata,
                  e_hid = paste(ctrl_e$esCtrl$n.hidden, collapse = "-"), e_batch = ctrl_e$esCtrl$n.batch, e_epoch = ctrl_e$esCtrl$n.epoch, e_pat = ctrl_e$esCtrl$early.stop.det, e_l1 = ctrl_e$esCtrl$l1.reg, e_lr = ctrl_e$esCtrl$learning.rate,
                  mu_hid = paste(ctrl_mu$esCtrl$n.hidden, collapse = "-"), mu_batch = ctrl_mu$esCtrl$n.batch, mu_epoch = ctrl_mu$esCtrl$n.epoch, mu_pat = ctrl_mu$esCtrl$early.stop.det, mu_l1 = ctrl_mu$esCtrl$l1.reg, mu_lr = ctrl_mu$esCtrl$learning.rate,
                  # 6. centered projection
                  rank_cen = qrC$rank, p_dav_u_cen = p_dav_u_cen, p_dav_r_cen = p_dav_r_cen, z_u_cen = z_u_cen, z_r_cen = z_r_cen,
                  lam1_share_cen = eigC[1] / sum(eigC), sd_null_cen = sqrt(2 * sum(eigC^2)),
                  secs_v6 = secs_v6, secs_total = proc.time()[["elapsed"]] - t0)
vec <- list(X = X, Z = Z, Y = Y, e_true = e_true, mu_true = mu_true, mu_star_true = mu_star_true, e_hat = e_hat, mu_hat = mu_hat,
            ys0_hat = ys0_hat, folds = folds, w = w, s_u = s_u, s_r = s_r, pred_u = pred_u, pred_r = pred_r, eig = eig,
            tau0 = tau0, tau0_opt = unname(tau0_opt), bw = bw, x_check = c(sum(X), sum(X^2)))
print(t(res[, !(names(res) %in% c("cfg_id", "rep_id", "tau_type", "n", "p", "sigma"))]))
save(res, vec, file = file.path(OUT_DIR, sprintf("test_%s_c%d_r%d.RData", TAU_TYPE, cfg_id, rep_id)))
