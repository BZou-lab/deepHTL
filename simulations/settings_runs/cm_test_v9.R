###############################################################################
# cm_test_v9.R -- 2026-09-22. cm_test_v6.R plus DATA-DRIVEN SELECTION of the bagged-DNN settings (pilot of the paper's
# `tune` mechanism). Same data seeds as cm_test.R / v6 / B64, so cells are paired on the data. For EVERY ensemble fit
# (e, mu, mu*, stage-2 unrevised/revised, in every fold) the L1 penalty, mini-batch size and epoch cap are chosen from the
# grid CM_TUNE_L1 x CM_TUNE_BATCH x CM_TUNE_EPOCH by the mean per-network OUT-OF-BAG gain (deepTL's @loss slot: null loss
# minus model loss on the left-out observations, log-likelihood for e, squared error for mu / mu*, weighted squared error =
# R-loss for stage 2, no penalty term) of a CM_TUNE_NENS-network pilot ensemble. All candidates are fitted from the same
# RNG state (same bootstrap samples, same initial weights); the stream is restored afterwards. Selections, the gain gap to
# the default setting, the rank of the default and the mean best epoch are saved in `sel` next to `res`.
# Unlike cm_test_v5.R (cross-validated R-LOSS for the nuisances, over-rejected): stage 1 is scored by its own predictive loss.
# env (optional): CM_TUNE (1) CM_TUNE_L1 CM_TUNE_BATCH CM_TUNE_EPOCH CM_TUNE_NENS, plus the v6 variables.
#   Rscript cm_test_v9.R <cfg 1-8> <rep>
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
if (SMOKE) { n <- 300; B <- 50; BK <- 50 }
ctrl <- cm_ctrl(L1); if (!SMOKE) { ctrl$esCtrl$n.epoch <- NEP; ctrl$esCtrl$early.stop.det <- PAT
  ctrl$esCtrl$n.batch <- NBAT; ctrl$n.ensemble <- NENS }
cat(sprintf("cfg %d: n=%d p=%d sigma=%g | tau=%s | rep %d | B=%d BK=%d l1=%g\n", cfg_id, n, p, sigma, TAU_TYPE, rep_id, B, BK, L1))
TUNE <- identical(Sys.getenv("CM_TUNE", "1"), "1")
num <- function(v, d) as.numeric(strsplit(Sys.getenv(v, d), ",")[[1]])
tg <- expand.grid(l1.reg = num("CM_TUNE_L1", "1e-7,1e-5,1e-3"), n.batch = as.integer(num("CM_TUNE_BATCH", "64,128,256")),
                  n.epoch = as.integer(num("CM_TUNE_EPOCH", "120,300")), KEEP.OUT.ATTRS = FALSE)
TN <- as.integer(Sys.getenv("CM_TUNE_NENS", "5"))
if (SMOKE) { tg <- tg[tg$n.batch != 128 & tg$n.epoch == min(tg$n.epoch), ]; tg$n.epoch <- 5L; TN <- 2L }
def_i <- which(tg$l1.reg == L1 & tg$n.batch == ctrl$esCtrl$n.batch & tg$n.epoch == ctrl$esCtrl$n.epoch); if (!length(def_i)) def_i <- NA
cat(sprintf("tune=%d candidates=%d pilot nets=%d default in grid=%s\n", TUNE, nrow(tg), TN, !is.na(def_i)))
sel_log <- list()
best_epoch <- function(mod) mean(vapply(mod@model.list, function(m) which.min(m@loss.traj), numeric(1)), na.rm = TRUE)
cat(sprintf("epochs=%d patience=%d clip=%.2f K=%d batch=%d ensemble=%d\n", ctrl$esCtrl$n.epoch, ctrl$esCtrl$early.stop.det, CLIP, K, ctrl$esCtrl$n.batch, ctrl$n.ensemble))
set.seed(100000 * cfg_id + 1000 * rep_id + 7)
d <- gen_cm(n, p, sigma, TAU_TYPE); X <- d$X; Y <- d$Y; Z <- d$Z
overlap <- mean(d$e < 0.05 | d$e > 0.95)
z_fac <- factor(ifelse(Z == 1, "A", "B"), levels = c("A", "B"))

## ---- stratified folds ----
folds <- integer(n)
for (lev in c(0, 1)) { ix <- which(Z == lev); folds[ix] <- sample(rep(1:K, length.out = length(ix))) }
fit_pred <- function(x, y, xnew, w = NULL, label = "", fold = NA) {
  obj <- if (is.null(w)) deepTL::importDnnet(x = x, y = y) else deepTL::importDnnet(x = x, y = y, w = w)
  ct <- ctrl
  if (TUNE) {
    seed <- sample.int(.Machine$integer.max, 1L); rs <- get(".Random.seed", envir = globalenv())
    gain <- bep <- rep(NA_real_, nrow(tg))
    for (i in seq_len(nrow(tg))) { cti <- ctrl; cti$n.ensemble <- TN; cti$verbose <- FALSE
      cti$esCtrl$l1.reg <- tg$l1.reg[i]; cti$esCtrl$n.batch <- tg$n.batch[i]; cti$esCtrl$n.epoch <- tg$n.epoch[i]
      set.seed(seed); m <- tryCatch(do.call(deepTL::ensemble_dnnet, c(list(object = obj), cti)), error = function(e) NULL)
      if (!is.null(m)) { l <- m@loss[is.finite(m@loss)]; gain[i] <- if (length(l)) mean(l) else NA; bep[i] <- best_epoch(m) } }
    assign(".Random.seed", rs, envir = globalenv())
    b <- which.max(gain); ct$esCtrl$l1.reg <- tg$l1.reg[b]; ct$esCtrl$n.batch <- tg$n.batch[b]; ct$esCtrl$n.epoch <- tg$n.epoch[b]
    sel_log[[length(sel_log) + 1]] <<- data.frame(model = label, fold = fold, l1 = tg$l1.reg[b], batch = tg$n.batch[b], epoch = tg$n.epoch[b],
      gain = gain[b], gain_default = if (is.na(def_i)) NA else gain[def_i], rank_default = if (is.na(def_i)) NA else rank(-gain, na.last = "keep")[def_i],
      n_ok = sum(!is.na(gain)), best_epoch_pilot = bep[b], best_epoch_final = NA_real_)
  }
  mod <- do.call(deepTL::ensemble_dnnet, c(list(object = obj), ct))
  if (TUNE) sel_log[[length(sel_log)]]$best_epoch_final <<- best_epoch(mod)
  pk <- deepTL::predict(mod, xnew)
  if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])
}

## ---- stage 1: e_hat, mu_hat ----
e_hat <- mu_hat <- numeric(n)
for (k in 1:K) { tr <- folds != k; te <- folds == k
  e_hat[te]  <- fit_pred(X[tr, ], z_fac[tr], X[te, ], label = "e", fold = k)
  mu_hat[te] <- fit_pred(X[tr, ], Y[tr],     X[te, ], label = "mu", fold = k) }
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
for (k in 1:K) { tr <- folds != k; te <- folds == k; ys0_hat[te] <- fit_pred(X[tr, ], Ystar[tr], X[te, ], label = "mu_star", fold = k) }
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
  pred_u[te] <- fit_pred(X[tr, ], Ytilde[tr],     X[te, ], w = w[tr], label = "stage2_u", fold = k)
  pred_r[te] <- fit_pred(X[tr, ], tildeYstar[tr], X[te, ], w = w[tr], label = "stage2_r", fold = k) }
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

res <- data.frame(cfg_id = cfg_id, rep_id = rep_id, tau_type = TAU_TYPE, n = n, p = p, sigma = sigma, overlap_out = overlap,
                  l1 = L1, n_epoch = ctrl$esCtrl$n.epoch, patience = ctrl$esCtrl$early.stop.det, clip = CLIP, K = K,
                  n_batch = ctrl$esCtrl$n.batch, n_ens = ctrl$n.ensemble, bw = bw, rank_proj = qrU$rank,
                  tau0 = tau0, tau0_opt = tau0_opt, lam = best$lam,
                  p_davies_u = p_davies_u, p_davies_r = p_davies_r, p_score_u = p_score_u, p_score_r = p_score_r,
                  p_perm_u = p_perm_u, p_perm_r = p_perm_r,
                  p_kperm_u = pk[["kperm_u"]], p_kperm_r = pk[["kperm_r"]], p_kperms_u = pk[["kperms_u"]], p_kperms_r = pk[["kperms_r"]],
                  p_proj_u = p_proj_u, p_proj_r = p_proj_r, p_projk_u = pk[["projk_u"]], p_projk_r = pk[["projk_r"]],
                  tune = as.integer(TUNE), n_cand = nrow(tg), n_pilot = TN)
sel <- if (length(sel_log)) do.call(rbind, sel_log) else NULL
if (!is.null(sel)) { sel$cfg_id <- cfg_id; sel$rep_id <- rep_id; sel$tau_type <- TAU_TYPE
  for (m in unique(sel$model)) { s <- sel[sel$model == m, ]; res[[paste0("sel_", m)]] <- paste(sprintf("%g/%d/%d", s$l1, s$batch, s$epoch), collapse = " ") } }
print(res, row.names = FALSE); if (!is.null(sel)) print(sel[, c("model","fold","l1","batch","epoch","gain","gain_default","rank_default","best_epoch_pilot","best_epoch_final")], row.names = FALSE)
save(res, sel, file = file.path(OUT_DIR, sprintf("test_%s_c%d_r%d.RData", TAU_TYPE, cfg_id, rep_id)))
