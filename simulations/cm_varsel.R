###############################################################################
# cm_varsel.R -- covariate-specific screen (marginal + conditional) on the
# correlated-mixed design (cm_dgp.R). Machinery identical to mixed_varsel.R /
# sce_varsel_blocks.R; only the DGP and the grid-cell handling differ.
#   Rscript cm_varsel.R <cfg 1-8> <rep>   env: NPERM=2000 THR=0.2 NBIN=5 CHUNK=50 NCORES SMOKE
###############################################################################
args <- commandArgs(trailingOnly = TRUE); cfg_id <- as.integer(args[1]); rep_id <- as.integer(args[2])
suppressPackageStartupMessages({ library(deepTL); library(MASS); library(parallel); library(ranger) })
source("cm_dgp.R")
SMOKE  <- identical(Sys.getenv("SMOKE"), "1")
OUT_DIR <- Sys.getenv("OUT_DIR", "out/varsel")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
B      <- if (SMOKE) 20L else as.integer(Sys.getenv("NPERM", "2000"))
THR    <- as.numeric(Sys.getenv("THR", "0.2")); NBIN <- as.integer(Sys.getenv("NBIN", "5"))
CHUNK  <- as.integer(Sys.getenv("CHUNK", "50"))
NCORES <- as.integer(Sys.getenv("NCORES", Sys.getenv("SLURM_CPUS_PER_TASK", "1")))
cfg <- CM_CONFIG[cfg_id, ]; n <- cfg$n; d <- cfg$p; sigma <- cfg$sigma; k_folds <- 5
if (SMOKE) n <- 300
ctrl <- cm_ctrl(cm_l1(sigma))
say <- function(...) { cat(...); flush.console() }
say(sprintf("corrmix design cfg %d | n=%d p=%d sigma=%g | B=%d THR=%.2f cores=%d rep=%d l1=%g\n", cfg_id, n, d, sigma, B, THR, NCORES, rep_id, cm_l1(sigma)))

set.seed(100000 * cfg_id + 1000 * rep_id + 23)
dat <- gen_cm(n, d, sigma, "S2"); X <- dat$X; Y <- dat$Y; Z <- dat$Z; tau <- dat$tau
z_fac <- factor(ifelse(Z == 1, "A", "B"), levels = c("A", "B"))

## ---- stage 1: cross-fitted nuisances ----
folds <- sample(rep(seq_len(k_folds), length.out = n))
e_hat <- mu_hat <- numeric(n)
for (k in seq_len(k_folds)) {
  tr <- folds != k; te <- folds == k
  zo <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = z_fac[tr])
  zm <- do.call(deepTL::ensemble_dnnet, c(list(object = zo), ctrl))
  pk <- deepTL::predict(zm, X[te, , drop = FALSE])
  e_hat[te] <- if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])
  yo <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = Y[tr])
  ym <- do.call(deepTL::ensemble_dnnet, c(list(object = yo), ctrl))
  mu_hat[te] <- as.numeric(deepTL::predict(ym, X[te, , drop = FALSE]))
}
e_hat  <- pmin(pmax(e_hat, 0.01), 0.99)
Ytilde <- (Y - mu_hat) / (Z - e_hat); w <- (Z - e_hat)^2

## ---- stage 2: cross-fitted tau-hat, models retained ----
pred_obs <- numeric(n); stage2 <- vector("list", k_folds)
for (k in seq_len(k_folds)) {
  tr <- folds != k; te <- folds == k
  o <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = Ytilde[tr], w = w[tr])
  m <- do.call(deepTL::ensemble_dnnet, c(list(object = o), ctrl))
  pred_obs[te] <- as.numeric(deepTL::predict(m, X[te, , drop = FALSE])); stage2[[k]] <- m
}
obs_loss <- sum(w * (Ytilde - pred_obs)^2) / n
say(sprintf("  L_obs=%.5f cor(tau_hat,tau)=%.3f\n", obs_loss, cor(pred_obs, tau)))

## ---- self-check: batched path reproduces obs_loss ----
id_loss <- local({ rs <- numeric(n)
  for (k in seq_len(k_folds)) { te <- which(folds == k); nt <- length(te); Xte <- X[te, , drop = FALSE]
    pb <- as.numeric(deepTL::predict(stage2[[k]], Xte[rep(seq_len(nt), 2L), , drop = FALSE]))
    rs[te] <- w[te] * (Ytilde[te] - pb[seq_len(nt)])^2 }
  sum(rs) / n })
if (abs(id_loss - obs_loss) / obs_loss > 1e-8) say("  WARNING: batched and unbatched paths differ\n")

## ---- conditioning strata ----
Cx <- cor(X); diag(Cx) <- 0; Cx[!is.finite(Cx)] <- 0
strata <- vector("list", d); n_cond <- integer(d)
for (j in seq_len(d)) {
  S <- which(abs(Cx[j, ]) > THR); n_cond[j] <- length(S)
  if (!length(S)) { strata[[j]] <- rep(1L, n); next }
  rf  <- ranger::ranger(y ~ ., data = data.frame(y = X[, j], X[, S, drop = FALSE]), num.trees = 200, min.node.size = 20)
  fit <- rf$predictions
  qs  <- unique(quantile(fit, seq(0, 1, length.out = NBIN + 1), na.rm = TRUE))
  strata[[j]] <- if (length(qs) > 2) cut(fit, breaks = qs, include.lowest = TRUE, labels = FALSE) else rep(1L, n)
}
say(sprintf("  |S_j| ranges %d-%d; %d of %d covariates conditioned\n", min(n_cond), max(n_cond), sum(n_cond > 0), d))

permute_col <- function(col, str, conditional) {
  nt <- length(col); if (!conditional) return(col[sample(nt)])
  for (b in unique(str[!is.na(str)])) { ii <- which(str == b); if (length(ii) > 1) col[ii] <- col[ii[sample(length(ii))]] }
  col }
perm_losses <- function(j, conditional, nperm) {
  out <- numeric(nperm); done <- 0L
  while (done < nperm) { m <- min(CHUNK, nperm - done); resid_sq <- matrix(0, m, n)
    for (k in seq_len(k_folds)) { te <- which(folds == k); nt <- length(te)
      Xte <- X[te, , drop = FALSE]; str_te <- strata[[j]][te]
      Xbig <- Xte[rep(seq_len(nt), times = m), , drop = FALSE]
      for (b in seq_len(m)) { rows <- ((b - 1) * nt + 1):(b * nt); Xbig[rows, j] <- permute_col(Xte[, j], str_te, conditional) }
      pb <- as.numeric(deepTL::predict(stage2[[k]], Xbig))
      for (b in seq_len(m)) { rows <- ((b - 1) * nt + 1):(b * nt); resid_sq[b, te] <- w[te] * (Ytilde[te] - pb[rows])^2 } }
    out[(done + 1):(done + m)] <- rowSums(resid_sq) / n; done <- done + m }
  out }

t0 <- Sys.time(); types <- cm_type(d)
res <- mclapply(seq_len(d), function(j) {
  set.seed(100000 * cfg_id + 97 * rep_id + j)
  Lm <- perm_losses(j, FALSE, B); Lc <- if (n_cond[j] > 0) perm_losses(j, TRUE, B) else Lm
  data.frame(cfg_id = cfg_id, rep_id = rep_id, n = n, p = d, sigma = sigma, variable = paste0("X", j), type = types[j], group = cm_group(j),
             cor_signal = max(abs(Cx[j, 1:5])), n_cond = n_cond[j],
             p_marg = (1 + sum(Lm <= obs_loss)) / (B + 1), p_cond = (1 + sum(Lc <= obs_loss)) / (B + 1),
             infl_marg = mean(Lm) / obs_loss, infl_cond = mean(Lc) / obs_loss, row.names = NULL)
}, mc.cores = max(1L, min(NCORES, d)), mc.preschedule = FALSE)
varsel <- do.call(rbind, res)
say(sprintf("  screen done in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
print(varsel[, c("variable", "type", "group", "n_cond", "p_marg", "p_cond")], row.names = FALSE)
save(varsel, obs_loss, file = file.path(OUT_DIR, sprintf("varsel_c%d_r%d.RData", cfg_id, rep_id)))
