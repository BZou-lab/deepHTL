###############################################################################
# cm_dgp.R -- the paper's Section 3 data-generating process. Default = registry
# design; CM_GAUSS=1 = benchmark design. "Correlated mixed" design for deepHTL: the Section 3 DGP
# (f, e, tau unchanged, Y = tau(X) Z + f(X) + eps) on a covariate vector with
#   * an equicorrelated block, cor(W_j, W_k) = CM_RHO (0.3) for 1 <= j != k <= 10,
#     independence elsewhere (latent Gaussian W);
#   * X3 binary with prevalence 0.4 (a modifier and a confounder),
#     X6 binary 0.4 and X7 binary 0.5 (correlated nulls), X_j = 1{W_j > q_j};
#   * every other coordinate X_j = W_j (standard normal).
# tau_type: tau0 (0), tau3 (3), S1 (simple HTE), S2 (complex HTE).
###############################################################################
CM_RHO <- as.numeric(Sys.getenv("CM_RHO", "0.3"))
CM_BIN <- c(`3` = 0.4, `6` = 0.4, `7` = 0.5)
# CM_GAUSS=1 reproduces the paper's original design exactly: X ~ N(0, I_p),
# no binarization (rho forced to 0, CM_BIN emptied). f, e, tau untouched.
CM_GAUSS <- identical(Sys.getenv("CM_GAUSS"), "1")
if (CM_GAUSS) { CM_RHO <- 0; CM_BIN <- c() }
# CM_NOBIN=1 keeps the correlation block but drops the binarization (the benchmark
# design with correlated Gaussian covariates, used for Web Table 7 with CM_RHO).
if (identical(Sys.getenv("CM_NOBIN"), "1")) CM_BIN <- c()

cm_X <- function(n, p) {
  Sigma <- diag(p); b <- seq_len(min(10, p)); Sigma[b, b] <- CM_RHO; diag(Sigma) <- 1
  W <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma); X <- W
  for (j in names(CM_BIN)) { jj <- as.integer(j); X[, jj] <- as.numeric(W[, jj] > qnorm(1 - CM_BIN[[j]])) }   # no-op when CM_BIN is empty
  colnames(X) <- paste0("X", seq_len(p)); X
}
cm_f   <- function(X) log(abs(X[,1]) + 1) - X[,2]^2 + sin(X[,3]) + 0.5 * X[,4] * X[,5]
cm_e   <- function(X) plogis(0.8 * sin(pi * X[,1] * X[,2]) + 0.6 * X[,3] * X[,4] + 0.5 * tanh(X[,5]))
cm_tau <- function(X, tau_type) switch(tau_type,
  tau0 = rep(0, nrow(X)), tau3 = rep(3, nrow(X)),
  S1   = -1 + 2 * log(exp(rowSums(X[, 1:5])) + 1),
  S2   = -1 + X[,1] * X[,2] + cos(X[,3])^2 + pmax(X[,4] - X[,5], 0),
  stop("tau_type must be tau0, tau3, S1 or S2"))
gen_cm <- function(n, p, sigma, tau_type) {
  X <- cm_X(n, p); tau <- cm_tau(X, tau_type); e <- cm_e(X); Z <- rbinom(n, 1, e)
  Y <- tau * Z + cm_f(X) + rnorm(n, 0, sigma)
  list(X = X, Y = Y, Z = Z, tau = tau, e = e)
}
# grid: 8 configurations, n fastest, then p, then sigma (cfg 1 = n1000 p20 s1 ... cfg 8 = n2000 p40 s3)
CM_CONFIG <- expand.grid(n = c(1000, 2000), p = c(20, 40), sigma = c(1, 3), KEEP.OUT.ATTRS = FALSE)
cm_group <- function(j) if (j <= 5) "signal" else if (j <= 10) "null_corr" else "null_indep"
cm_type  <- function(p) { ty <- rep("continuous", p); if (length(CM_BIN)) ty[as.integer(names(CM_BIN))] <- "binary"; ty }
# DNN control identical to the paper's runs; SMOKE=1 shrinks it for smoke tests
# l1: the paper's permutation-test scripts use 1e-5 at sigma = 1 and 1e-3 at
# sigma = 3 (cm_l1); weight_dnn (estimation) uses 1e-3 throughout.
cm_l1 <- function(sigma) if (sigma == 1) 1e-5 else 1e-3
cm_ctrl <- function(l1 = 1e-3) {
  smoke <- identical(Sys.getenv("SMOKE"), "1")
  list(n.ensemble = if (smoke) 2 else 30, verbose = FALSE,
       esCtrl = list(n.hidden = c(128, 64, 32), n.batch = 256, n.epoch = if (smoke) 5 else 120,
                     norm.x = TRUE, norm.y = TRUE, activate = "relu", accel = "rcpp",
                     l1.reg = l1, plot = FALSE, learning.rate.adaptive = "adam", early.stop.det = 20))
}
