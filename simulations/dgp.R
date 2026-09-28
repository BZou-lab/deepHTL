# Registry design of the paper: Y = tau(X) Z + f(X) + eps, X from a latent Gaussian with cor rho on X1..X10 and
# X3, X6, X7 binary; tau_type "hte" (Section 5), "tau0" or "tau3" (type I error). One run of the tests, screen and estimator below.
gen_cm <- function(n, p, sigma, tau_type = c("hte", "tau0", "tau3"), rho = 0.3) {
  tau_type <- match.arg(tau_type)
  Sigma <- diag(p); b <- seq_len(min(10, p)); Sigma[b, b] <- rho; diag(Sigma) <- 1
  X <- MASS::mvrnorm(n, rep(0, p), Sigma)
  X[, 3] <- as.numeric(X[, 3] > qnorm(0.6)); X[, 6] <- as.numeric(X[, 6] > qnorm(0.6)); X[, 7] <- as.numeric(X[, 7] > 0)
  f   <- log(abs(X[, 1]) + 1) - X[, 2]^2 + sin(X[, 3]) + 0.5 * X[, 4] * X[, 5]
  e   <- plogis(0.8 * sin(pi * X[, 1] * X[, 2]) + 0.6 * X[, 3] * X[, 4] + 0.5 * tanh(X[, 5]))
  tau <- switch(tau_type, hte = -1 + X[, 1] * X[, 2] + cos(X[, 3])^2 + pmax(X[, 4] - X[, 5], 0),
                tau0 = rep(0, n), tau3 = rep(3, n))
  Z <- rbinom(n, 1, e); Y <- tau * Z + f + rnorm(n, 0, sigma)
  list(X = X, Y = Y, Z = Z, tau = tau)
}

library(deepHTL)
set.seed(1)
d   <- gen_cm(n = 2000, p = 20, sigma = 1, tau_type = "hte")
obj <- importTrt(d$X, d$Y, d$Z)

fit_ks <- davies_test(obj)                                    # kernel score test
fit_pt <- cv_perm_test(obj, B = 2000)                         # permutation test
c(kernel = fit_ks$p_davies, permutation = fit_pt$revised$p_value)

vs <- cv_perm_vsel(obj, k_folds = 5, B = 2000, n_cores = 4)   # covariate-specific screen
vs$screen[order(vs$screen$p_marg), c("variable", "p_marg")]

fit  <- weight_dnn(obj)                                       # deepHTL estimator
test <- gen_cm(n = 2000, p = 20, sigma = 1, tau_type = "hte")
log(mean((predict(fit, test$X, which = "revised") - test$tau)^2))   # log-MSE on an independent test set
