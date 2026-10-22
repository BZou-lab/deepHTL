#' Recommended Control Lists for the Bagged-DNN
#'
#' The training budgets used in the paper. Every network has hidden layers of 128, 64 and
#' 32 ReLU units, Adam with learning rate 1e-3, standardized inputs and outputs, bootstrap
#' training with the weights of the best out-of-bag epoch kept, and an L1 penalty. The
#' outcome networks (the regressions of `Y` and of `Y - tau0 * Z` on the covariates) use
#' mini-batches of 128, at most 500 epochs and patience 50. The propensity network, and the
#' second-stage effect networks, use mini-batches of 256, at most 120 epochs and patience 20.
#'
#' @param model `"outcome"` or `"propensity"`.
#' @param l1 L1 penalty. Default 1e-3. The simulations of the paper used 1e-5 at noise level
#'   sigma = 1 and 1e-3 at sigma = 3.
#' @param n_ensemble Number of networks in the ensemble. Default 30.
#' @param n_hidden Integer vector of hidden layer sizes.
#' @param learning_rate Adam learning rate. Default 1e-3.
#'
#' @return A list with `n.ensemble`, `verbose` and `esCtrl`, as taken by
#'   `deepTL::ensemble_dnnet()` and by the `ctrl` arguments of [davies_test()],
#'   [cv_perm_test()], [weight_dnn()] and [cv_perm_vsel()].
#' @export
dnn_ctrl <- function(model = c("outcome", "propensity"), l1 = 1e-3, n_ensemble = 30,
                     n_hidden = c(128, 64, 32), learning_rate = 1e-3) {
  model <- match.arg(model)
  budget <- if (model == "outcome") list(n.batch = 128, n.epoch = 500, early.stop.det = 50)
            else list(n.batch = 256, n.epoch = 120, early.stop.det = 20)
  list(n.ensemble = n_ensemble, verbose = FALSE,
       esCtrl = list(n.hidden = n_hidden, n.batch = budget$n.batch, n.epoch = budget$n.epoch,
                     norm.x = TRUE, norm.y = TRUE, activate = "relu", accel = "rcpp",
                     l1.reg = l1, plot = FALSE, learning.rate.adaptive = "adam",
                     learning.rate = learning_rate, early.stop.det = budget$early.stop.det))
}

.ctrl1 <- function(l1 = 3e-4) {
  ct <- dnn_ctrl("outcome", l1 = l1, n_hidden = c(20, 18, 16, 14, 12, 10))
  ct$esCtrl$n.batch <- 64
  ct$esCtrl$n.epoch <- 1000
  ct
}

.x_in <- function(X) {
  X <- as.matrix(X)
  j <- which(apply(X, 2, function(v) length(unique(v)) > 2))
  if (!length(j)) return(X)
  S <- scale(X[, j, drop = FALSE])^2
  nm <- colnames(X); if (is.null(nm)) nm <- paste0("X", seq_len(ncol(X)))
  colnames(S) <- paste0(nm[j], "_2")
  cbind(X, S)
}

# Folds stratified by treatment arm: every arm is spread evenly over the K folds.
stratified_folds <- function(z, K) {
  z <- as.factor(z)
  folds <- integer(length(z))
  for (lev in levels(z)) {
    ix <- which(z == lev)
    folds[ix] <- if (length(ix) < K) sample(seq_len(K), length(ix), replace = TRUE)
                 else sample(rep(seq_len(K), length.out = length(ix)))
  }
  folds
}

# Treatment arm coding shared by the tests: the factor level "A" (or the value 1) is treated.
trt_arms <- function(z) {
  z_fac <- if (is.factor(z)) z else factor(ifelse(z == 1, "A", "B"), levels = c("A", "B"))
  list(z_fac = z_fac, z_num = as.numeric(z_fac == "A"))
}
