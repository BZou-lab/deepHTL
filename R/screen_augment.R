#' @importFrom glmnet cv.glmnet
#' @importFrom stats coef cor
#' @importFrom utils combn
NULL

#' Hierarchical Lasso Screen with Interaction-Augmented Inputs
#'
#' Selects the inputs of the nuisance networks within a training set. A lasso is fitted
#' for `Y` (gaussian) and for `Z` (binomial) on the covariates, their squares and all
#' pairwise products, with the penalty chosen by `nfolds`-fold cross-validation
#' (`lambda.min`). Every covariate that appears in a selected feature is kept, and when fewer
#' than `min_keep` covariates are kept, the set is filled up with the covariates most
#' correlated with `Y`. The outcome networks take the kept covariates together with the
#' selected squares and products as inputs ([nuisance_input()] with `augmented = TRUE`), the
#' propensity network the kept covariates only (`augmented = FALSE`). The kernel score test and
#' the second-stage effect model use all p covariates.
#'
#' A main-effect screen misses covariates that act only through squares or products, which
#' left the outcome network undertrained in the simulations of the paper. With this screen
#' and the outcome budget of [dnn_ctrl()], the root mean squared error of the outcome model
#' fell from 1.25 to 0.58 in the hardest configuration (n = 1000, p = 40, sigma = 1).
#'
#' @param X Numeric matrix of covariates (training rows).
#' @param Y Numeric outcome.
#' @param Z Treatment indicator (0/1, or a factor whose first level is the treated arm).
#' @param min_keep Integer. Minimum number of covariates kept. Default 5.
#' @param nfolds Integer. Folds of the cross-validation choosing the lasso penalty. Default 10.
#'
#' @return An object of class `screen_augment`, a list with `vars` the indices of the kept
#'   covariates, `feats` the indices of the selected columns of the dictionary
#'   `[X, X^2, X_i X_j]` (columns 1 to p linear, p + 1 to 2p squares, then the products in
#'   the order of `pairs`), `pairs` the 2 x p(p - 1)/2 index matrix of the products, `p` and
#'   `labels` the names of the selected columns.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 400; p <- 8
#' X <- matrix(rnorm(n * p), n, p)
#' Z <- rbinom(n, 1, plogis(0.6 * X[, 3] * X[, 4]))
#' Y <- X[, 1] - X[, 2]^2 + 0.5 * X[, 4] * X[, 5] + Z + rnorm(n)
#' sel <- screen_augment(X, Y, Z)
#' sel$labels
#' dim(nuisance_input(sel, X))                      # outcome inputs
#' dim(nuisance_input(sel, X, augmented = FALSE))   # propensity inputs
#' }
#' @export
screen_augment <- function(X, Y, Z, min_keep = 5, nfolds = 10) {
  X <- as.matrix(X); p <- ncol(X)
  Y <- as.numeric(Y)
  Z <- if (is.factor(Z)) as.numeric(Z == levels(Z)[1]) else as.numeric(Z)
  if (length(Y) != nrow(X) || length(Z) != nrow(X)) stop("X, Y and Z must have the same number of rows")

  pairs <- if (p >= 2) utils::combn(p, 2) else matrix(integer(0), 2, 0)
  Fall <- augment_dictionary(X, pairs)
  fvars <- c(as.list(seq_len(p)), as.list(seq_len(p)),
             lapply(seq_len(ncol(pairs)), function(k) pairs[, k]))

  lasso_nz <- function(y, fam) {
    cf <- tryCatch(as.numeric(stats::coef(glmnet::cv.glmnet(Fall, y, family = fam, alpha = 1, nfolds = nfolds),
                                          s = "lambda.min"))[-1],
                   error = function(e) rep(0, ncol(Fall)))
    which(cf != 0)
  }
  feats <- sort(unique(c(lasso_nz(Y, "gaussian"), lasso_nz(Z, "binomial"))))
  vars <- sort(unique(unlist(fvars[feats])))
  if (length(vars) < min_keep) {
    top <- order(-abs(as.numeric(stats::cor(X, Y))))[seq_len(min(min_keep, p))]
    vars <- sort(unique(c(vars, top)))
  }

  nm <- colnames(X); if (is.null(nm)) nm <- paste0("X", seq_len(p))
  lab_all <- c(nm, paste0(nm, "^2"), if (ncol(pairs)) paste0(nm[pairs[1, ]], ":", nm[pairs[2, ]]))
  structure(list(vars = vars, feats = feats, pairs = pairs, p = p,
                 labels = list(vars = nm[vars], feats = lab_all[feats])),
            class = "screen_augment")
}

#' @export
print.screen_augment <- function(x, ...) {
  n_sq <- sum(x$feats > x$p & x$feats <= 2 * x$p); n_pr <- sum(x$feats > 2 * x$p)
  cat(sprintf("screen_augment: %d of %d covariates kept (%s)\n", length(x$vars), x$p,
              paste(x$labels$vars, collapse = ", ")))
  cat(sprintf("  selected features: %d linear, %d squares, %d products\n",
              sum(x$feats <= x$p), n_sq, n_pr))
  invisible(x)
}

#' Inputs of the Nuisance Networks after the Screen
#'
#' Builds the input matrix of a nuisance network for new covariate rows from a
#' [screen_augment()] object.
#'
#' @param screen An object returned by [screen_augment()].
#' @param X Numeric matrix with the same p columns as the screened covariates.
#' @param augmented Logical. `TRUE` (default) returns the kept covariates followed by the
#'   selected squares and products (the inputs of the outcome models), `FALSE` the kept
#'   covariates only (the inputs of the propensity model).
#'
#' @return A numeric matrix with `nrow(X)` rows.
#' @export
nuisance_input <- function(screen, X, augmented = TRUE) {
  if (!inherits(screen, "screen_augment")) stop("screen must be a screen_augment object")
  X <- as.matrix(X); p <- screen$p
  if (ncol(X) != p) stop(sprintf("X must have %d columns", p))
  out <- X[, screen$vars, drop = FALSE]
  colnames(out) <- screen$labels$vars
  if (!augmented) return(out)
  eng <- screen$feats[screen$feats > p]
  if (!length(eng)) return(out)
  Fe <- vapply(eng, function(f) {
    if (f <= 2 * p) X[, f - p]^2
    else { k <- f - 2 * p; X[, screen$pairs[1, k]] * X[, screen$pairs[2, k]] }
  }, numeric(nrow(X)))
  Fe <- matrix(Fe, nrow = nrow(X))
  colnames(Fe) <- screen$labels$feats[screen$feats > p]
  cbind(out, Fe)
}

# Dictionary [X, X^2, X_i X_j] used by the screen (pairs from utils::combn(p, 2)).
augment_dictionary <- function(X, pairs) {
  cbind(X, X^2, X[, pairs[1, ], drop = FALSE] * X[, pairs[2, ], drop = FALSE])
}

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
