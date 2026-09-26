#' @importFrom stats dist median predict coef lm var rnorm rbinom plogis
#' @importFrom methods is
#' @importFrom CompQuadForm davies liu
#' @importFrom glmnet cv.glmnet
#' @importFrom deepTL importDnnet ensemble_dnnet importTrt
NULL

#' Davies Test for Heterogeneous Treatment Effects
#'
#' The analytic global test of deepHTL: cross-fitted bagged-DNN nuisance estimates, the
#' revised constant effect, and the product-form kernel score test of [kernel_score_test()]
#' with the Davies reference distribution, reported for the unrevised and the revised
#' residual.
#'
#' Nuisance estimation uses `k_folds` folds stratified by treatment arm. Within each
#' training fold [screen_augment()] selects the network inputs (when `screen = TRUE`): the
#' propensity network takes the kept covariates, the outcome networks the kept covariates
#' plus the selected squares and products. The propensity estimates are clipped to
#' `[clip, 1 - clip]`. The constant effect of the revised transformation blends the weighted
#' residual estimator with a linear regression coefficient, the blend chosen by a two-fold
#' cross-validated variance criterion, and the outcome model of `Y - tau0 * Z` is fitted with
#' the same folds and inputs. The kernel test uses all p covariates.
#'
#' @param object An object containing the data (usually class \code{Trt}).
#'                Must contain slots \code{@x}, \code{@y}, and \code{@z}.
#' @param ctrl Optional list. Control parameters of the propensity network (see
#'   [dnn_ctrl()]). `NULL` (default) uses `dnn_ctrl("propensity")`.
#' @param k_folds Integer. Number of folds for cross-fitting nuisance parameters (default 5).
#' @param tune Optional grid of candidate network settings: a data.frame whose columns are
#'   `esCtrl` entries (see [dnn_tune_grid()]) or `TRUE` for the default grid. When given, the
#'   settings of every network are selected within each training fold by the mean out-of-bag
#'   loss of a pilot ensemble, log-loss for the propensity network and squared error for the
#'   outcome networks. `NULL` (default) uses the control lists as supplied, which is what the
#'   paper does.
#' @param n_tune Integer. Networks per candidate in the pilot ensembles. Default 5.
#' @param screen Logical. Select the nuisance inputs within each training fold by
#'   [screen_augment()]. Default `TRUE`.
#' @param ctrl_mu Optional list. Control parameters of the outcome networks. `NULL` (default)
#'   uses `dnn_ctrl("outcome")` when `ctrl` is also `NULL`, otherwise `ctrl`.
#' @param clip Numeric. Propensity estimates are clipped to `[clip, 1 - clip]`. Default 0.05.
#' @param min_keep Integer. Minimum number of covariates kept by the screen. Default 5.
#'
#' @return A list containing:
#' \item{Q}{The revised test statistic.}
#' \item{p_davies}{P-value of the revised test (Davies method, mixture of chi-squares).}
#' \item{tau_hat}{The estimated constant treatment effect (revised).}
#' \item{unrevised, revised}{The [kernel_score_test()] objects of the two residuals.}
#' \item{tau0}{The constant removed by the revised transformation.}
#' \item{nuisance}{The cross-fitted `e_hat`, `mu_hat`, `mu_star_hat` and the fold labels.}
#' \item{screen}{Per-fold number of kept covariates and outcome inputs, `NULL` when `screen = FALSE`.}
#' \item{tuning}{Settings selected in each fold when `tune` is used, otherwise `NULL`.}
#' @export
davies_test <- function(object, ctrl = NULL, k_folds = 5, tune = NULL, n_tune = 5,
                        screen = TRUE, ctrl_mu = NULL, clip = 0.05, min_keep = 5) {
  arms <- trt_arms(object@z)
  z_fac <- arms$z_fac; z_num <- arms$z_num
  x <- object@x
  y <- object@y
  n <- nrow(x)

  if (is.null(ctrl)) {
    ctrl <- dnn_ctrl("propensity")
    if (is.null(ctrl_mu)) ctrl_mu <- dnn_ctrl("outcome")
  }
  if (is.null(ctrl_mu)) ctrl_mu <- ctrl

  K <- k_folds
  folds <- stratified_folds(z_fac, K)
  tune_log <- list()
  sels <- vector("list", K)

  e_hat <- mu_hat <- rep(NA_real_, n)

  for (k in 1:K) {
    tr <- which(folds != k); te <- which(folds == k)

    if (screen) {
      sels[[k]] <- screen_augment(x[tr, , drop = FALSE], y[tr], z_num[tr], min_keep = min_keep)
      Xe <- nuisance_input(sels[[k]], x, augmented = FALSE)
      Xmu <- nuisance_input(sels[[k]], x, augmented = TRUE)
    } else {
      Xe <- Xmu <- x
    }

    z_obj <- deepTL::importDnnet(x = Xe[tr, , drop = FALSE], y = z_fac[tr])
    ctrl_z_mod <- tune_en_dnn_ctrl(z_obj, ctrl, tune, n_tune)
    tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_z_mod, k, "propensity")
    z_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = z_obj), ctrl_z_mod))
    pk <- deepTL::predict(z_mod, Xe[te, , drop = FALSE])
    e_hat[te] <- if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])

    y_obj <- deepTL::importDnnet(x = Xmu[tr, , drop = FALSE], y = y[tr])
    ctrl_y_mod <- tune_en_dnn_ctrl(y_obj, ctrl_mu, tune, n_tune)
    tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_y_mod, k, "outcome")
    y_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = y_obj), ctrl_y_mod))
    mu_hat[te] <- as.numeric(deepTL::predict(y_mod, Xmu[te, , drop = FALSE]))
  }

  e_hat <- pmin(pmax(e_hat, clip), 1 - clip)

  w <- (z_num - e_hat)^2
  tau0 <- sum((y - mu_hat) * (z_num - e_hat)) / sum(w)

  beta1 <- tryCatch(stats::coef(stats::lm(y ~ z_num + e_hat))[2], error=function(e) 0)
  if(is.na(beta1)) beta1 <- 0

  lambdas <- seq(0, 1, length.out = 21)
  best <- list(score = Inf, lam = NA)
  fold_internal <- sample(rep(1:2, length.out = length(z_num)))

  for (lam in lambdas) {
    c_lam <- lam * tau0 + (1 - lam) * beta1
    score <- 0
    for (k in 1:2) {
      idx_tr <- fold_internal != k; idx_te <- fold_internal == k
      ystar_tr <- y[idx_tr] - c_lam * z_num[idx_tr]

      xi_mod <- tryCatch(glmnet::cv.glmnet(as.matrix(x[idx_tr, , drop = FALSE]), ystar_tr, alpha = 1), error=function(e) NULL)

      if(!is.null(xi_mod)) {
        xi_hat_te <- as.numeric(stats::predict(xi_mod, as.matrix(x[idx_te, , drop = FALSE]), s = "lambda.min"))
        zr_te <- z_num[idx_te] - e_hat[idx_te]
        lab_te <- (y[idx_te] - c_lam * z_num[idx_te] - xi_hat_te) / zr_te
        score <- score + stats::var(lab_te, na.rm = TRUE)
      } else {
        score <- Inf
      }
    }
    if (score < best$score) best <- list(score = score, lam = lam)
  }

  if (is.na(best$lam)) best$lam <- 1
  tau0 <- unname(best$lam * tau0 + (1 - best$lam) * beta1)

  ys0_hat <- rep(NA_real_, n)
  Ystar <- y - tau0 * z_num

  for (k in 1:K) {
    tr <- which(folds != k); te <- which(folds == k)
    Xmu <- if (screen) nuisance_input(sels[[k]], x, augmented = TRUE) else x
    ys0_obj <- deepTL::importDnnet(x = Xmu[tr, , drop = FALSE], y = Ystar[tr])
    ctrl_ys0_mod <- tune_en_dnn_ctrl(ys0_obj, ctrl_mu, tune, n_tune)
    tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_ys0_mod, k, "outcome_revised")
    ys0_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = ys0_obj), ctrl_ys0_mod))
    ys0_hat[te] <- as.numeric(deepTL::predict(ys0_mod, Xmu[te, , drop = FALSE]))
  }

  ## product-form kernel score test on the unrevised and the revised residual, sharing the
  ## kernel matrix and the null eigenvalues (they depend on X and e_hat only)
  unrev <- kernel_score_test(y, z_num, e_hat, mu_hat, x, tau0 = 0, return_kernel = TRUE)
  rev <- kernel_score_test(y, z_num, e_hat, ys0_hat, x, tau0 = tau0, bandwidth = unrev$bandwidth,
                           K = unrev$K, lambda = unrev$lambda)
  unrev$K <- NULL

  screen_tab <- if (screen) {
    data.frame(fold = seq_len(K),
               n_vars = vapply(sels, function(s) length(s$vars), integer(1)),
               n_inputs = vapply(sels, function(s) length(s$vars) + sum(s$feats > s$p), integer(1)))
  } else NULL

  list(
    Q = rev$Q,
    p_davies = rev$p_value,
    tau_hat = tau0 + rev$theta_hat,
    unrevised = unrev,
    revised = rev,
    tau0 = tau0,
    nuisance = list(e_hat = e_hat, mu_hat = mu_hat, mu_star_hat = ys0_hat, folds = folds),
    screen = screen_tab,
    tuning = if (length(tune_log)) do.call(rbind, tune_log) else NULL
  )
}
