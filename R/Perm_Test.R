#' @title Sample-Split Permutation Test for Treatment Heterogeneity
#' 
#' @description Performs a sample-split permutation test of treatment effect heterogeneity.
#' The nuisance models are cross-fitted on the full sample, the second-stage effect networks
#' are fitted on one half of the sample (S1), and the observed loss on the other half (S2) is
#' compared with the losses after shuffling the S2 predictions within treatment arm.
#' 
#' Nuisance estimation follows [davies_test()]: folds stratified by treatment arm, propensity
#' estimates clipped to `[clip, 1 - clip]`, and the revised constant chosen by the blended
#' estimator. The halves are drawn within treatment arm and the second-stage effect networks
#' use all p covariates.
#'
#' @param object An object of class `Trt` containing the covariates, outcome, and treatment assignment.
#' @param k_folds Integer. The number of folds for cross-fitting the nuisance models. Default is 5.
#' @param B Integer. The number of permutation shuffles to perform. Default is 1000.
#' @param en_dnn_ctrl A list of control parameters for the `ensemble_dnnet` function, used
#'   for the second-stage effect networks (see [dnn_ctrl()]).
#'   `NULL` (default) uses `dnn_ctrl("propensity", l1 = 1e-5)`.
#' @param tune Optional grid of candidate network settings: a data.frame whose columns are
#'   `esCtrl` entries (see [dnn_tune_grid()]) or `TRUE` for the default grid. When given, the
#'   settings of every network are selected within each training fold by the mean out-of-bag
#'   loss of a pilot ensemble, log-loss for the propensity network, squared error for the
#'   outcome networks and weighted squared error (the R-loss on held-out data) for the
#'   second-stage networks. `NULL` (default) uses the control list as supplied.
#' @param n_tune Integer. Networks per candidate in the pilot ensembles. Default 5.
#' @param ctrl_mu Optional list. Control parameters of the propensity and outcome networks.
#'   `NULL` (default) uses hidden layers 20-18-16-14-12-10, mini-batches of 64, at most 1000
#'   epochs, patience 50 and L1 penalty 3e-4.
#' @param clip Numeric. Propensity estimates are clipped to `[clip, 1 - clip]`. Default 0.05.
#' @param split_frac Numeric. Share of each treatment arm assigned to the training half S1.
#'   Default 0.5.
#' @param groups Optional vector of group labels (e.g. donor id), one per subject. Members of a
#'   group are kept in the same half and in the same cross-fitting fold. `NULL` (default)
#'   treats every subject as its own group.
#'
#' @return A list containing two sub-lists (`unrevised` and `revised`), each providing the
#' observed weighted mean squared error on S2 (`obs_mse`) and the resulting permutation p-value
#' (`p_value`), `tau0` the constant removed by the revised transformation, `n_train` and
#' `n_test` the sizes of S1 and S2, and `tuning`, the settings selected in each fold
#' (fold `NA` for the stage-2 fits on S1) when `tune` is used (`NULL` otherwise).
#'
#' @importFrom stats coef lm predict var
#' @importFrom glmnet cv.glmnet
#' @export
cv_perm_test <- function(object, k_folds = 5, B = 1000, en_dnn_ctrl = NULL, tune = NULL, n_tune = 5,
                         ctrl_mu = NULL, clip = 0.05, split_frac = 0.5, groups = NULL) {
  arms <- trt_arms(object@z)
  z_fac <- arms$z_fac; z_num <- arms$z_num

  X <- object@x
  y <- object@y
  n <- nrow(X)
  check_split_args(n, split_frac, groups)

  if (is.null(en_dnn_ctrl)) en_dnn_ctrl <- dnn_ctrl("propensity", l1 = 1e-5)
  if (is.null(ctrl_mu)) ctrl_mu <- .ctrl1()

  folds <- if (is.null(groups)) stratified_folds(z_fac, k_folds) else
    group_parts(z_num, groups, seq_len(k_folds - 1) / k_folds)
  e_hat <- mu_hat <- rep(NA_real_, n)
  tune_log <- list()
  Xmu <- .x_in(X)

  for (k in 1:k_folds) {
    tr <- folds != k
    te <- folds == k

    z_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = z_fac[tr])
    ctrl_z_mod <- tune_en_dnn_ctrl(z_obj, ctrl_mu, tune, n_tune)
    tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_z_mod, k, "propensity")
    z_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = z_obj), ctrl_z_mod))
    pk <- deepTL::predict(z_mod, X[te, , drop = FALSE])
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
  
  beta1 <- tryCatch({
    mod <- stats::lm(as.numeric(y) ~ as.numeric(z_num) + as.numeric(e_hat))
    b1 <- stats::coef(mod)
    if (length(b1) == 0 || is.na(b1)) 0 else as.numeric(b1)
  }, error = function(e) 0)
  
  lambdas <- seq(0, 1, length.out = 21)
  best <- list(score = Inf, lam = NA)
  fold_internal <- sample(rep(1:2, length.out = length(z_num)))
  
  for (lam in lambdas) {
    c_lam <- lam * tau0 + (1 - lam) * beta1
    score <- 0
    for (k_int in 1:2) {
      idx_tr <- fold_internal != k_int
      idx_te <- fold_internal == k_int
      ystar_tr <- y[idx_tr] - c_lam * z_num[idx_tr]
      
      xi_mod <- tryCatch(glmnet::cv.glmnet(as.matrix(X[idx_tr, , drop = FALSE]), ystar_tr, alpha = 1), error = function(e) NULL)
      
      if(!is.null(xi_mod)) {
        xi_hat_te <- as.numeric(stats::predict(xi_mod, as.matrix(X[idx_te, , drop = FALSE]), s = "lambda.min"))
        zr_te <- z_num[idx_te] - e_hat[idx_te]
        lab_te <- (y[idx_te] - c_lam * z_num[idx_te] - xi_hat_te) / zr_te
        score <- score + stats::var(lab_te, na.rm = TRUE)
      } else {
        score <- Inf
      }
    }
    if (score < best$score) best <- list(score = score, lam = lam)
  }
  
  opt_lam <- if(is.na(best$lam)) 1 else best$lam
  tau0_opt <- opt_lam * tau0 + (1 - opt_lam) * beta1
  
  ys0_hat <- rep(NA_real_, n)
  Ystar <- y - tau0_opt * z_num
  
  for (k in 1:k_folds) {
    tr <- folds != k
    te <- folds == k
    
    ys0_obj <- deepTL::importDnnet(x = Xmu[tr, , drop = FALSE], y = Ystar[tr])
    ctrl_ys0_mod <- tune_en_dnn_ctrl(ys0_obj, ctrl_mu, tune, n_tune)
    tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_ys0_mod, k, "outcome_revised")
    ys0_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = ys0_obj), ctrl_ys0_mod))
    ys0_hat[te] <- as.numeric(deepTL::predict(ys0_mod, Xmu[te, , drop = FALSE]))
  }

  Ytilde_u <- (y - mu_hat) / (z_num - e_hat)
  Ytilde_r <- (y - tau0_opt * z_num - ys0_hat) / (z_num - e_hat)
  
  ## sample split within arm: stage-2 networks fitted on S1, loss and permutations on S2
  S1 <- group_parts(z_num, groups, split_frac) == 1L
  S2 <- !S1
  n1 <- sum(S1)
  n2 <- sum(S2)
  if (n1 == 0 || n2 == 0) stop("both halves of the sample split must be non-empty")

  obj_u <- deepTL::importDnnet(x = X[S1, , drop = FALSE], y = Ytilde_u[S1], w = w[S1])
  ctrl_mod_u <- tune_en_dnn_ctrl(obj_u, en_dnn_ctrl, tune, n_tune)
  tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_mod_u, NA_integer_, "stage2_unrevised")
  mod_u <- do.call(deepTL::ensemble_dnnet, c(list(object = obj_u), ctrl_mod_u))
  pred_u <- as.numeric(deepTL::predict(mod_u, X[S2, , drop = FALSE]))

  obj_r <- deepTL::importDnnet(x = X[S1, , drop = FALSE], y = Ytilde_r[S1], w = w[S1])
  ctrl_mod_r <- tune_en_dnn_ctrl(obj_r, en_dnn_ctrl, tune, n_tune)
  tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_mod_r, NA_integer_, "stage2_revised")
  mod_r <- do.call(deepTL::ensemble_dnnet, c(list(object = obj_r), ctrl_mod_r))
  pred_r <- as.numeric(deepTL::predict(mod_r, X[S2, , drop = FALSE]))

  yu <- Ytilde_u[S2]
  yr <- Ytilde_r[S2]
  w2 <- w[S2]
  obs_mse_u <- sum(w2 * (yu - pred_u)^2) / n2
  obs_mse_r <- sum(w2 * (yr - pred_r)^2) / n2
  
  perm_mse_u <- numeric(B)
  perm_mse_r <- numeric(B)
  
  arm_idx <- list(which(z_num[S2] == 0), which(z_num[S2] == 1))
  
  for (b in 1:B) {
    pred_u_perm <- pred_u
    pred_r_perm <- pred_r
    
    # one within-arm shuffle shared by both models
    for (ix in arm_idx) {
      shuf <- ix[sample.int(length(ix))]
      pred_u_perm[ix] <- pred_u[shuf]
      pred_r_perm[ix] <- pred_r[shuf]
    }
    
    perm_mse_u[b] <- sum(w2 * (yu - pred_u_perm)^2) / n2
    perm_mse_r[b] <- sum(w2 * (yr - pred_r_perm)^2) / n2
  }
  
  p_val_u <- (sum(perm_mse_u <= obs_mse_u) + 1) / (B + 1)
  p_val_r <- (sum(perm_mse_r <= obs_mse_r) + 1) / (B + 1)
  
  list(
    unrevised = list(obs_mse = obs_mse_u, p_value = p_val_u),
    revised = list(obs_mse = obs_mse_r, p_value = p_val_r),
    tau0 = unname(tau0_opt),
    n_train = n1,
    n_test = n2,
    tuning = if (length(tune_log)) do.call(rbind, tune_log) else NULL
  )
}

# Part labels 1, 2, ... that keep every group whole and are stratified by treatment arm: within
# each arm the groups (single subjects when `groups` is NULL) are taken in random order and cut
# where the cumulative share of the arm's subjects passes `cuts`. A group with members in both
# arms goes with the arm of most of its members. `cuts = split_frac` gives the halves S1 (1) and
# S2 (2), `cuts = (1:(K - 1)) / K` gives K cross-fitting folds.
group_parts <- function(z_num, groups, cuts) {
  g <- if (is.null(groups)) seq_along(z_num) else match(groups, unique(groups))
  size <- tabulate(g)
  arm <- as.numeric(tapply(z_num, g, mean) > 0.5)
  part <- integer(length(size))
  for (a in c(0, 1)) {
    ga <- which(arm == a)
    ga <- ga[sample.int(length(ga))]
    part[ga] <- 1L + findInterval(cumsum(size[ga]) / sum(size[ga]), cuts, left.open = TRUE)
  }
  part[g]
}

check_split_args <- function(n, split_frac, groups) {
  if (!is.numeric(split_frac) || length(split_frac) != 1 || !(split_frac > 0 && split_frac < 1))
    stop("split_frac must be a number in (0, 1)")
  if (!is.null(groups) && (length(groups) != n || anyNA(groups)))
    stop("groups must have one non-missing label per subject")
}
