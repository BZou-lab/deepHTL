#' @title Cross-Fitted Permutation Test for Treatment Heterogeneity
#' 
#' @description Performs a cross-fitted permutation test to evaluate treatment effect 
#' heterogeneity. The test utilizes deep neural networks to estimate nuisance parameters 
#' and constructs residuals under the null hypothesis of a constant treatment effect.
#' 
#' @param object An object of class `Trt` containing the covariates, outcome, and treatment assignment.
#' @param k_folds Integer. The number of folds for cross-fitting. Default is 5.
#' @param B Integer. The number of permutation shuffles to perform per fold. Default is 1000.
#' @param en_dnn_ctrl A list of control parameters for the `ensemble_dnnet` function. If `NULL`, default parameters are used.
#' 
#' @return A list containing two sub-lists (`unrevised` and `revised`), each providing the 
#' observed mean squared error (`obs_mse`) and the resulting permutation p-value (`p_value`).
#' 
#' @importFrom stats coef lm predict var
#' @importFrom glmnet cv.glmnet
#' @export
cv_perm_test <- function(object, k_folds = 5, B = 1000, en_dnn_ctrl = NULL) {
  z_fac <- if (is.factor(object@z)) object@z else factor(ifelse(object@z == 1, "A", "B"), levels = c("A", "B"))
  z_num <- if (is.numeric(object@z)) object@z else as.numeric(z_fac == "A")
  
  X <- object@x
  y <- object@y
  n <- nrow(X)
  
  if (is.null(en_dnn_ctrl)) {
    en_dnn_ctrl <- list(
      n.ensemble = 30, verbose = FALSE,
      esCtrl = list(
        n.hidden = c(128, 64, 32),
        n.batch = 256,
        n.epoch = 120,
        norm.x = TRUE, norm.y = TRUE,
        activate = "relu", accel = "rcpp",
        l1.reg = 1e-3,
        plot = FALSE,
        learning.rate.adaptive = "adam",
        early.stop.det = 20
      )
    )
  }
  
  K <- k_folds
  idx_list <- lapply(levels(z_fac), function(lev) which(z_fac == lev))
  folds <- integer(length(z_fac))
  for (ix in idx_list) {
    if(length(ix) < K) {
      folds[ix] <- sample(1:K, length(ix), replace = TRUE)
    } else {
      kseq <- rep(1:K, length.out = length(ix))
      folds[ix] <- sample(kseq, length(ix))
    }
  }
  
  e_hat <- mu_hat <- rep(NA_real_, n)
  
  for (k in 1:K) {
    tr <- which(folds != k)
    te <- which(folds == k)
    
    z_obj <- importDnnet(x = X[tr, , drop = FALSE], y = z_fac[tr])
    z_mod <- do.call(ensemble_dnnet, c(list(object = z_obj), en_dnn_ctrl))
    pk <- deepTL::predict(z_mod, X[te, , drop = FALSE])
    e_hat[te] <- if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])
    
    y_obj <- importDnnet(x = X[tr, , drop = FALSE], y = y[tr])
    y_mod <- do.call(ensemble_dnnet, c(list(object = y_obj), en_dnn_ctrl))
    mu_hat[te] <- as.numeric(deepTL::predict(y_mod, X[te, , drop = FALSE]))
  }
  
  e_hat <- pmin(pmax(e_hat, 1e-2), 1 - 1e-2)
  w <- (z_num - e_hat)^2
  
  tau0 <- sum((y - mu_hat) * (z_num - e_hat)) / sum(w)
  beta1 <- tryCatch(stats::coef(stats::lm(y ~ z_num + e_hat))[2], error = function(e) 0)
  if(is.na(beta1)) beta1 <- 0
  
  lambdas <- seq(0, 1, length.out = 21)
  best <- list(score = Inf, lam = NA)
  fold_internal <- sample(rep(1:2, length.out = length(z_num)))
  
  for (lam in lambdas) {
    c_lam <- lam * tau0 + (1 - lam) * beta1
    score <- 0
    for (k in 1:2) {
      idx_tr <- fold_internal != k
      idx_te <- fold_internal == k
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
  tau0 <- best$lam * tau0 + (1 - best$lam) * beta1
  
  ys0_hat <- rep(NA_real_, n)
  Ystar <- y - tau0 * z_num
  
  for (k in 1:K) {
    tr <- which(folds != k)
    te <- which(folds == k)
    
    ys0_obj <- importDnnet(x = X[tr, , drop = FALSE], y = Ystar[tr])
    ys0_mod <- do.call(ensemble_dnnet, c(list(object = ys0_obj), en_dnn_ctrl))
    ys0_hat[te] <- as.numeric(deepTL::predict(ys0_mod, X[te, , drop = FALSE]))
  }

  Ytilde_u <- (y - mu_hat) / (z_num - e_hat)
  Ytilde_r <- (y - tau0 * z_num - ys0_hat) / (z_num - e_hat)
  
  obs_mse_u_folds <- numeric(K)
  obs_mse_r_folds <- numeric(K)
  perm_mse_u_folds <- matrix(0, nrow = K, ncol = B)
  perm_mse_r_folds <- matrix(0, nrow = K, ncol = B)
  
  for (k in 1:K) {
    tr <- which(folds != k)
    te <- which(folds == k)
    
    obj_u_tr <- importDnnet(x = X[tr, , drop = FALSE], y = Ytilde_u[tr], w = w[tr])
    mod_u_tr <- do.call(ensemble_dnnet, c(list(object = obj_u_tr), en_dnn_ctrl))
    pred_u_te <- as.numeric(deepTL::predict(mod_u_tr, X[te, , drop = FALSE]))
    
    obj_r_tr <- importDnnet(x = X[tr, , drop = FALSE], y = Ytilde_r[tr], w = w[tr])
    mod_r_tr <- do.call(ensemble_dnnet, c(list(object = obj_r_tr), en_dnn_ctrl))
    pred_r_te <- as.numeric(deepTL::predict(mod_r_tr, X[te, , drop = FALSE]))
    
    obs_mse_u_folds[k] <- sum(w[te] * (Ytilde_u[te] - pred_u_te)^2)
    obs_mse_r_folds[k] <- sum(w[te] * (Ytilde_r[te] - pred_r_te)^2)
    
    idx_1_te <- which(z_num[te] == 1)
    idx_0_te <- which(z_num[te] == 0)
    n_te <- length(te)
    
    # Inner permutation loop
    for (b in 1:B) {
      shuffled_idx <- numeric(n_te)
      if(length(idx_1_te) > 0) shuffled_idx[idx_1_te] <- sample(idx_1_te)
      if(length(idx_0_te) > 0) shuffled_idx[idx_0_te] <- sample(idx_0_te)
      
      pred_u_te_perm <- pred_u_te[shuffled_idx]
      pred_r_te_perm <- pred_r_te[shuffled_idx]
      
      perm_mse_u_folds[k, b] <- sum(w[te] * (Ytilde_u[te] - pred_u_te_perm)^2)
      perm_mse_r_folds[k, b] <- sum(w[te] * (Ytilde_r[te] - pred_r_te_perm)^2)
    }
  }
  
  obs_mse_u <- sum(obs_mse_u_folds) / n
  obs_mse_r <- sum(obs_mse_r_folds) / n
  
  perm_mse_u <- colSums(perm_mse_u_folds) / n
  perm_mse_r <- colSums(perm_mse_r_folds) / n
  
  p_val_u <- (sum(perm_mse_u <= obs_mse_u) + 1) / (B + 1)
  p_val_r <- (sum(perm_mse_r <= obs_mse_r) + 1) / (B + 1)
  
  list(
    unrevised = list(obs_mse = obs_mse_u, p_value = p_val_u),
    revised = list(obs_mse = obs_mse_r, p_value = p_val_r)
  )
}