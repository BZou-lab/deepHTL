#' @title Cross-Fitted Permutation Test for Treatment Heterogeneity (Nested Half-Half)
#' 
#' @description Performs a nested half-half cross-fitted permutation test to evaluate 
#' treatment effect heterogeneity. Stage 1 uses a strict 2-fold split to isolate nuisance 
#' predictions. Stage 2 performs a nested k-fold permutation test within each quarantined half.
#' 
#' @param object An object of class `Trt` containing the covariates, outcome, and treatment assignment.
#' @param k_folds Integer. The number of nested folds for Stage 2 cross-fitting. Default is 5.
#' @param B Integer. The number of permutation shuffles to perform per fold. Default is 1000.
#' @param en_dnn_ctrl A list of control parameters for the `ensemble_dnnet` function.
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
        l1.reg = 1e-4,
        plot = FALSE,
        learning.rate.adaptive = "adam",
        early.stop.det = 20
      )
    )
  }
  
  fold_master <- sample(rep(1:2, length.out = n))
  e_hat <- mu_hat <- rep(NA_real_, n)
  
  for (half in 1:2) {
    tr <- which(fold_master != half) 
    te <- which(fold_master == half) 
    
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
  
  beta1 <- tryCatch({
    mod <- stats::lm(as.numeric(y) ~ as.numeric(z_num) + as.numeric(e_hat))
    b1 <- stats::coef(mod)
    if (length(b1) == 0 || any(is.na(b1))) 0 else as.numeric(b1)
  }, error = function(e) 0)
  
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
  
  Ystar <- y - tau0 * z_num
  ys0_hat <- rep(NA_real_, n)
  
  for (half in 1:2) {
    tr <- which(fold_master != half)
    te <- which(fold_master == half)
    
    ys0_obj <- importDnnet(x = X[tr, , drop = FALSE], y = Ystar[tr])
    ys0_mod <- do.call(ensemble_dnnet, c(list(object = ys0_obj), en_dnn_ctrl))
    ys0_hat[te] <- as.numeric(deepTL::predict(ys0_mod, X[te, , drop = FALSE]))
  }

  Ytilde_u <- (y - mu_hat) / (z_num - e_hat)
  Ytilde_r <- (y - tau0 * z_num - ys0_hat) / (z_num - e_hat)
  
  obs_mse_u_total <- 0
  obs_mse_r_total <- 0
  perm_mse_u_total <- rep(0, B)
  perm_mse_r_total <- rep(0, B)
  
  for (half in 1:2) {
    idx_half <- which(fold_master == half)
    n_half <- length(idx_half)
    z_fac_half <- z_fac[idx_half]
    idx_list_half <- lapply(levels(z_fac_half), function(lev) which(z_fac_half == lev))
    inner_folds <- integer(n_half)
    
    for (ix in idx_list_half) {
      if(length(ix) < k_folds) {
        inner_folds[ix] <- sample(1:k_folds, length(ix), replace = TRUE)
      } else {
        kseq <- rep(1:k_folds, length.out = length(ix))
        inner_folds[ix] <- sample(kseq, length(ix))
      }
    }
    
    for (k in 1:k_folds) {
      tr_inner_idx <- idx_half[inner_folds != k]
      te_inner_idx <- idx_half[inner_folds == k]
      
      obj_u_tr <- importDnnet(x = X[tr_inner_idx, , drop = FALSE], y = Ytilde_u[tr_inner_idx], w = w[tr_inner_idx])
      mod_u_tr <- do.call(ensemble_dnnet, c(list(object = obj_u_tr), en_dnn_ctrl))
      pred_u_te <- as.numeric(deepTL::predict(mod_u_tr, X[te_inner_idx, , drop = FALSE]))
      
      obj_r_tr <- importDnnet(x = X[tr_inner_idx, , drop = FALSE], y = Ytilde_r[tr_inner_idx], w = w[tr_inner_idx])
      mod_r_tr <- do.call(ensemble_dnnet, c(list(object = obj_r_tr), en_dnn_ctrl))
      pred_r_te <- as.numeric(deepTL::predict(mod_r_tr, X[te_inner_idx, , drop = FALSE]))
      
      obs_mse_u_total <- obs_mse_u_total + sum(w[te_inner_idx] * (Ytilde_u[te_inner_idx] - pred_u_te)^2)
      obs_mse_r_total <- obs_mse_r_total + sum(w[te_inner_idx] * (Ytilde_r[te_inner_idx] - pred_r_te)^2)
      
      z_te <- z_num[te_inner_idx]
      idx_1_te <- which(z_te == 1)
      idx_0_te <- which(z_te == 0)
      
      for (b in 1:B) {
        shuffled_idx <- numeric(length(te_inner_idx))
        if(length(idx_1_te) > 0) shuffled_idx[idx_1_te] <- sample(idx_1_te)
        if(length(idx_0_te) > 0) shuffled_idx[idx_0_te] <- sample(idx_0_te)
        
        pred_u_te_perm <- pred_u_te[shuffled_idx]
        pred_r_te_perm <- pred_r_te[shuffled_idx]
        
        perm_mse_u_total[b] <- perm_mse_u_total[b] + sum(w[te_inner_idx] * (Ytilde_u[te_inner_idx] - pred_u_te_perm)^2)
        perm_mse_r_total[b] <- perm_mse_r_total[b] + sum(w[te_inner_idx] * (Ytilde_r[te_inner_idx] - pred_r_te_perm)^2)
      }
    }
  }
  
  obs_mse_u <- obs_mse_u_total / n
  obs_mse_r <- obs_mse_r_total / n
  
  perm_mse_u <- perm_mse_u_total / n
  perm_mse_r <- perm_mse_r_total / n
  
  p_val_u <- (sum(perm_mse_u <= obs_mse_u) + 1) / (B + 1)
  p_val_r <- (sum(perm_mse_r <= obs_mse_r) + 1) / (B + 1)
  
  list(
    unrevised = list(obs_mse = obs_mse_u, p_value = p_val_u),
    revised = list(obs_mse = obs_mse_r, p_value = p_val_r)
  )
}