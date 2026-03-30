#' @title Cross-Fitted Permutation Test for Treatment Heterogeneity (Same-Fold)
#' 
#' @description Performs a same-fold cross-fitted permutation test to evaluate 
#' treatment effect heterogeneity. Uses a single set of folds across all stages 
#' to estimate nuisance parameters, adjust the baseline, and locally permute 
#' the Stage 2 predictions to preserve weight-variance alignment.
#' 
#' @param object An object of class `Trt` containing the covariates, outcome, and treatment assignment.
#' @param k_folds Integer. The number of folds for cross-fitting. Default is 5.
#' @param B Integer. The number of permutation shuffles to perform. Default is 1000.
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
  
  folds <- sample(rep(1:k_folds, length.out = n))
  e_hat <- mu_hat <- rep(NA_real_, n)
  
  for (k in 1:k_folds) {
    tr <- folds != k
    te <- folds == k
    
    z_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = z_fac[tr])
    z_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = z_obj), en_dnn_ctrl))
    pk <- deepTL::predict(z_mod, X[te, , drop = FALSE])
    e_hat[te] <- if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])
    
    y_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = y[tr])
    y_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = y_obj), en_dnn_ctrl))
    mu_hat[te] <- as.numeric(deepTL::predict(y_mod, X[te, , drop = FALSE]))
  }
  
  e_hat <- pmin(pmax(e_hat, 5e-2), 1 - 5e-2)
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
    
    ys0_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = Ystar[tr])
    ys0_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = ys0_obj), en_dnn_ctrl))
    ys0_hat[te] <- as.numeric(deepTL::predict(ys0_mod, X[te, , drop = FALSE]))
  }

  Ytilde_u <- (y - mu_hat) / (z_num - e_hat)
  Ytilde_r <- (y - tau0_opt * z_num - ys0_hat) / (z_num - e_hat)
  
  pred_u <- numeric(n)
  pred_r <- numeric(n)
  
  for (k in 1:k_folds) {
    tr <- folds != k
    te <- folds == k
    
    obj_u_tr <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = Ytilde_u[tr], w = w[tr])
    mod_u_tr <- do.call(deepTL::ensemble_dnnet, c(list(object = obj_u_tr), en_dnn_ctrl))
    pred_u[te] <- as.numeric(deepTL::predict(mod_u_tr, X[te, , drop = FALSE]))
    
    obj_r_tr <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = Ytilde_r[tr], w = w[tr])
    mod_r_tr <- do.call(deepTL::ensemble_dnnet, c(list(object = obj_r_tr), en_dnn_ctrl))
    pred_r[te] <- as.numeric(deepTL::predict(mod_r_tr, X[te, , drop = FALSE]))
  }
  
  obs_mse_u <- sum(w * (Ytilde_u - pred_u)^2) / n
  obs_mse_r <- sum(w * (Ytilde_r - pred_r)^2) / n
  
  perm_mse_u <- numeric(B)
  perm_mse_r <- numeric(B)
  
  idx_1 <- which(z_num == 1)
  idx_0 <- which(z_num == 0)
  
  for (b in 1:B) {
    pred_u_perm <- numeric(n)
    pred_r_perm <- numeric(n)
    
    for (k in 1:k_folds) {
      te <- which(folds == k)
      
      idx_1_te <- intersect(idx_1, te)
      idx_0_te <- intersect(idx_0, te)
      
      if(length(idx_1_te) > 0) {
        shuf_1 <- sample(idx_1_te)
        pred_u_perm[idx_1_te] <- pred_u[shuf_1]
        pred_r_perm[idx_1_te] <- pred_r[shuf_1]
      }
      if(length(idx_0_te) > 0) {
        shuf_0 <- sample(idx_0_te)
        pred_u_perm[idx_0_te] <- pred_u[shuf_0]
        pred_r_perm[idx_0_te] <- pred_r[shuf_0]
      }
    }
    
    perm_mse_u[b] <- sum(w * (Ytilde_u - pred_u_perm)^2) / n
    perm_mse_r[b] <- sum(w * (Ytilde_r - pred_r_perm)^2) / n
  }
  
  p_val_u <- (sum(perm_mse_u <= obs_mse_u) + 1) / (B + 1)
  p_val_r <- (sum(perm_mse_r <= obs_mse_r) + 1) / (B + 1)
  
  list(
    unrevised = list(obs_mse = obs_mse_u, p_value = p_val_u),
    revised = list(obs_mse = obs_mse_r, p_value = p_val_r)
  )
}