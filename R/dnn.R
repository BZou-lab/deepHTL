#' @importFrom stats predict coef lm var
#' @importFrom methods is
#' @importFrom deepTL importDnnet ensemble_dnnet importTrt
#' @import glmnet
NULL

#' Weighted DNN Estimator
#'
#' Estimates treatment effects using an ensemble of Deep Neural Networks with
#' bias correction steps.
#'
#' @param object An object containing the data (usually class \code{Trt}).
#' @param k_folds Integer. Number of folds for cross-fitting (default 5).
#' @param en_dnn_ctrl List. Control parameters for the ensemble DNN.
#'
#' @return An object of class \code{weight_dnn} containing the fitted models.
#' @export

weight_dnn <- function(object, k_folds = 5, en_dnn_ctrl = NULL) {
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
  make_stratified_folds <- function(z, K) {
    z <- as.factor(z)
    idx_list <- lapply(levels(z), function(lev) which(z == lev))
    folds <- integer(length(z))
    for (ix in idx_list) {
      if(length(ix) < K) {
         folds[ix] <- sample(1:K, length(ix), replace=TRUE)
      } else {
         kseq <- rep(1:K, length.out = length(ix))
         folds[ix] <- sample(kseq, length(ix))
      }
    }
    folds
  }
  folds <- make_stratified_folds(z_fac, K)
  
  e_hat <- mu_hat <- rep(NA_real_, n)
  z_mod_folds <- vector("list", K)
  y_mod_folds <- vector("list", K)
  
  for (k in 1:K) {
    tr <- which(folds != k); te <- which(folds == k)
    
    z_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = z_fac[tr])
    z_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = z_obj), en_dnn_ctrl))
    pk <- deepTL::predict(z_mod, X[te, , drop = FALSE])
    e_hat[te] <- if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])
    z_mod_folds[[k]] <- z_mod
    
    y_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = y[tr])
    y_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = y_obj), en_dnn_ctrl))
    mu_hat[te] <- as.numeric(deepTL::predict(y_mod, X[te, , drop = FALSE]))
    y_mod_folds[[k]] <- y_mod
  }
  
  e_hat <- pmin(pmax(e_hat, 1e-2), 1 - 1e-2)
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
      xi_mod <- tryCatch(glmnet::cv.glmnet(as.matrix(X[idx_tr, , drop = FALSE]), ystar_tr, alpha = 1), error=function(e) NULL)
      
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
    tr <- which(folds != k); te <- which(folds == k)
    
    ys0_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = Ystar[tr])
    ys0_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = ys0_obj), en_dnn_ctrl))
    ys0_hat[te] <- as.numeric(deepTL::predict(ys0_mod, X[te, , drop = FALSE]))
  }

  Ytilde_u <- (y - mu_hat) / (z_num - e_hat)
  Ytilde_r <- (y - tau0 * z_num - ys0_hat) / (z_num - e_hat)
  
  tau_mod_u_folds <- vector("list", K)
  tau_mod_r_folds <- vector("list", K)
  
  for (k in 1:K) {
    tr <- which(folds != k)
    
    obj_u_tr <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = Ytilde_u[tr], w = w[tr])
    tau_mod_u_folds[[k]] <- do.call(deepTL::ensemble_dnnet, c(list(object = obj_u_tr), en_dnn_ctrl))
    
    obj_r_tr <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = Ytilde_r[tr], w = w[tr])
    tau_mod_r_folds[[k]] <- do.call(deepTL::ensemble_dnnet, c(list(object = obj_r_tr), en_dnn_ctrl))
  }
  
  mod <- list(
    folds = folds,
    nuisance = list(X = X, y = y, z_num = z_num, e_hat = e_hat, mu_hat = mu_hat, ys0_hat = ys0_hat, w = w),
    unrevised = list(tau_mod_folds = tau_mod_u_folds),
    revised = list(tau_mod_folds = tau_mod_r_folds, tau0 = tau0)
  )
  class(mod) <- "weight_dnn"
  mod
}

#' Predictions for weight_dnn objects
#'
#' @param object A fitted weight_dnn object.
#' @param newx Matrix of new covariates.
#' @param which Character string. Which estimator to predict: "both", "unrevised", or "revised".
#' @param ... Additional arguments (unused)
#' 
#' @return A numeric vector or data frame of predictions.
#' @export
predict.weight_dnn <- function(object, newx, which = c("both", "unrevised", "revised"), ...) {
  which <- match.arg(which)
  
  K <- length(object$unrevised$tau_mod_folds)
  
  pred_u_mat <- sapply(1:K, function(k) {
    as.numeric(deepTL::predict(object$unrevised$tau_mod_folds[[k]], newx))
  })
  tu <- rowMeans(pred_u_mat)
  
  pred_r_mat <- sapply(1:K, function(k) {
    as.numeric(deepTL::predict(object$revised$tau_mod_folds[[k]], newx))
  })
  tr <- rowMeans(pred_r_mat) + object$revised$tau0
  
  if (which == "unrevised") return(tu)
  if (which == "revised") return(tr)
  data.frame(unrevised = tu, revised = tr)
}