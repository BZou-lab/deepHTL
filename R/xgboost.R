#' @importFrom stats predict coef lm var
#' @importFrom methods is
#' @import xgboost
#' @import glmnet
#' @import rlearner
NULL

#' Weighted XGBoost Estimator
#'
#' Estimates treatment effects using Gradient Boosting (XGBoost) with bias correction steps
#' (Robinson transformation + one-step correction).
#'
#' @param object An object containing the data (usually class \code{Trt}).
#'               Must contain slots \code{@x}, \code{@y}, and \code{@z}.
#' @param k Integer. Number of folds for cross-validation (default 3).
#'
#' @return An object of class \code{weight_xgboost}.
#' @export
weight_xgboost <- function(object, k = 3) {
  x <- as.matrix(object@x)
  y <- object@y
  z <- if (is.factor(object@z)) {
    as.numeric(object@z == levels(object@z)[1])
  } else {
    as.numeric(object@z)
  }
  
  y_fit <- rlearner::cvboost(
    x, y, k_folds = k, 
    objective = "reg:squarederror"
  )
  mu_hat <- as.numeric(stats::predict(y_fit))
  
  z_fit <- rlearner::cvboost(
    x, z, k_folds = k, 
    objective = "binary:logistic"
  )
  e_hat <- as.numeric(stats::predict(z_fit))
  
  w <- (z - e_hat)^2
  pseudo_u <- (y - mu_hat) / (z - e_hat) 
  
  tau_fit_u <- rlearner::cvboost(
    x, pseudo_u,
    weights = w, k_folds = k,
    objective = "reg:squarederror"
  )
  
  tau0 <- sum((y - mu_hat) * (z - e_hat)) / sum(w)
  beta1 <- stats::coef(stats::lm(y ~ z + e_hat))[2]
  
  lambdas <- seq(0, 1, length.out = 21)
  best <- list(score = Inf, lam = NA)
  fold <- sample(rep(1:2, length.out = length(z)))
  
  for (lam in lambdas) {
    c_lam <- lam * tau0 + (1 - lam) * beta1
    score <- 0
    for (kk in 1:2) {
      idx_tr <- fold != kk; idx_te <- fold == kk
      ystar_tr <- y[idx_tr] - c_lam * z[idx_tr]
      
      xi_mod <- glmnet::cv.glmnet(as.matrix(x[idx_tr, , drop = FALSE]), ystar_tr, alpha = 1)
      xi_hat_te <- as.numeric(stats::predict(xi_mod, as.matrix(x[idx_te, , drop = FALSE]), s = "lambda.min"))
      
      zr_te <- z[idx_te] - e_hat[idx_te]
      lab_te <- (y[idx_te] - c_lam * z[idx_te] - xi_hat_te) / zr_te
      score  <- score + stats::var(lab_te, na.rm = TRUE)
    }
    if (score < best$score) best <- list(score = score, lam = lam)
  }
  
  tau0 <- best$lam * tau0 + (1 - best$lam) * beta1
  
  ys0 <- y - tau0 * z
  ys0_fit <- rlearner::cvboost(
    x, ys0, k_folds = k, 
    objective = "reg:squarederror"
  )
  ys0_pred <- as.numeric(stats::predict(ys0_fit))
  
  pseudo_r <- (y - tau0 * z - ys0_pred) / (z - e_hat)
  tau_fit_r <- rlearner::cvboost(
    x, pseudo_r,
    weights = w,
    k_folds = k,
    objective = "reg:squarederror"
  )
  
  model <- list(
    y_fit = y_fit,
    z_fit = z_fit,
    mu_hat = mu_hat,
    e_hat = e_hat,
    unrevised = list(tau_fit = tau_fit_u),
    revised = list(tau_fit = tau_fit_r, ys0_fit = ys0_fit, tau0 = tau0)
  )
  class(model) <- "weight_xgboost"
  return(model)
}

#' Predictions for weight_xgboost objects
#'
#' @param object A fitted weight_xgboost object.
#' @param newx Matrix of new covariates.
#' @param which Character string. Which estimator to predict: "both", "unrevised", or "revised".
#' @param ... Additional arguments (unused).
#'
#' @return A numeric vector or data frame.
#' @export
predict.weight_xgboost <- function(object, newx, which = c("both", "unrevised", "revised"), ...) {
  which <- match.arg(which)
  x_new <- as.matrix(newx)

  pred_unrev <- as.numeric(stats::predict(object$unrevised$tau_fit, x_new))
  pred_rev <- as.numeric(stats::predict(object$revised$tau_fit, x_new)) + object$revised$tau0
  
  if (which == "unrevised") return(pred_unrev)
  if (which == "revised") return(pred_rev)
  data.frame(unrevised = pred_unrev, revised = pred_rev)
}