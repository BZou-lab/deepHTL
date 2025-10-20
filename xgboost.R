library(rlearner); library(xgboost); library(glmnet)

# ---- weight_xgboost ----

weight_xgboost <- function(object, k = 3) {
  x <- as.matrix(object@x)
  y <- object@y
  z <- if (is.factor(object@z)) as.numeric(object@z == levels(object@z)[1]) else as.numeric(object@z)
  
  y_fit <- cvboost(
    x, y, k_folds = k, 
    objective = "reg:squarederror"
  )
  
  mu_hat <- as.numeric(predict(y_fit))
  
  z_fit <- cvboost(
    x, z, k_folds = k, 
    objective = "binary:logistic"
  )
  e_hat <- as.numeric(predict(z_fit))
  
  w <- (z - e_hat)^2
  pseudo_u <- (y - mu_hat) /(z - e_hat) 
  tau_fit_u <- cvboost(
    x, pseudo_u,
    weights = w, k_folds = k,
    objective = "reg:squarederror"
  )
  
  tau0 <- sum((y - mu_hat) * (z - e_hat)) / sum(w)
  
  beta1 <- coef(lm(y ~ z + e_hat))[2]
  
  lambdas <- seq(0, 1, length.out = 21)
  best <- list(score = Inf, lam = NA)
  fold <- sample(rep(1:2, length.out=length(z)))
  
  for (lam in lambdas) {
    c_lam <- lam * tau0 + (1 - lam) * beta1
    score <- 0
    for (k in 1:2) {
      idx_tr <- fold != k; idx_te <- fold == k
      ystar_tr <- y[idx_tr] - c_lam * z[idx_tr]
      xi_mod <- glmnet::cv.glmnet(as.matrix(x[idx_tr,]), ystar_tr, alpha=1)
      xi_hat_te<- as.numeric(predict(xi_mod, as.matrix(x[idx_te,]), s="lambda.min"))
      zr_te <- z[idx_te] - e_hat[idx_te]
      lab_te <- (y[idx_te] - c_lam * z[idx_te] - xi_hat_te) / zr_te
      score  <- score + var(lab_te, na.rm=TRUE)
    }
    if (score < best$score) best <- list(score = score, lam = lam)
  }
  tau0 <- best$lam * tau0 + (1 - best$lam) * beta1
  
  ys0 <- y - tau0 * z
  ys0_fit <- cvboost(
    x, ys0, k_folds = k, 
    objective = "reg:squarederror"
  )
  ys0_pred <- as.numeric(predict(ys0_fit))
  
  pseudo_r <- (y - tau0 * z - ys0_pred) / (z - e_hat)
  tau_fit_r <- cvboost(
    x, pseudo_r,
    weights = w,
    k_folds = 3, 
    objective = "reg:squarederror",
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

# ------- predict method -------
predict.weight_xgboost <- function(fit, newx, which = c("both","unrevised","revised")) {
  which <- match.arg(which)
  x_new <- as.matrix(newx)
  
  pred_unrev <- as.numeric(predict(fit$unrevised$tau_fit, x_new))
  pred_rev <- as.numeric(predict(fit$revised$tau_fit, x_new)) + fit$revised$tau0
  
  if (which == "unrevised") return(pred_unrev)
  if (which == "revised") return(pred_rev)
  data.frame(unrevised = pred_unrev, revised = pred_rev)
}
