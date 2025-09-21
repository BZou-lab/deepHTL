library(rlearner); library(xgboost)

# ------- weight_xgboost -------

weight_xgboost <- function(object, num_search_rounds = 10, nrounds_max = 1000) {
  x <- as.matrix(object@x)
  y <- object@y
  z <- object@z
  
  y_fit <- cvboost(x, y,
                   objective = "reg:squarederror",
                   num_search_rounds = num_search_rounds, 
                   ntrees_max = nrounds_max)
  mu_hat <- as.numeric(predict(y_fit))
  
  z_fit <- cvboost(
    x, z,
    objective = "binary:logistic",
    num_search_rounds = num_search_rounds,
    ntrees_max = nrounds_max
  )
  e_hat <- as.numeric(predict(z_fit))
  
  pseudo_u <- (y - mu_hat) / (z - e_hat)
  w <- (z - e_hat)^2
  tau_fit_u <- cvboost(
    x, pseudo_u, weights = w,
    objective = "reg:squarederror",
    num_search_rounds = num_search_rounds,
    ntrees_max = nrounds_max
  )
  
  tau0 <- sum((y - mu_hat) * (z - e_hat)) / sum((z - e_hat)^2)
  ys0 <- y - tau0 * z
  ys0_fit <- cvboost(
    x, ys0,
    objective = "reg:squarederror",
    num_search_rounds = num_search_rounds,
    ntrees_max = nrounds_max
  )
  ys0_pred <- as.numeric(predict(ys0_fit))
  
  pseudo_r <- (y - tau0 * z - ys0_pred) / (z - e_hat)
  tau_fit_r <- cvboost(
    x, pseudo_r, weights = w,
    objective = "reg:squarederror",
    num_search_rounds = num_search_rounds,
    ntrees_max = nrounds_max
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

predict.weight_xgboost <- function(fit, newx, which = c("both","unrevised","revised")) {
  which <- match.arg(which)
  x_new <- as.matrix(newx)
  
  pred_unrev <- as.numeric(predict(fit$unrevised$tau_fit, x_new))
  pred_rev <- as.numeric(predict(fit$revised$tau_fit, x_new)) + fit$revised$tau0
  
  if (which == "unrevised") return(pred_unrev)
  if (which == "revised") return(pred_rev)
  data.frame(unrevised = pred_unrev, revised = pred_rev)
}
