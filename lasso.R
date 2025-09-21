library(glmnet)

# ------- weight_lasso -------

weight_lasso <- function(object) {
  x <- as.matrix(object@x)
  y <- object@y
  z <- if (is.factor(object@z)) as.numeric(object@z == levels(object@z)[1]) else as.numeric(object@z)
  
  e_mod <- cv.glmnet(x, z, family = "binomial", alpha = 1)
  e_hat <- as.numeric(predict(e_mod, newx = x, s = "lambda.min", type = "response"))
  
  mu_mod <- cv.glmnet(x, y, alpha = 1)
  mu_hat <- as.numeric(predict(mu_mod, newx = x, s = "lambda.min"))
  
  w <- (z - e_hat)^2
  pseudo_u <- (y - mu_hat) / (z - e_hat)
  tau_mod_u <- cv.glmnet(x, pseudo_u, weights = w, alpha = 1)

  tau0 <- sum((y - mu_hat) * (z - e_hat)) / sum((z - e_hat)^2)
  
  ys0 <- y - tau0 * z
  ys0_mod <- cv.glmnet(x, ys0, alpha = 1)
  ys0_hat <- as.numeric(predict(ys0_mod, newx = x, s = "lambda.min"))
  
  pseudo_r <- (y - tau0 * z - ys0_hat) / (z - e_hat)
  tau_mod_r <- cv.glmnet(x, pseudo_r, weights = w, alpha = 1)
  
  mod <- list(
    mu_mod = mu_mod,
    e_mod = e_mod,
    mu_hat = mu_hat,
    e_hat = e_hat,
    unrevised = list(tau_mod = tau_mod_u),
    revised = list(tau_mod = tau_mod_r, ys0_mod = ys0_mod, tau0 = tau0)
  )
  class(mod) <- "weight_lasso"
  return(mod)
}

predict.weight_lasso <- function(fit, newx, which = c("both","unrevised","revised")) {
  which <- match.arg(which)
  x <- as.matrix(newx)
  
  tau_u <- as.numeric(predict(fit$unrevised$tau_mod, newx = x, s = "lambda.min"))
  tau_r <- as.numeric(predict(fit$revised$tau_mod, newx = x, s = "lambda.min")) + fit$revised$tau0
  
  if (which == "unrevised") return(tau_u)
  if (which == "revised") return(tau_r)
  data.frame(unrevised = tau_u, revised = tau_r)
}
