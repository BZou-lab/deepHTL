library(deepTL); library(caret); library(glmnet)

# ---- weight_dnn ----
weight_dnn <- function(object, en_dnn_ctrl = NULL) {
  z_fac <- if (is.factor(object@z)) object@z else factor(ifelse(object@z == 1, "A", "B"), levels = c("A", "B"))
  z_num <- if (is.numeric(object@z)) object@z else as.numeric(z_fac == "A")
  X <- object@x
  y <- object@y
  n <- nrow(X)
  
  if (is.null(en_dnn_ctrl))
    en_dnn_ctrl <- list(n.ensemble = 80, verbose = FALSE,
                        esCtrl = list(
                          n.hidden = c(64, 32), n.batch = 128, n.epoch = 200,
                          norm.x = TRUE, norm.y = TRUE, activate = "relu",
                          accel = "rcpp", l1.reg = 1e-3, plot = FALSE,
                          learning.rate.adaptive = "adam", early.stop.det = 25
                        )
    )
  
  z_obj <- importDnnet(x = X, y = z_fac)
  z_mod <- do.call("ensemble_dnnet", c(list(object = z_obj), en_dnn_ctrl))
  pz <- predict(z_mod, X)
  e_hat <- if (is.null(dim(pz))) as.numeric(pz) else as.numeric(pz[, "A"])
  
  y_obj <- importDnnet(x = X, y = y)
  y_mod <- do.call("ensemble_dnnet", c(list(object = y_obj), en_dnn_ctrl))
  mu_hat <- as.numeric(predict(y_mod, X))
  
  w <- (z_num - e_hat)^2
  
  semi_dat <- importDnnet(
    x = X,
    y = (y - mu_hat) / (z_num - e_hat),
    w = w
  )
  tau_mod_u <- do.call("ensemble_dnnet", c(list(object = semi_dat), en_dnn_ctrl))
  
  tau0 <- sum((y - mu_hat) * (z_num - e_hat)) / sum(w)
  beta1 <- coef(lm(y ~ z_num + e_hat))[2]
  
  lambdas <- seq(0, 1, length.out = 21)
  best <- list(score = Inf, lam = NA)
  fold <- sample(rep(1:2, length.out = length(z_num)))
  
  for (lam in lambdas) {
    c_lam <- lam * tau0 + (1 - lam) * beta1
    score <- 0
    for (k in 1:2) {
      idx_tr <- fold != k; idx_te <- fold == k
      ystar_tr <- y[idx_tr] - c_lam * z_num[idx_tr]
      xi_mod <- glmnet::cv.glmnet(as.matrix(X[idx_tr, , drop = FALSE]), ystar_tr, alpha = 1)
      xi_hat_te <- as.numeric(predict(xi_mod, as.matrix(X[idx_te, , drop = FALSE]), s = "lambda.min"))
      zr_te <- z_num[idx_te] - e_hat[idx_te]
      lab_te <- (y[idx_te] - c_lam * z_num[idx_te] - xi_hat_te) / zr_te
      score <- score + var(lab_te, na.rm = TRUE)
    }
    if (score < best$score) best <- list(score = score, lam = lam)
  }
  tau0 <- best$lam * tau0 + (1 - best$lam) * beta1
  
  ys0_obj <- importDnnet(x = X, y = y - tau0 * z_num)
  ys0_mod <- do.call("ensemble_dnnet", c(list(object = ys0_obj), en_dnn_ctrl))
  ys0_hat <- as.numeric(predict(ys0_mod, X))
  
  semi_dat_r <- importDnnet(
    x = X,
    y = (y - tau0 * z_num - ys0_hat) / (z_num - e_hat),
    w = w
  )
  tau_mod_r <- do.call("ensemble_dnnet", c(list(object = semi_dat_r), en_dnn_ctrl))
  
  mod <- list(
    e_mod = z_mod, mu_mod = y_mod,
    unrevised = list(tau_mod = tau_mod_u),
    revised = list(tau_mod = tau_mod_r, ys0_mod = ys0_mod, tau0 = tau0)
  )
  class(mod) <- "weight_dnn"
  mod
}

predict.weight_dnn <- function(fit, newx, which = c("both","unrevised","revised")) { 
  which <- match.arg(which) 
  tu <- as.numeric(predict(fit$unrevised$tau_mod, newx))
  tr <- as.numeric(predict(fit$revised$tau_mod, newx)) + fit$revised$tau0 
  if (which == "unrevised") return(tu) 
  if (which == "revised") return(tr) 
  data.frame(unrevised = tu, revised = tr)
}
