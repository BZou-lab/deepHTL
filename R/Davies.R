#' @importFrom stats dist median predict coef lm var rnorm rbinom plogis
#' @importFrom methods is
#' @importFrom CompQuadForm davies liu
#' @importFrom glmnet cv.glmnet
#' @importFrom deepTL importDnnet ensemble_dnnet importTrt
NULL

#' Davies Test for Heterogeneous Treatment Effects
#'
#' Performs a hypothesis test for the presence of Heterogeneous Treatment Effects (HTE)
#' using the Davies method on standardized residuals from a bias-corrected Deep Neural Network model.
#'
#' @param object An object containing the data (usually class \code{Trt}).
#'               Must contain slots \code{@x}, \code{@y}, and \code{@z}.
#' @param ctrl Optional list. Control parameters for the deepTL estimation.
#'             If NULL, defaults are used.
#' @param B Integer. Number of permutations for the permutation test (default 20000).
#'
#' @return A list containing:
#' \item{Q}{The observed test statistic.}
#' \item{p_davies}{P-value calculated using the Davies method (mixture of chi-squares).}
#' \item{p_perm}{P-value calculated using permutation.}
#' @export
davies_test <- function(object, ctrl = NULL, B = 20000) {
  if (!is.factor(object@z)) {
    object@z <- factor(ifelse(object@z == 1, "A", "B"), levels = c("A", "B"))
  }
  
  x <- object@x
  y <- object@y
  z_num <- as.numeric(object@z == "A")
  n <- nrow(x)
  
  if (is.null(ctrl)) {
    ctrl <- list(
      n.ensemble = 100, verbose = FALSE,
      esCtrl = list(
        n.hidden = c(10, 5) * 2, n.batch = 100, n.epoch = 200,
        norm.x = TRUE, norm.y = TRUE,
        activate = "relu", accel = "rcpp",
        l1.reg = 1e-4, plot = FALSE,
        learning.rate.adaptive = "adam",
        early.stop.det = 100
      )
    )
  }
  
  z_obj <- deepTL::importDnnet(x = x, y = object@z)
  z_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = z_obj), ctrl))
  e_hat <- deepTL::predict(z_mod, x)[, "A"]
  e_hat <- pmin(pmax(e_hat, 1e-2), 1 - 1e-2) 
  
  y_obj <- deepTL::importDnnet(x = x, y = y)
  y_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = y_obj), ctrl))
  mu_hat <- as.numeric(deepTL::predict(y_mod, x))
  
  w <- (z_num - e_hat)^2
  tau0 <- sum((y - mu_hat) * (z_num - e_hat)) / sum(w)

  beta1 <- stats::coef(stats::lm(y ~ z_num + e_hat))[2]
  lambdas <- seq(0, 1, length.out = 21)
  best <- list(score = Inf, lam = NA)
  fold <- sample(rep(1:2, length.out = length(z_num)))
  
  for (lam in lambdas) {
    c_lam <- lam * tau0 + (1 - lam) * beta1
    score <- 0
    for (k in 1:2) {
      idx_tr <- fold != k; idx_te <- fold == k
      ystar_tr <- y[idx_tr] - c_lam * z_num[idx_tr]
      
      xi_mod <- glmnet::cv.glmnet(as.matrix(x[idx_tr, , drop = FALSE]), ystar_tr, alpha = 1)
      xi_hat_te <- as.numeric(stats::predict(xi_mod, as.matrix(x[idx_te, , drop = FALSE]), s = "lambda.min"))
      
      zr_te <- z_num[idx_te] - e_hat[idx_te]
      lab_te <- (y[idx_te] - c_lam * z_num[idx_te] - xi_hat_te) / zr_te
      score <- score + stats::var(lab_te, na.rm = TRUE)
    }
    if (score < best$score) best <- list(score = score, lam = lam)
  }
  
  tau0 <- best$lam * tau0 + (1 - best$lam) * beta1
  
  Ystar <- y - tau0 * z_num
  ystar_obj <- deepTL::importDnnet(x = x, y = Ystar)
  ystar_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = ystar_obj), ctrl))
  mu_star_hat <- as.numeric(deepTL::predict(ystar_mod, x))
  
  tildeYstar <- (Ystar - mu_star_hat) / (z_num - e_hat)
  tau_star <- sum(w * tildeYstar) / sum(w)
  tau_rev <- tau0 + tau_star
  
  r_rev <- tildeYstar - tau_star
  sig2_rev <- mean(w * r_rev^2)
  v_rev <- sig2_rev / w
  std_r_rev <- r_rev / sqrt(v_rev)
  
  P <- diag(n) - tcrossprod(sqrt(w)) / sum(w)
  Dmat <- as.matrix(stats::dist(x))
  band <- stats::median(Dmat[upper.tri(Dmat, diag = FALSE)])
  if (band == 0) band <- 1 
  K <- exp(-(Dmat^2) / (2 * band^2))
  Kp <- P %*% K %*% P
  Kp <- 0.5 * (Kp + t(Kp))
  eig <- eigen(Kp, symmetric = TRUE, only.values = TRUE)$values
  eig <- eig[eig > 1e-8]
  s <- std_r_rev
  Q_obs <- as.numeric(t(s) %*% Kp %*% s)
  
  # Davies P-value
  p_davies <- if (length(eig)) {
    p <- tryCatch(CompQuadForm::davies(Q_obs, lambda = eig)$Qq, error = function(e) NA_real_)
    if (!is.finite(p) || p < 0) {
      tryCatch(CompQuadForm::liu(Q_obs, lambda = eig), error = function(e2) NA_real_)
    } else {
      p
    }
  } else {
    NA_real_
  }
  
  # Permutation P-value
  if (B > 0) {
    Q_perm <- replicate(B, {
      s_perm <- sample(s, replace = FALSE)
      as.numeric(t(s_perm) %*% Kp %*% s_perm)
    })
    p_perm <- (sum(Q_perm >= Q_obs) + 1) / (B + 1)
  } else {
    p_perm <- NA_real_
  }
  
  list(
    Q = Q_obs,
    p_davies = p_davies,
    p_perm = p_perm,
    tau_hat = tau_rev
  )
}
