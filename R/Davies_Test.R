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
#'                Must contain slots \code{@x}, \code{@y}, and \code{@z}.
#' @param ctrl Optional list. Control parameters for the deepTL estimation.
#'             If NULL, defaults are used.
#' @param k_folds Integer. Number of folds for cross-fitting nuisance parameters (default 5).
#'
#' @return A list containing:
#' \item{Q}{The observed test statistic.}
#' \item{p_davies}{P-value calculated using the Davies method (mixture of chi-squares).}
#' \item{tau_hat}{The estimated average treatment effect.}
#' @export
davies_test <- function(object, ctrl = NULL, k_folds = 5) {
  if (!is.factor(object@z)) {
    object@z <- factor(ifelse(object@z == 1, "A", "B"), levels = c("A", "B"))
  }
  
  x <- object@x
  y <- object@y
  z_fac <- object@z
  z_num <- as.numeric(z_fac == "A")
  n <- nrow(x)
  
  if (is.null(ctrl)) {
    ctrl <- list(
      n.ensemble = 30, verbose = FALSE,
      esCtrl = list(
        n.hidden = c(128, 64, 32), n.batch = 256, n.epoch = 120,
        norm.x = TRUE, norm.y = TRUE,
        activate = "relu", accel = "rcpp",
        l1.reg = 1e-3, plot = FALSE,
        learning.rate.adaptive = "adam",
        early.stop.det = 100
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
  
  for (k in 1:K) {
    tr <- which(folds != k); te <- which(folds == k)
    
    z_obj <- deepTL::importDnnet(x = x[tr, , drop = FALSE], y = z_fac[tr])
    z_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = z_obj), ctrl))
    pk <- deepTL::predict(z_mod, x[te, , drop = FALSE])
    e_hat[te] <- if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])
    
    y_obj <- deepTL::importDnnet(x = x[tr, , drop = FALSE], y = y[tr])
    y_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = y_obj), ctrl))
    mu_hat[te] <- as.numeric(deepTL::predict(y_mod, x[te, , drop = FALSE]))
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
      
      xi_mod <- tryCatch(glmnet::cv.glmnet(as.matrix(x[idx_tr, , drop = FALSE]), ystar_tr, alpha = 1), error=function(e) NULL)
      
      if(!is.null(xi_mod)) {
        xi_hat_te <- as.numeric(stats::predict(xi_mod, as.matrix(x[idx_te, , drop = FALSE]), s = "lambda.min"))
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
    ys0_obj <- deepTL::importDnnet(x = x[tr, , drop = FALSE], y = Ystar[tr])
    ys0_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = ys0_obj), ctrl))
    ys0_hat[te] <- as.numeric(deepTL::predict(ys0_mod, x[te, , drop = FALSE]))
  }
  mu_star_hat <- ys0_hat
  
  tildeYstar <- (Ystar - mu_star_hat) / (z_num - e_hat)
  tau_star <- sum(w * tildeYstar) / sum(w)
  tau_rev <- tau0 + tau_star
  
  r_rev <- tildeYstar - tau_star
  sig2_rev <- mean(w * r_rev^2)
  v_rev <- sig2_rev / w
  std_r_rev <- r_rev / sqrt(v_rev)
  
  P <- diag(n) - tcrossprod(sqrt(w)) / sum(w)
  Dmat <- as.matrix(stats::dist(scale(x, center = TRUE, scale = TRUE)))
  band <- stats::median(Dmat[upper.tri(Dmat, diag = FALSE)])
  if (band == 0) band <- 1 
  
  K_mat <- exp(-(Dmat^2) / (2 * band^2))
  Kp <- P %*% K_mat %*% P
  Kp <- 0.5 * (Kp + t(Kp))
  eig <- eigen(Kp, symmetric = TRUE, only.values = TRUE)$values
  eig <- eig[eig > 1e-8]
  s <- std_r_rev
  Q_obs <- as.numeric(t(s) %*% Kp %*% s)
  
  p_davies <- if (length(eig)) {
    p <- tryCatch(CompQuadForm::davies(Q_obs, lambda = eig)$Qq, error = function(e) NA_real_)
    if (is.na(p) || !is.finite(p) || p < 0 || p > 1) {
      tryCatch(CompQuadForm::liu(Q_obs, lambda = eig), error = function(e2) NA_real_)
    } else {
      p
    }
  } else {
    NA_real_
  }
  
  list(
    Q = Q_obs,
    p_davies = p_davies,
    tau_hat = tau_rev
  )
}