#' @importFrom stats lm coef var median dist predict
#' @importFrom methods is
NULL

# ==============================================================================
# Internal Helper Functions (Not Exported)
# ==============================================================================

#' Squared Euclidean Distance
#' @keywords internal
sqdist <- function(A, B = NULL) {
  if (is.null(B)) B <- A
  aa <- rowSums(A*A)
  bb <- rowSums(B*B)
  out <- outer(aa, bb, "+") - 2 * tcrossprod(A, B)
  pmax(out, 0)
}

#' Gaussian Kernel Calculation
#' @keywords internal
gaussian_kernel <- function(X, Z = NULL, b) {
  sigma <- 1 / b
  exp(-sqdist(X, Z) / (2 * sigma^2))
}

#' Kernel Ridge Regression Fitter
#' @keywords internal
krr_fit <- function(X, y, b, lambda, weights = NULL, jitter = 1e-8) {
  n <- nrow(X)
  w <- if (is.null(weights)) rep(1, n) else as.numeric(weights)
  W2 <- sqrt(pmax(w, 0))
  K  <- gaussian_kernel(X, b = b)
  Kt <- (W2 * K) * rep(W2, each = n)
  yt <- W2 * y
  # Solve (K_tilde + lambda*I) a = y_tilde
  a  <- solve(Kt + (n * lambda + jitter) * diag(n), yt, tol = 1e-7)
  beta <- W2 * a
  list(X = X, b = b, lambda = lambda, beta = beta, fitted = as.numeric(K %*% beta))
}

#' Prediction for KRR internal model
#' @keywords internal
krr_predict <- function(model, Xnew) {
  as.numeric(gaussian_kernel(Xnew, model$X, b = model$b) %*% model$beta)
}

#' Heuristic for Bandwidth Selection
#' @keywords internal
median_bandwidth <- function(x) {
  n <- nrow(x)
  samp <- if (n > 2000) sample.int(n, 2000) else seq_len(n)
  D <- sqdist(x[samp, , drop = FALSE])
  md <- sqrt(stats::median(D[upper.tri(D, diag = FALSE)]))
  if (!is.finite(md) || md <= 0) md <- 1
  1 / md
}

#' Fast Cross-Validation for KRR
#' @keywords internal
cv_krr_fast <- function(x, y, weights = NULL, k_folds = 3,
                        b_range = NULL,
                        lambda_range = 10^(seq(-4, 2, 0.5))) {
  x <- as.matrix(x)
  n <- nrow(x)
  if (is.null(k_folds)) k_folds <- max(3, min(10, floor(n/4)))
  
  foldid <- sample(rep(seq_len(k_folds), length.out = n))
  
  if (is.null(b_range)) {
    b0 <- median_bandwidth(x)
    b_range <- b0 * c(0.5, 1, 2)
  }
  
  best <- list(mse = Inf, b = NA, lambda = NA, fit = NULL)
  
  for (b in b_range) {
    Kfull <- gaussian_kernel(x, b = b)
    fold_cache <- vector("list", k_folds)
    for (f in seq_len(k_folds)) {
      tr <- (foldid != f); te <- !tr
      Ktr <- Kfull[tr, tr, drop = FALSE]
      ytr <- y[tr]
      
      if (is.null(weights)) {
        W2 <- rep(1, sum(tr)) 
      } else {
        W2 <- sqrt(pmax(as.numeric(weights[tr]), 0))
      }
      
      Ktilde <- (W2 * Ktr) * rep(W2, each = length(W2))
      ytilde <- W2 * ytr
      
      ee <- eigen(Ktilde + 1e-10 * diag(nrow(Ktilde)), symmetric = TRUE, only.values = FALSE)
      U  <- ee$vectors
      d  <- pmax(ee$values, 0)
      
      Kte <- Kfull[te, tr, drop = FALSE]
      fold_cache[[f]] <- list(U = U, d = d, ytilde = ytilde, W2 = W2, Kte = Kte, ntr = sum(tr))
    }
  
    for (lambda in lambda_range) {
      pred <- numeric(n)
      for (f in seq_len(k_folds)) {
        te <- (foldid == f)
        fc <- fold_cache[[f]]
        inv_diag <- 1 / (fc$d + fc$ntr * lambda)
        Uy <- crossprod(fc$U, fc$ytilde)
        a  <- fc$U %*% (inv_diag * Uy)
        beta_tr <- fc$W2 * a
        pred[te] <- as.numeric(fc$Kte %*% beta_tr)
      }
      
      mse <- if (is.null(weights)) mean((y - pred)^2) else mean(as.numeric(weights) * (y - pred)^2)
      
      if (mse < best$mse) best <- list(mse = mse, b = b, lambda = lambda, fit = pred)
    }
  }
  
  final <- krr_fit(x, y, b = best$b, lambda = best$lambda, weights = weights)
  list(b = best$b, lambda = best$lambda, fit = best$fit, model = final)
}

# ==============================================================================
# Exported Functions
# ==============================================================================

#' Weighted Kernel Ridge Regression Estimator
#'
#' Estimates treatment effects using Kernel Ridge Regression (KRR) with
#' bias correction steps (Robinson transformation + one-step correction).
#'
#' @param object An object containing the data (usually class \code{Trt}).
#' @param x Optional matrix of covariates (if object is NULL).
#' @param w Optional treatment vector (if object is NULL).
#' @param y Optional outcome vector (if object is NULL).
#' @param k_folds Integer. Number of folds for cross-validation (default 3).
#' @param b_range Vector. Range of bandwidths to search. If NULL, derived heuristically.
#' @param lambda_range Vector. Range of ridge penalties to search.
#' @param lambdas_blend Vector. Range of mixing parameters for bias correction.
#'
#' @return An object of class \code{weight_kern}.
#' @export
weight_kern <- function(object = NULL, x = NULL, w = NULL, y = NULL,
                        k_folds = 3,
                        b_range = NULL,
                        lambda_range = 10^(seq(-4, 2, 0.5)),
                        lambdas_blend = seq(0, 1, length.out = 21)) {
  if (!is.null(object)) {
    x <- object@x
    y <- object@y
    z <- object@z
    w <- if (is.factor(z)) as.numeric(z == levels(z)[1]) else as.numeric(z)
  }
  
  if (is.null(x) || is.null(w) || is.null(y)) {
    stop("Provide either `object` with @x,@y,@z, or raw `x, w, y`.")
  }
  
  x <- as.matrix(x); w <- as.numeric(w); y <- as.numeric(y)
  n <- nrow(x)
  if (is.null(k_folds)) k_folds <- max(3, min(10, floor(n/4)))
  
  p_fit <- cv_krr_fast(x, w, k_folds = k_folds, b_range = b_range, lambda_range = lambda_range)
  p_hat <- p_fit$fit
  
  m_fit <- cv_krr_fast(x, y, k_folds = k_folds, b_range = b_range, lambda_range = lambda_range)
  m_hat <- m_fit$fit
  
  zr <- w - p_hat
  wt_tau <- zr^2
  ytilde_u <- (y - m_hat) / zr
  tau_u <- cv_krr_fast(x, ytilde_u, weights = wt_tau, k_folds = k_folds, 
                       b_range = b_range, lambda_range = lambda_range)
  tau0  <- sum((y - m_hat) * zr) / sum(wt_tau)
  beta1 <- stats::coef(stats::lm(y ~ w + p_hat))[2]
  
  fold2 <- sample(rep(1:2, length.out = n))
  best  <- list(score = Inf, lam = NA)
  
  for (lam in lambdas_blend) {
    c_lam <- lam * tau0 + (1 - lam) * beta1
    score <- 0
    for (kk in 1:2) {
      tr <- fold2 != kk; te <- !tr
      ystar_tr <- y[tr] - c_lam * w[tr]
      xi_fit <- cv_krr_fast(x[tr, , drop = FALSE], ystar_tr, k_folds = 2, 
                            b_range = b_range, lambda_range = lambda_range)
      xi_te <- krr_predict(xi_fit$model, x[te, , drop = FALSE])
      
      lab_te <- (y[te] - c_lam * w[te] - xi_te) / zr[te]
      score <- score + stats::var(lab_te, na.rm = TRUE)
    }
    if (score < best$score) best <- list(score = score, lam = lam)
  }
  
  tau0_blend <- best$lam * tau0 + (1 - best$lam) * beta1
  
  ys0 <- y - tau0_blend * w
  ys_fit <- cv_krr_fast(x, ys0, k_folds = k_folds, b_range = b_range, lambda_range = lambda_range)
  ys0_hat <- ys_fit$fit
  
  ytilde_r <- (y - tau0_blend * w - ys0_hat) / zr
  tau_r <- cv_krr_fast(x, ytilde_r, weights = wt_tau, k_folds = k_folds, 
                       b_range = b_range, lambda_range = lambda_range)
  
  structure(list(
    p_fit = p_fit, m_fit = m_fit, p_hat = p_hat, m_hat = m_hat,
    unrevised = list(tau_fit = tau_u),
    revised   = list(tau_fit = tau_r, ys_fit = ys_fit, tau0 = tau0_blend, blend_lambda = best$lam)
  ), class = "weight_kern")
}

#' Predictions for weight_kern objects
#'
#' @param object A fitted weight_kern object.
#' @param newx Matrix of new covariates. If NULL, returns fitted values.
#' @param which Character string. Which estimator to predict: "both", "unrevised", or "revised".
#' @param ... Additional arguments (currently unused).
#'
#' @return A numeric vector or data frame of predictions.
#' @export
predict.weight_kern <- function(object, newx = NULL,
                                which = c("both", "unrevised", "revised"), ...) {
  which <- match.arg(which)
  
  if (is.null(newx)) {
    tu <- object$unrevised$tau_fit$fit
    tr <- object$revised$tau_fit$fit + object$revised$tau0
  } else {
    newx <- as.matrix(newx)
    tu <- krr_predict(object$unrevised$tau_fit$model, newx)
    tr <- krr_predict(object$revised$tau_fit$model, newx) + object$revised$tau0
  }
  
  if (which == "unrevised") return(tu)
  if (which == "revised")   return(tr)
  data.frame(unrevised = tu, revised = tr)
}
