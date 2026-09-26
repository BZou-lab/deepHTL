#' @importFrom stats dist median sd
#' @importFrom CompQuadForm davies liu
NULL

#' Kernel Score Test for Treatment Effect Heterogeneity
#'
#' The analytic global test of deepHTL, computed from cross-fitted nuisance estimates
#' (product form). Let \eqn{d_i = Z_i - \hat e(X_i)} and let
#' \eqn{R_i = Y_i - \tau_0 Z_i - \hat\mu(X_i)} be the residual, with \eqn{\tau_0 = 0} for the
#' unrevised residual and, for the revised residual, \eqn{\tau_0} the estimated constant
#' effect and \eqn{\hat\mu} the outcome model fitted to \eqn{Y - \tau_0 Z}. With
#' \eqn{\hat\theta = \sum_i d_i R_i / \sum_i d_i^2} (for the revised residual this is the
#' one-step correction of the constant), the scores
#' \eqn{q_i = d_i (R_i - \hat\theta d_i)} sum to zero exactly,
#' \eqn{\hat\sigma^2 = n^{-1} \sum_i (R_i - \hat\theta d_i)^2}, and the statistic is
#' \eqn{Q = q' K q / \hat\sigma^2} with \eqn{K} the Gaussian kernel matrix of the standardized
#' covariates (bandwidth = median pairwise distance). Under the null hypothesis of a constant
#' conditional average treatment effect, with the nuisance functions known,
#' \eqn{R - \hat\theta d = M_d \epsilon} with \eqn{M_d = I - d d'/(d'd)}, so
#' \eqn{q = D M_d \epsilon} with \eqn{D = diag(d)} and \eqn{Q} converges to
#' \eqn{\sum_j \lambda_j \chi^2_{1,j}} where \eqn{\lambda_j} are the eigenvalues of
#' \eqn{M_d D K D M_d}. The p-value is computed by the Davies method, with the Liu
#' approximation as a fallback.
#'
#' This is the kernel machine score test of Liu, Lin and Ghosh (2007) applied to the working
#' model \eqn{R = \theta d + d\, r(X) + \epsilon}, \eqn{r \sim GP(0, \sigma_r^2 K)}, testing
#' \eqn{\sigma_r^2 = 0}. With \eqn{\delta_e = \hat e - e} and \eqn{\delta_\mu = \hat\mu - \mu},
#' \eqn{E(q \mid X) = \delta_e \delta_\mu - \theta_0 \delta_e^2} under the null, so nuisance
#' error enters only through products and squares (Neyman orthogonality). The earlier sign
#' form (`form = "sign"`, kept for comparison) had a mean shift of first order in
#' \eqn{\delta_\mu} and inflated when the outcome model was inaccurate.
#'
#' @param Y Numeric outcome vector.
#' @param Z Treatment indicator (0/1).
#' @param e_hat Cross-fitted propensity scores, ideally clipped away from 0 and 1.
#' @param mu_hat Cross-fitted outcome regression of `Y - tau0 * Z` on the covariates.
#' @param X Numeric matrix of covariates (all p covariates, used only for the kernel).
#' @param tau0 Constant effect removed before `mu_hat` was fitted. `0` (default) gives the
#'   unrevised test, the estimated constant effect gives the revised test.
#' @param form `"product"` (default, the deepHTL test) or `"sign"`, the legacy statistic
#'   \eqn{s' K s} with \eqn{s_i = sign(d_i)(R_i - \hat\theta d_i)/\hat\sigma} and the
#'   eigenvalues of \eqn{P K P}, \eqn{P = I - |d||d|'/(d'd)}.
#' @param bandwidth `"median"` (default), the median pairwise Euclidean distance of the
#'   standardized covariates, or a positive number \eqn{h} for
#'   \eqn{K_{ij} = \exp\{-\|x_i - x_j\|^2 / (2 h^2)\}}.
#' @param K Optional precomputed n x n kernel matrix (`bandwidth` is then only recorded).
#' @param lambda Optional eigenvalues of the null matrix, for repeated calls with the same
#'   `X`, `e_hat` and `form` (they depend on `d` and `K` only).
#' @param eig_tol Eigenvalues below this value are dropped. Default 1e-8.
#' @param return_kernel Logical. Include `K` in the result so that a second call (for
#'   instance the revised test after the unrevised one) can reuse it. Default `FALSE`.
#'
#' @return An object of class `kernel_score_test`, a list with `Q` the statistic, `p_value`,
#'   `lambda` the eigenvalues of the null matrix, `theta_hat`, `sigma_hat`, `scores` the
#'   vector \eqn{q} (or \eqn{s \hat\sigma} for the sign form), `form`, `bandwidth`,
#'   `method` (`"davies"` or `"liu"`) and, when `return_kernel = TRUE`, `K`.
#'
#' @references
#' Liu, D., Lin, X. and Ghosh, D. (2007). Semiparametric regression of multidimensional
#' genetic pathway data: least-squares kernel machines and linear mixed models.
#' Biometrics, 63(4):1079-1088.
#'
#' Davies, R. B. (1980). The distribution of a linear combination of chi-squared random
#' variables. Journal of the Royal Statistical Society, Series C, 29(3):323-333.
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 300; p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' e <- plogis(0.5 * X[, 1]); Z <- rbinom(n, 1, e)
#' Y <- X[, 2] + Z * 3 + rnorm(n)
#' # oracle nuisances, constant effect: the unrevised residual keeps the constant
#' unrev <- kernel_score_test(Y, Z, e, X[, 2] + 3 * e, X, return_kernel = TRUE)
#' rev <- kernel_score_test(Y, Z, e, X[, 2], X, tau0 = 3, K = unrev$K, lambda = unrev$lambda)
#' c(unrev$p_value, rev$p_value)
#' }
#' @export
kernel_score_test <- function(Y, Z, e_hat, mu_hat, X, tau0 = 0, form = c("product", "sign"),
                              bandwidth = "median", K = NULL, lambda = NULL, eig_tol = 1e-8,
                              return_kernel = FALSE) {
  form <- match.arg(form)
  Y <- as.numeric(Y); Z <- as.numeric(Z); e_hat <- as.numeric(e_hat); mu_hat <- as.numeric(mu_hat)
  n <- length(Y)
  if (length(Z) != n || length(e_hat) != n || length(mu_hat) != n)
    stop("Y, Z, e_hat and mu_hat must have the same length")
  if (is.null(K)) {
    kern <- score_kernel(X, bandwidth)
    K <- kern$K; bandwidth <- kern$bandwidth
  } else {
    if (!is.matrix(K) || nrow(K) != n || ncol(K) != n) stop("K must be an n x n matrix")
    bandwidth <- if (is.numeric(bandwidth)) as.numeric(bandwidth) else NA_real_
  }

  d <- Z - e_hat
  R <- Y - tau0 * Z - mu_hat
  theta_hat <- sum(d * R) / sum(d^2)
  u <- R - theta_hat * d
  sigma_hat <- sqrt(mean(u^2))
  v <- if (form == "product") d * u / sigma_hat else sign(d) * u / sigma_hat
  Q <- as.numeric(crossprod(v, K %*% v))

  if (is.null(lambda)) lambda <- score_null_eigen(K, d, form, eig_tol)
  pv <- davies_pvalue(Q, lambda)

  out <- list(Q = Q, p_value = pv$p, lambda = lambda, theta_hat = theta_hat, sigma_hat = sigma_hat,
              scores = v * sigma_hat, form = form, bandwidth = bandwidth, method = pv$method, n = n)
  if (return_kernel) out$K <- K
  class(out) <- "kernel_score_test"
  out
}

#' @export
print.kernel_score_test <- function(x, ...) {
  cat(sprintf("Kernel score test (%s form): Q = %.3f, p-value = %.4g (%s, %d eigenvalues)\n",
              x$form, x$Q, x$p_value, x$method, length(x$lambda)))
  cat(sprintf("  theta_hat = %.4f, sigma_hat = %.4f, bandwidth = %.3f\n", x$theta_hat, x$sigma_hat, x$bandwidth))
  invisible(x)
}

# Gaussian kernel matrix of the standardized covariates. bandwidth = "median" uses the median
# pairwise Euclidean distance (1 when it is not positive), otherwise a positive number.
score_kernel <- function(X, bandwidth = "median") {
  X <- as.matrix(X)
  s <- apply(X, 2, stats::sd)
  s[!is.finite(s) | s == 0] <- 1
  Xs <- scale(X, center = TRUE, scale = s)
  Dm <- as.matrix(stats::dist(Xs))
  if (identical(bandwidth, "median")) {
    bw <- stats::median(Dm[upper.tri(Dm, diag = FALSE)])
    if (!is.finite(bw) || bw <= 0) bw <- 1
  } else {
    bw <- as.numeric(bandwidth)
    if (length(bw) != 1L || !is.finite(bw) || bw <= 0) stop("bandwidth must be \"median\" or a positive number")
  }
  list(K = exp(-(Dm^2) / (2 * bw^2)), bandwidth = bw)
}

# Eigenvalues of the null matrix: M_d D K D M_d for the product form (M_d = I - d d'/(d'd),
# D = diag(d)), P K P for the sign form (P = I - |d||d|'/(d'd)). Only eigenvalues above
# eig_tol are kept.
score_null_eigen <- function(K, d, form = c("product", "sign"), eig_tol = 1e-8) {
  form <- match.arg(form)
  n <- length(d)
  if (form == "product") {
    M <- diag(n) - tcrossprod(d) / sum(d^2)
    A <- M %*% (d * t(d * K)) %*% M
  } else {
    a <- abs(d)
    P <- diag(n) - tcrossprod(a) / sum(d^2)
    A <- P %*% K %*% P
  }
  A <- 0.5 * (A + t(A))
  ev <- eigen(A, symmetric = TRUE, only.values = TRUE)$values
  ev[ev > eig_tol]
}

# P(sum_j lambda_j chi^2_1 > Q) by Davies (1980), Liu et al. (2009) when Davies fails or
# returns a value outside [0, 1].
davies_pvalue <- function(Q, lambda) {
  if (!length(lambda)) return(list(p = NA_real_, method = NA_character_))
  p <- tryCatch(CompQuadForm::davies(Q, lambda = lambda)$Qq, error = function(e) NA_real_)
  method <- "davies"
  if (is.na(p) || !is.finite(p) || p < 0 || p > 1) {
    p <- tryCatch(CompQuadForm::liu(Q, lambda = lambda), error = function(e) NA_real_)
    method <- "liu"
  }
  list(p = p, method = method)
}
