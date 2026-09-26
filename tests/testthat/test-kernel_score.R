test_that("kernel_score_test follows the product-form algebra and returns a valid p-value", {
  set.seed(11)
  n <- 150; p <- 4
  X <- matrix(rnorm(n * p), n, p)
  e <- plogis(0.5 * X[, 1]); Z <- rbinom(n, 1, e)
  mu <- X[, 2] - X[, 3]^2
  Y <- mu + 2 * Z + rnorm(n)

  res <- kernel_score_test(Y, Z, e, mu + 2 * e, X, return_kernel = TRUE)
  expect_s3_class(res, "kernel_score_test")
  expect_true(is.finite(res$Q) && res$Q >= 0)
  expect_true(res$p_value >= 0 && res$p_value <= 1)
  expect_equal(sum(res$scores), 0, tolerance = 1e-10)
  expect_true(all(res$lambda > 0))

  # explicit algebra: theta, q, sigma, Q = q' K q / sigma^2, eigenvalues of M_d D K D M_d
  d <- Z - e; R <- Y - (mu + 2 * e)
  theta <- sum(d * R) / sum(d^2); u <- R - theta * d; q <- d * u; s2 <- mean(u^2)
  expect_equal(res$theta_hat, theta)
  expect_equal(res$Q, as.numeric(crossprod(q, res$K %*% q)) / s2)
  M <- diag(n) - tcrossprod(d) / sum(d^2)
  A <- M %*% diag(d) %*% res$K %*% diag(d) %*% M
  ev <- eigen(0.5 * (A + t(A)), symmetric = TRUE, only.values = TRUE)$values
  expect_equal(res$lambda, ev[ev > 1e-8], tolerance = 1e-8)

  # revised call reusing the kernel and the eigenvalues
  rev <- kernel_score_test(Y, Z, e, mu, X, tau0 = 2, K = res$K, lambda = res$lambda)
  expect_true(rev$p_value >= 0 && rev$p_value <= 1)
  expect_equal(rev$lambda, res$lambda)

  # legacy sign form still runs
  sg <- kernel_score_test(Y, Z, e, mu + 2 * e, X, form = "sign")
  expect_true(sg$p_value >= 0 && sg$p_value <= 1)
})

test_that("screen_augment keeps at least min_keep covariates and builds the inputs", {
  set.seed(5)
  n <- 200; p <- 6
  X <- matrix(rnorm(n * p), n, p)
  Z <- rbinom(n, 1, plogis(0.6 * X[, 3] * X[, 4]))
  Y <- X[, 1] - X[, 2]^2 + Z + rnorm(n)
  sel <- screen_augment(X, Y, Z, min_keep = 5)
  expect_s3_class(sel, "screen_augment")
  expect_true(length(sel$vars) >= 5)
  expect_true(all(sel$vars %in% seq_len(p)))

  Xmu <- nuisance_input(sel, X)
  Xe <- nuisance_input(sel, X, augmented = FALSE)
  expect_equal(nrow(Xmu), n); expect_equal(nrow(Xe), n)
  expect_equal(ncol(Xe), length(sel$vars))
  expect_equal(ncol(Xmu), length(sel$vars) + sum(sel$feats > p))
  expect_equal(unname(Xe), unname(X[, sel$vars, drop = FALSE]))
  # single new row
  expect_equal(dim(nuisance_input(sel, X[1, , drop = FALSE])), c(1L, ncol(Xmu)))
})

test_that("dnn_ctrl returns the two training budgets", {
  o <- dnn_ctrl("outcome"); e <- dnn_ctrl("propensity", l1 = 1e-5)
  expect_equal(unlist(o$esCtrl[c("n.batch", "n.epoch", "early.stop.det")]), c(n.batch = 128, n.epoch = 500, early.stop.det = 50))
  expect_equal(unlist(e$esCtrl[c("n.batch", "n.epoch", "early.stop.det")]), c(n.batch = 256, n.epoch = 120, early.stop.det = 20))
  expect_equal(e$esCtrl$l1.reg, 1e-5)
  expect_equal(o$n.ensemble, 30)
})
