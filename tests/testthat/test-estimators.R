test_that("All estimators run and predict correctly", {
  set.seed(123)
  n <- 100
  p <- 5
  x <- matrix(rnorm(n * p), n, p)
  z <- rbinom(n, 1, 0.5)
  y <- rnorm(n)
  obj <- deepTL::importTrt(x = x, y = y, z = z)
  newx <- matrix(rnorm(5 * p), 5, p)

  # --- Test Lasso ---
  fit_lasso <- weight_lasso(obj)
  expect_s3_class(fit_lasso, "weight_lasso")
  pred_lasso <- predict(fit_lasso, newx, which = "revised")
  expect_type(pred_lasso, "double")
  expect_length(pred_lasso, 5)
  
  # --- Test Kernel ---
  fit_kern <- weight_kern(obj)
  expect_s3_class(fit_kern, "weight_kern")
  pred_kern <- predict(fit_kern, newx, which = "revised")
  expect_length(pred_kern, 5)
  
  # --- Test XGBoost ---
  skip_if_not_installed("xgboost")
  fit_xgb <- weight_xgboost(obj)
  expect_s3_class(fit_xgb, "weight_xgboost")
  pred_xgb <- predict(fit_xgb, newx, which = "revised")
  expect_length(pred_xgb, 5)

  # --- Test DNN ---
  fit_dnn <- weight_dnn(obj)
  expect_s3_class(fit_dnn, "weight_dnn")
  pred_dnn <- predict(fit_dnn, newx, which = "revised")
  expect_length(pred_dnn, 5)
})

test_that("Davies test returns valid p-values", {
  skip_on_cran()
  
  set.seed(123)
  n <- 100; p <- 5
  x <- matrix(rnorm(n * p), n, p)
  z <- as.factor(rbinom(n, 1, 0.5)); levels(z) <- c("A", "B")
  y <- rnorm(n)
  obj <- deepTL::importTrt(x = x, y = y, z = z)
  
  ctrl <- list(
    n.ensemble = 3, verbose = FALSE,
    esCtrl = list(n.hidden = c(10), n.batch = 10, n.epoch = 5)
  )
  
  res <- davies_test(obj, ctrl = ctrl)
  
  expect_true(!is.na(res$p_davies) && is.numeric(res$p_davies))
  expect_true(res$p_davies >= 0 && res$p_davies <= 1)
})

test_that("Cross-fitted permutation test returns valid structure and p-values", {
  skip_on_cran() 
  
  set.seed(123)
  n <- 100; p <- 5
  x <- matrix(rnorm(n * p), n, p)
  z <- as.factor(rbinom(n, 1, 0.5)); levels(z) <- c("A", "B")
  y <- rnorm(n)
  obj <- deepTL::importTrt(x = x, y = y, z = z)
  
  # Tiny network and ensemble controls for fast testing
  ctrl <- list(
    n.ensemble = 2, verbose = FALSE,
    esCtrl = list(n.hidden = c(10), n.batch = 10, n.epoch = 2)
  )
  
  # Run the permutation test with minimal folds and B
  res <- cv_perm_test(obj, k_folds = 2, B = 5, en_dnn_ctrl = ctrl)
  
  # Check structure
  expect_type(res, "list")
  expect_named(res, c("unrevised", "revised"))
  
  # Check unrevised outputs
  expect_true(is.numeric(res$unrevised$obs_mse))
  expect_true(is.numeric(res$unrevised$p_value))
  expect_true(res$unrevised$p_value >= 0 && res$unrevised$p_value <= 1)
  
  # Check revised outputs
  expect_true(is.numeric(res$revised$obs_mse))
  expect_true(is.numeric(res$revised$p_value))
  expect_true(res$revised$p_value >= 0 && res$revised$p_value <= 1)
})

