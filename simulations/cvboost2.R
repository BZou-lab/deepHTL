###############################################################################
# cvboost2.R -- rlearner::cvboost re-implemented against the xgboost >= 2 R API
# (the installed 3.2.1 dropped cb.cv.predict, which rlearner 1.1.0 still calls).
# Same procedure: 10 random hyper-parameter draws, k-fold xgb.cv with early
# stopping, keep the draw with the smallest CV loss, refit on all data with the
# best number of rounds. predict() without newx returns the out-of-fold CV
# predictions (as rlearner does), with newx the refitted booster's predictions.
# nthread = 1: the SLURM tasks are single-core.
###############################################################################
cvboost2 <- function(x, y, weights = NULL, k_folds = NULL,
                     objective = c("reg:squarederror", "binary:logistic"),
                     ntrees_max = 1000, num_search_rounds = 10,
                     early_stopping_rounds = 10, nthread = 1, verbose = FALSE) {
  objective <- match.arg(objective)
  eval <- if (objective == "reg:squarederror") "rmse" else "logloss"
  if (is.null(k_folds)) k_folds <- floor(max(3, min(10, length(y) / 4)))
  if (is.null(weights)) weights <- rep(1, length(y))
  dtrain <- xgboost::xgb.DMatrix(data = as.matrix(x), label = y, weight = weights, nthread = nthread)
  best <- list(loss = Inf)
  get_pred <- function(f) if (!is.null(f$pred)) f$pred else if (!is.null(f$cv_predict$pred)) f$cv_predict$pred else f$cv_predict
  get_best_iter <- function(f) {
    bi <- f$best_iteration; if (is.null(bi)) bi <- f$early_stop$best_iteration
    if (is.null(bi) || is.na(bi)) bi <- nrow(f$evaluation_log); as.integer(bi) }
  for (iter in seq_len(num_search_rounds)) {
    param <- list(objective = objective, eval_metric = eval, nthread = nthread,
                  subsample = sample(c(0.5, 0.75, 1), 1), colsample_bytree = sample(c(0.6, 0.8, 1), 1),
                  eta = sample(c(0.005, 0.01, 0.015, 0.025, 0.05, 0.08, 0.1, 0.2), 1),
                  max_depth = sample(3:20, 1), gamma = runif(1, 0, 0.2),
                  min_child_weight = sample(1:20, 1), max_delta_step = sample(1:10, 1))
    seed_number <- sample.int(1e5, 1); set.seed(seed_number)
    fit <- xgboost::xgb.cv(params = param, data = dtrain, nrounds = ntrees_max, nfold = k_folds,
                           prediction = TRUE, early_stopping_rounds = early_stopping_rounds,
                           maximize = FALSE, verbose = verbose)
    metric <- paste0("test_", eval, "_mean")
    loss <- min(as.numeric(fit$evaluation_log[[metric]]))
    if (loss < best$loss) best <- list(loss = loss, seed = seed_number, param = param,
                                       pred = as.numeric(get_pred(fit)), niter = get_best_iter(fit))
  }
  set.seed(best$seed)
  xgb_fit <- xgboost::xgb.train(params = best$param, data = dtrain, nrounds = best$niter, verbose = 0)
  structure(list(xgb_fit = xgb_fit, cv_pred = best$pred, best_param = best$param,
                 best_loss = best$loss, best_ntreelimit = best$niter), class = "cvboost2")
}
predict.cvboost2 <- function(object, newx = NULL, ...) {
  if (is.null(newx)) return(object$cv_pred)
  as.numeric(predict(object$xgb_fit, xgboost::xgb.DMatrix(as.matrix(newx), nthread = 1)))
}
