#' Candidate settings for the bagged-DNN
#'
#' Default grid searched by the `tune` argument of [cv_perm_test()], [davies_test()],
#' [weight_dnn()] and [cv_perm_vsel()]: the L1 penalty and the mini-batch size of the
#' networks. The number of epochs is not part of the grid because every network keeps the
#' weights of its best validation epoch (early stopping inside `deepTL::ensemble_dnnet`), so
#' `n.epoch` only needs to be a generous upper bound.
#'
#' @param l1 Numeric vector of L1 penalties.
#' @param n_batch Integer vector of mini-batch sizes.
#' @return A data.frame with one candidate per row, columns `l1.reg` and `n.batch`.
#' @export
dnn_tune_grid <- function(l1 = c(1e-7, 1e-5, 1e-3), n_batch = c(64, 128, 256)) {
  expand.grid(l1.reg = l1, n.batch = n_batch, KEEP.OUT.ATTRS = FALSE)
}

# Select the `esCtrl` entries named in `grid` for one ensemble_dnnet fit.
#
# `object` is a dnnetInput (deepTL::importDnnet), weighted or not. For every candidate a
# pilot ensemble of `n_tune` networks is fitted and scored by the mean of deepTL's per-network
# out-of-bag loss (`@loss`): each network is trained on a bootstrap sample and the slot holds
# its improvement over the null model on the observations left out of that sample, squared
# error for regression, log-likelihood for classification, weighted when the input carries
# weights, and computed without the penalty term. Higher is better. All candidates start from
# the same RNG state, so they share bootstrap samples and initial weights, and the caller's
# RNG stream is restored afterwards (it advances by exactly one draw).
#
# Stage 1 (propensity, outcome) is thereby selected by out-of-sample predictive loss, stage 2
# (the weighted regression of the pseudo-outcome on X) by the out-of-bag weighted squared
# error, which is the R-loss on held-out data.
tune_en_dnn_ctrl <- function(object, ctrl, grid = NULL, n_tune = 5L) {
  if (is.null(grid) || isFALSE(grid)) return(ctrl)
  if (isTRUE(grid)) grid <- dnn_tune_grid()
  grid <- as.data.frame(grid, stringsAsFactors = FALSE)
  if (!nrow(grid)) return(ctrl)
  if (nrow(grid) == 1L) { ctrl$esCtrl[names(grid)] <- as.list(grid[1L, , drop = FALSE]); return(ctrl) }
  seed <- sample.int(.Machine$integer.max, 1L)
  had_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = globalenv(), inherits = FALSE) else NULL
  gain <- vapply(seq_len(nrow(grid)), function(i) {
    ct <- ctrl
    ct$n.ensemble <- as.integer(n_tune)
    ct$verbose <- FALSE
    ct$esCtrl[names(grid)] <- as.list(grid[i, , drop = FALSE])
    set.seed(seed)
    fit <- tryCatch(do.call(deepTL::ensemble_dnnet, c(list(object = object), ct)),
                    error = function(e) NULL)
    if (is.null(fit)) return(NA_real_)
    l <- methods::slot(fit, "loss")
    l <- l[is.finite(l)]
    if (length(l)) mean(l) else NA_real_
  }, numeric(1))
  if (had_seed) assign(".Random.seed", old_seed, envir = globalenv())
  tab <- cbind(grid, oob_gain = gain, selected = FALSE)
  if (all(is.na(gain))) { attr(ctrl, "tuning") <- tab; return(ctrl) }
  best <- which.max(gain)
  tab$selected[best] <- TRUE
  ctrl$esCtrl[names(grid)] <- as.list(grid[best, , drop = FALSE])
  attr(ctrl, "tuning") <- tab
  ctrl
}

# One-row record of a selection made by tune_en_dnn_ctrl(), NULL when nothing was tuned.
tune_record <- function(ctrl, fold, model) {
  tab <- attr(ctrl, "tuning")
  if (is.null(tab) || !any(tab$selected)) return(NULL)
  sel <- tab[tab$selected, setdiff(names(tab), c("oob_gain", "selected")), drop = FALSE]
  data.frame(fold = fold, model = model, sel, oob_gain = tab$oob_gain[tab$selected],
             row.names = NULL, stringsAsFactors = FALSE)
}
