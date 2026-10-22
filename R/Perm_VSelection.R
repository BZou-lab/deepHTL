#' @title Covariate-Specific Permutation Screen for Effect Modifiers
#'
#' @description Localizes treatment effect heterogeneity to individual
#' covariates. After the global tests establish that \code{tau(X)} varies,
#' this screen asks along which covariates it varies: for each covariate
#' \code{X_j} it tests \code{H_0j}: \code{tau(x)} does not depend on
#' \code{x_j}. The stage-2 model is fitted once on one half of the sample
#' (S1, drawn within treatment arm), and \code{X_j} is permuted within the
#' other half (S2), where the permuted loss is compared with the observed one.
#'
#' Two permutation schemes are computed from the same fit, so their
#' comparison is controlled:
#' \itemize{
#'   \item \code{p_marg}: \code{X_j} is shuffled across all subjects of S2. This
#'     breaks its association with the outcome, as intended, but also with
#'     the remaining covariates, so under dependence the model is evaluated
#'     off the support of \code{X} and null covariates correlated with true
#'     modifiers can be rejected too often (Strobl et al. 2008; Hooker et
#'     al. 2021).
#'   \item \code{p_cond}: conditional scheme. Let
#'     \code{S_j = \{k : |cor(X_j, X_k)| > cor_threshold\}}. A random forest
#'     of \code{X_j} on \code{X_{S_j}} is fitted, its out-of-bag fitted
#'     values on S2 are cut into \code{n_strata} quantile strata, and \code{X_j}
#'     is permuted only within strata, so a subject exchanges its value only
#'     with others whose covariate profile predicts a similar \code{X_j}.
#'     This approximates sampling from \code{P(X_j | X_-j)} (Berrett et al.
#'     2020). When \code{S_j} is empty the scheme reduces exactly to the
#'     marginal one.
#' }
#'
#' Permutations are batched: \code{chunk} permuted copies of S2 are
#' stacked into a single \code{predict()} call, which is what makes B in the
#' hundreds affordable with an ensemble DNN.
#'
#' @param object An object of class `Trt` containing the covariates, outcome, and treatment assignment.
#' @param k_folds Integer. The number of folds for cross-fitting the nuisance models. Default is 5.
#' @param B Integer. The number of permutations per covariate and scheme. Default is 500.
#' @param cor_threshold Numeric. Threshold c defining the conditioning set
#' \code{S_j}. Default is 0.2.
#' @param n_strata Integer. Number of quantile strata M for the conditional
#' scheme. Default is 5.
#' @param chunk Integer. Number of permuted copies stacked per \code{predict()}
#' call. Default is 50.
#' @param n_cores Integer. Number of cores used to parallelize over covariates
#' via \code{parallel::mclapply} (forking; on Windows use 1). Default is 1.
#' For reproducible parallel runs set \code{RNGkind("L'Ecuyer-CMRG")} before
#' seeding.
#' @param en_dnn_ctrl A list of control parameters for the `ensemble_dnnet` function, used
#'   for the second-stage effect network.
#' @param tune Optional grid of candidate network settings: a data.frame whose columns are
#'   `esCtrl` entries (see [dnn_tune_grid()]) or `TRUE` for the default grid. When given, the
#'   settings of every network are selected within each training fold by the mean out-of-bag
#'   loss of a pilot ensemble, log-loss for the propensity network, squared error for the
#'   outcome network and weighted squared error (the R-loss on held-out data) for the
#'   second-stage networks. `NULL` (default) uses the control list as supplied.
#' @param n_tune Integer. Networks per candidate in the pilot ensembles. Default 5.
#' @param split_frac Numeric. Share of each treatment arm assigned to the training half S1.
#'   Default 0.5.
#' @param groups Optional vector of group labels (e.g. donor id), one per subject. Members of a
#'   group are kept in the same half and in the same cross-fitting fold. `NULL` (default)
#'   treats every subject as its own group.
#' @param var_groups Optional named list of column sets (names or indices of columns of X)
#'   that are permuted jointly as whole rows, e.g. the indicator columns of one categorical
#'   covariate. Columns in no set are screened one at a time. For a set, the conditional
#'   scheme stratifies on its first principal component. Default `NULL`.
#'
#' @return A list. \code{screen} is a data.frame with one row per screened
#' covariate or column set: \code{variable}, \code{columns} (the permuted
#' columns), \code{n_cond} (size of the
#' conditioning set \code{S_j}), the permutation p-values \code{p_marg} and
#' \code{p_cond}, the inflation ratios \code{infl_marg} and \code{infl_cond}
#' (mean permuted loss over observed loss), and the within-sample importance
#' ranks \code{rank_marg} and \code{rank_cond} (1 = most important; ties in p
#' broken by inflation). \code{obs_loss} is the observed weighted loss on S2 of
#' the stage-2 model fitted on S1, \code{n_train} and \code{n_test} are the sizes
#' of S1 and S2. \code{tuning} lists the settings selected in each
#' fold (fold \code{NA} for the stage-2 fit) when \code{tune} is used, otherwise it is \code{NULL}. Covariates whose conditional distribution given
#' the others is degenerate (e.g. deterministic functions of other columns)
#' should be removed before screening.
#'
#' @references
#' Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T. and Zeileis, A.
#' (2008). Conditional variable importance for random forests.
#' \emph{BMC Bioinformatics}, 9:307.
#'
#' Hooker, G., Mentch, L. and Zhou, S. (2021). Unrestricted permutation
#' forces extrapolation: variable importance requires at least one more
#' model, or there is no free variable importance.
#' \emph{Statistics and Computing}, 31:82.
#'
#' Berrett, T. B., Wang, Y., Barber, R. F. and Samworth, R. J. (2020). The
#' conditional permutation test for independence while controlling for
#' confounders. \emph{JRSS-B}, 82(1):175-197.
#'
#' @importFrom stats cor quantile
#' @importFrom parallel mclapply
#' @importFrom ranger ranger
#' @export
cv_perm_vsel <- function(object, k_folds = 5, B = 500, cor_threshold = 0.2,
                         n_strata = 5, chunk = 50, n_cores = 1,
                         en_dnn_ctrl = NULL, tune = NULL, n_tune = 5,
                         split_frac = 0.5, groups = NULL, var_groups = NULL) {
  z_fac <- if (is.factor(object@z)) object@z else factor(ifelse(object@z == 1, "A", "B"), levels = c("A", "B"))
  z_num <- if (is.numeric(object@z)) object@z else as.numeric(z_fac == "A")

  X <- as.matrix(object@x)
  y <- object@y
  n <- nrow(X)
  d <- ncol(X)
  vnames <- colnames(X)
  if (is.null(vnames)) vnames <- paste0("X", seq_len(d))
  colnames(X) <- vnames
  check_split_args(n, split_frac, groups)

  ## screened units: the column sets of var_groups, every other column on its own
  units <- as.list(seq_len(d))
  names(units) <- vnames
  if (!is.null(var_groups)) {
    gcols <- lapply(var_groups, function(v) if (is.character(v)) match(v, vnames) else as.integer(v))
    all_cols <- unlist(gcols)
    if (is.null(names(var_groups)) || any(!nzchar(names(var_groups))) ||
        any(lengths(gcols) == 0) || anyNA(all_cols) || any(!all_cols %in% seq_len(d)) ||
        anyDuplicated(all_cols))
      stop("var_groups must be a named list of distinct, non-empty sets of columns of X")
    single <- setdiff(seq_len(d), all_cols)
    units <- c(gcols, units[single])
    if (anyDuplicated(names(units))) stop("names of var_groups must differ from the column names")
    units <- units[order(vapply(units, min, numeric(1)))]
  }
  u <- length(units)

  if (is.null(en_dnn_ctrl)) {
    en_dnn_ctrl <- list(
      n.ensemble = 30, verbose = FALSE,
      esCtrl = list(
        n.hidden = c(128, 64, 32),
        n.batch = 256,
        n.epoch = 120,
        norm.x = TRUE, norm.y = TRUE,
        activate = "relu", accel = "rcpp",
        l1.reg = 1e-3,
        plot = FALSE,
        learning.rate.adaptive = "adam",
        early.stop.det = 20
      )
    )
  }

  ## ---- stage 1: cross-fitted nuisances -------------------------------------
  folds <- if (is.null(groups)) sample(rep(seq_len(k_folds), length.out = n)) else
    group_parts(z_num, groups, seq_len(k_folds - 1) / k_folds)
  e_hat <- mu_hat <- rep(NA_real_, n)
  tune_log <- list()
  ctrl1 <- .ctrl1()
  Xmu <- .x_in(X)

  for (k in seq_len(k_folds)) {
    tr <- folds != k
    te <- folds == k

    z_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = z_fac[tr])
    ctrl_z_mod <- tune_en_dnn_ctrl(z_obj, ctrl1, tune, n_tune)
    tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_z_mod, k, "propensity")
    z_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = z_obj), ctrl_z_mod))
    pk <- deepTL::predict(z_mod, X[te, , drop = FALSE])
    e_hat[te] <- if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])

    y_obj <- deepTL::importDnnet(x = Xmu[tr, , drop = FALSE], y = y[tr])
    ctrl_y_mod <- tune_en_dnn_ctrl(y_obj, ctrl1, tune, n_tune)
    tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_y_mod, k, "outcome")
    y_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = y_obj), ctrl_y_mod))
    mu_hat[te] <- as.numeric(deepTL::predict(y_mod, Xmu[te, , drop = FALSE]))
  }

  e_hat  <- pmin(pmax(e_hat, 0.01), 0.99)
  Ytilde <- (y - mu_hat) / (z_num - e_hat)
  w      <- (z_num - e_hat)^2

  ## ---- stage 2: tau-hat fitted once on S1, evaluated on S2 -----------------
  S1 <- group_parts(z_num, groups, split_frac) == 1L
  i2 <- which(!S1)
  n1 <- sum(S1); n2 <- length(i2)
  if (n1 == 0 || n2 == 0) stop("both halves of the sample split must be non-empty")
  o <- deepTL::importDnnet(x = X[S1, , drop = FALSE], y = Ytilde[S1], w = w[S1])
  ctrl_m <- tune_en_dnn_ctrl(o, en_dnn_ctrl, tune, n_tune)
  tune_log[[length(tune_log) + 1]] <- tune_record(ctrl_m, NA_integer_, "stage2")
  stage2 <- do.call(deepTL::ensemble_dnnet, c(list(object = o), ctrl_m))
  X2 <- X[i2, , drop = FALSE]
  y2 <- Ytilde[i2]
  w2 <- w[i2]
  pred_obs <- as.numeric(deepTL::predict(stage2, X2))
  obs_loss <- sum(w2 * (y2 - pred_obs)^2) / n2

  ## ---- self-check: the batched permutation path must reproduce obs_loss ----
  # The permuted losses come from a batched predict() on stacked copies while
  # obs_loss does not. If any normalisation were recomputed from the prediction
  # input rather than reused from training, the two code paths would differ and
  # the observed and permuted statistics would not be exchangeable under H0.
  id_loss <- local({
    pb <- as.numeric(deepTL::predict(stage2, X2[rep(seq_len(n2), times = 2L), , drop = FALSE]))
    sum(w2 * (y2 - pb[seq_len(n2)])^2) / n2
  })
  if (abs(id_loss - obs_loss) / obs_loss > 1e-8)
    warning("batched and unbatched prediction paths differ; permutation p-values may not be calibrated")

  ## ---- conditioning strata on S2: one forest per unit, fitted once ---------
  Cx <- stats::cor(X)
  diag(Cx) <- 0
  Cx[!is.finite(Cx)] <- 0
  strata <- vector("list", u)
  n_cond <- integer(u)
  for (j in seq_len(u)) {
    cols <- units[[j]]
    S <- setdiff(which(colSums(abs(Cx[cols, , drop = FALSE]) > cor_threshold) > 0), cols)
    n_cond[j] <- length(S)
    if (!length(S)) { strata[[j]] <- rep(1L, n2); next }
    # a column set is represented by the first principal component of its standardized columns
    target <- if (length(cols) == 1L) X2[, cols] else {
      sc <- scale(X2[, cols, drop = FALSE]); sc[!is.finite(sc)] <- 0
      svd(sc, nu = 1L, nv = 0L)$u[, 1L]
    }
    rf <- ranger::ranger(y = target, x = X2[, S, drop = FALSE],
                         num.trees = 200, min.node.size = 20)
    fit <- rf$predictions
    qs <- unique(stats::quantile(fit, seq(0, 1, length.out = n_strata + 1), na.rm = TRUE))
    strata[[j]] <- if (length(qs) > 2)
      cut(fit, breaks = qs, include.lowest = TRUE, labels = FALSE) else rep(1L, n2)
  }

  # one permuted row order of S2 (within strata for the conditional scheme)
  permute_rows <- function(str, conditional) {
    if (!conditional) return(sample.int(n2))
    idx <- seq_len(n2)
    for (bn in unique(str[!is.na(str)])) {
      ii <- which(str == bn)
      if (length(ii) > 1) idx[ii] <- ii[sample(length(ii))]
    }
    idx
  }

  # batched: `chunk` permutations of S2 stacked into one predict() call; the
  # columns of a unit move together as whole rows
  perm_losses <- function(j, conditional, nperm) {
    cols <- units[[j]]
    out <- numeric(nperm); done <- 0L
    while (done < nperm) {
      m <- min(chunk, nperm - done)
      Xbig <- X2[rep(seq_len(n2), times = m), , drop = FALSE]
      for (b in seq_len(m)) {
        rows <- ((b - 1) * n2 + 1):(b * n2)
        Xbig[rows, cols] <- X2[permute_rows(strata[[j]], conditional), cols, drop = FALSE]
      }
      pb <- as.numeric(deepTL::predict(stage2, Xbig))
      for (b in seq_len(m)) {
        rows <- ((b - 1) * n2 + 1):(b * n2)
        out[done + b] <- sum(w2 * (y2 - pb[rows])^2) / n2
      }
      done <- done + m
    }
    out
  }

  ## ---- run the screen, optionally parallel over units ----------------------
  run_one <- function(j) {
    Lm <- perm_losses(j, FALSE, B)
    # empty conditioning set: the conditional scheme IS the marginal one
    Lc <- if (n_cond[j] > 0) perm_losses(j, TRUE, B) else Lm
    data.frame(variable = names(units)[j],
               columns = paste(vnames[units[[j]]], collapse = "+"),
               n_cond = n_cond[j],
               p_marg = (1 + sum(Lm <= obs_loss)) / (B + 1),
               p_cond = (1 + sum(Lc <= obs_loss)) / (B + 1),
               infl_marg = mean(Lm) / obs_loss,
               infl_cond = mean(Lc) / obs_loss,
               row.names = NULL)
  }
  res <- if (n_cores > 1) {
    parallel::mclapply(seq_len(u), run_one,
                       mc.cores = min(n_cores, u), mc.preschedule = FALSE)
  } else {
    lapply(seq_len(u), run_one)
  }
  ok <- vapply(res, is.data.frame, logical(1))
  if (length(res) != u || !all(ok))
    stop("screen failed for: ", paste(names(units)[!ok], collapse = ", "))
  screen <- do.call(rbind, res)

  ## ---- within-sample ranking (ties in p broken by inflation) ---------------
  rank_by <- function(p, infl) {
    o <- order(p, -infl); r <- integer(length(p)); r[o] <- seq_along(o); r
  }
  screen$rank_marg <- rank_by(screen$p_marg, screen$infl_marg)
  screen$rank_cond <- rank_by(screen$p_cond, screen$infl_cond)

  list(screen = screen, obs_loss = obs_loss, n_train = n1, n_test = n2,
       tuning = if (length(tune_log)) do.call(rbind, tune_log) else NULL)
}
