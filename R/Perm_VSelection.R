#' @title Covariate-Specific Permutation Screen for Effect Modifiers
#'
#' @description Localizes treatment effect heterogeneity to individual
#' covariates. After the global tests establish that \code{tau(X)} varies,
#' this screen asks along which covariates it varies: for each covariate
#' \code{X_j} it tests \code{H_0j}: \code{tau(x)} does not depend on
#' \code{x_j}, by permuting the j-th column of the held-out design and
#' re-evaluating the cross-fitted stage-2 model, so no additional model
#' fitting is required beyond the fits already used for estimation.
#'
#' Two permutation schemes are computed from the same fits, so their
#' comparison is controlled:
#' \itemize{
#'   \item \code{p_marg}: \code{X_j} is shuffled across all subjects. This
#'     breaks its association with the outcome, as intended, but also with
#'     the remaining covariates, so under dependence the model is evaluated
#'     off the support of \code{X} and null covariates correlated with true
#'     modifiers can be rejected too often (Strobl et al. 2008; Hooker et
#'     al. 2021).
#'   \item \code{p_cond}: conditional scheme. Let
#'     \code{S_j = \{k : |cor(X_j, X_k)| > cor_threshold\}}. A random forest
#'     of \code{X_j} on \code{X_{S_j}} is fitted, its out-of-bag fitted
#'     values are cut into \code{n_strata} quantile strata, and \code{X_j}
#'     is permuted only within strata, so a subject exchanges its value only
#'     with others whose covariate profile predicts a similar \code{X_j}.
#'     This approximates sampling from \code{P(X_j | X_-j)} (Berrett et al.
#'     2020). When \code{S_j} is empty the scheme reduces exactly to the
#'     marginal one.
#' }
#'
#' Permutations are batched: \code{chunk} permuted copies of a fold are
#' stacked into a single \code{predict()} call, which is what makes B in the
#' hundreds affordable with an ensemble DNN.
#'
#' @param object An object of class `Trt` containing the covariates, outcome, and treatment assignment.
#' @param k_folds Integer. The number of folds for cross-fitting. Default is 5.
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
#' @param en_dnn_ctrl A list of control parameters for the `ensemble_dnnet` function.
#'
#' @return A list with two elements. \code{screen} is a data.frame with one
#' row per covariate: \code{variable}, \code{n_cond} (size of the
#' conditioning set \code{S_j}), the permutation p-values \code{p_marg} and
#' \code{p_cond}, the inflation ratios \code{infl_marg} and \code{infl_cond}
#' (mean permuted loss over observed loss), and the within-sample importance
#' ranks \code{rank_marg} and \code{rank_cond} (1 = most important; ties in p
#' broken by inflation). \code{obs_loss} is the observed weighted loss of the
#' cross-fitted stage-2 model. Covariates whose conditional distribution given
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
                         en_dnn_ctrl = NULL) {
  z_fac <- if (is.factor(object@z)) object@z else factor(ifelse(object@z == 1, "A", "B"), levels = c("A", "B"))
  z_num <- if (is.numeric(object@z)) object@z else as.numeric(z_fac == "A")

  X <- as.matrix(object@x)
  y <- object@y
  n <- nrow(X)
  d <- ncol(X)
  vnames <- colnames(X)
  if (is.null(vnames)) vnames <- paste0("X", seq_len(d))
  colnames(X) <- vnames

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
  folds <- sample(rep(seq_len(k_folds), length.out = n))
  e_hat <- mu_hat <- rep(NA_real_, n)

  for (k in seq_len(k_folds)) {
    tr <- folds != k
    te <- folds == k

    z_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = z_fac[tr])
    z_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = z_obj), en_dnn_ctrl))
    pk <- deepTL::predict(z_mod, X[te, , drop = FALSE])
    e_hat[te] <- if (is.null(dim(pk))) as.numeric(pk) else as.numeric(pk[, "A"])

    y_obj <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = y[tr])
    y_mod <- do.call(deepTL::ensemble_dnnet, c(list(object = y_obj), en_dnn_ctrl))
    mu_hat[te] <- as.numeric(deepTL::predict(y_mod, X[te, , drop = FALSE]))
  }

  e_hat  <- pmin(pmax(e_hat, 0.01), 0.99)
  Ytilde <- (y - mu_hat) / (z_num - e_hat)
  w      <- (z_num - e_hat)^2

  ## ---- stage 2: cross-fitted tau-hat, fold models retained -----------------
  pred_obs <- numeric(n)
  stage2 <- vector("list", k_folds)
  for (k in seq_len(k_folds)) {
    tr <- folds != k
    te <- folds == k
    o <- deepTL::importDnnet(x = X[tr, , drop = FALSE], y = Ytilde[tr], w = w[tr])
    m <- do.call(deepTL::ensemble_dnnet, c(list(object = o), en_dnn_ctrl))
    pred_obs[te] <- as.numeric(deepTL::predict(m, X[te, , drop = FALSE]))
    stage2[[k]] <- m
  }
  obs_loss <- sum(w * (Ytilde - pred_obs)^2) / n

  ## ---- self-check: the batched permutation path must reproduce obs_loss ----
  # The permuted losses come from a batched predict() on stacked copies while
  # obs_loss does not. If any normalisation were recomputed from the prediction
  # input rather than reused from training, the two code paths would differ and
  # the observed and permuted statistics would not be exchangeable under H0.
  id_loss <- local({
    rs <- numeric(n)
    for (k in seq_len(k_folds)) {
      te <- which(folds == k); nt <- length(te)
      Xte <- X[te, , drop = FALSE]
      Xbig <- Xte[rep(seq_len(nt), times = 2L), , drop = FALSE]
      pb <- as.numeric(deepTL::predict(stage2[[k]], Xbig))
      rs[te] <- w[te] * (Ytilde[te] - pb[seq_len(nt)])^2
    }
    sum(rs) / n
  })
  if (abs(id_loss - obs_loss) / obs_loss > 1e-8)
    warning("batched and unbatched prediction paths differ; permutation p-values may not be calibrated")

  ## ---- conditioning strata: one forest per covariate, fitted once ----------
  Cx <- stats::cor(X)
  diag(Cx) <- 0
  Cx[!is.finite(Cx)] <- 0
  strata <- vector("list", d)
  n_cond <- integer(d)
  for (j in seq_len(d)) {
    S <- which(abs(Cx[j, ]) > cor_threshold)
    n_cond[j] <- length(S)
    if (!length(S)) { strata[[j]] <- rep(1L, n); next }
    rf <- ranger::ranger(y = X[, j], x = X[, S, drop = FALSE],
                         num.trees = 200, min.node.size = 20)
    fit <- rf$predictions
    qs <- unique(stats::quantile(fit, seq(0, 1, length.out = n_strata + 1), na.rm = TRUE))
    strata[[j]] <- if (length(qs) > 2)
      cut(fit, breaks = qs, include.lowest = TRUE, labels = FALSE) else rep(1L, n)
  }

  # one permuted copy of fold k's column j
  permute_col <- function(col, str, conditional) {
    nt <- length(col)
    if (!conditional) return(col[sample(nt)])
    for (bn in unique(str[!is.na(str)])) {
      ii <- which(str == bn)
      if (length(ii) > 1) col[ii] <- col[ii[sample(length(ii))]]
    }
    col
  }

  # batched: `chunk` permutations stacked into one predict() call per fold
  perm_losses <- function(j, conditional, nperm) {
    out <- numeric(nperm); done <- 0L
    while (done < nperm) {
      m <- min(chunk, nperm - done)
      resid_sq <- matrix(0, m, n)
      for (k in seq_len(k_folds)) {
        te <- which(folds == k); nt <- length(te)
        Xte <- X[te, , drop = FALSE]
        str_te <- strata[[j]][te]
        Xbig <- Xte[rep(seq_len(nt), times = m), , drop = FALSE]
        for (b in seq_len(m)) {
          rows <- ((b - 1) * nt + 1):(b * nt)
          Xbig[rows, j] <- permute_col(Xte[, j], str_te, conditional)
        }
        pb <- as.numeric(deepTL::predict(stage2[[k]], Xbig))
        for (b in seq_len(m)) {
          rows <- ((b - 1) * nt + 1):(b * nt)
          resid_sq[b, te] <- w[te] * (Ytilde[te] - pb[rows])^2
        }
      }
      out[(done + 1):(done + m)] <- rowSums(resid_sq) / n
      done <- done + m
    }
    out
  }

  ## ---- run the screen, optionally parallel over covariates -----------------
  run_one <- function(j) {
    Lm <- perm_losses(j, FALSE, B)
    # empty conditioning set: the conditional scheme IS the marginal one
    Lc <- if (n_cond[j] > 0) perm_losses(j, TRUE, B) else Lm
    data.frame(variable = vnames[j], n_cond = n_cond[j],
               p_marg = (1 + sum(Lm <= obs_loss)) / (B + 1),
               p_cond = (1 + sum(Lc <= obs_loss)) / (B + 1),
               infl_marg = mean(Lm) / obs_loss,
               infl_cond = mean(Lc) / obs_loss,
               row.names = NULL)
  }
  res <- if (n_cores > 1) {
    parallel::mclapply(seq_len(d), run_one,
                       mc.cores = min(n_cores, d), mc.preschedule = FALSE)
  } else {
    lapply(seq_len(d), run_one)
  }
  screen <- do.call(rbind, res)

  ## ---- within-sample ranking (ties in p broken by inflation) ---------------
  rank_by <- function(p, infl) {
    o <- order(p, -infl); r <- integer(length(p)); r[o] <- seq_along(o); r
  }
  screen$rank_marg <- rank_by(screen$p_marg, screen$infl_marg)
  screen$rank_cond <- rank_by(screen$p_cond, screen$infl_cond)

  list(screen = screen, obs_loss = obs_loss)
}
