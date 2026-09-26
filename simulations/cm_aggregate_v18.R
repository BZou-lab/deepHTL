###############################################################################
# cm_aggregate_v18.R -- Table 3 of the paper from the output of cm_test_v18r2.R.
#   Rscript cm_aggregate_v18.R <out dir rho 0> <out dir rho 0.3> [alpha]
# Reads every test_<tau>_c<cfg>_r<rep>.RData below the directories given (several
# directories per rho may be passed as a comma-separated list) and prints, for each
# (tau, estimator, n, p, sigma), the rejection rate at alpha (default 0.05) of
#   Anal. = the product-form kernel score test (p_score_u, p_score_r)
#   Perm. = the permutation test with the fold x arm shuffle (p_perm_u, p_perm_r)
# under rho = 0 and rho = 0.3, in the row order of Table 3, plus the number of
# replications and the mean nuisance error (rmse_e, rmse_mu, rmse_ms) per cell.
# The legacy sign-form p-values (p_davies_u/r) and the fold x arm x e_hat-quintile
# permutation (p_permstr_u/r) are kept in the long summary file for reference.
###############################################################################
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript cm_aggregate_v18.R <dirs rho 0> <dirs rho 0.3> [alpha]")
alpha <- if (length(args) >= 3) as.numeric(args[3]) else 0.05
read_dirs <- function(dirs) {
  fs <- unlist(lapply(strsplit(dirs, ",")[[1]], list.files, pattern = "^test_.*\\.RData$", full.names = TRUE, recursive = TRUE))
  do.call(rbind, lapply(fs, function(f) { e <- new.env(); load(f, e); r <- e$res
    data.frame(tau = r$tau_type, cfg = r$cfg_id, rep = r$rep_id, n = r$n, p = r$p, sigma = r$sigma,
               screen = r$screen, n_sel = r$n_sel, n_feat = r$n_feat, e_input = r$e_input,
               mu_batch = r$mu_batch, mu_epoch = r$mu_epoch, mu_pat = r$mu_pat,
               anal_u = r$p_score_u, anal_r = r$p_score_r, perm_u = r$p_perm_u, perm_r = r$p_perm_r,
               sign_u = r$p_davies_u, sign_r = r$p_davies_r, permstr_u = r$p_permstr_u, permstr_r = r$p_permstr_r,
               rmse_e = r$rmse_e, rmse_mu = r$rmse_mu, rmse_ms = r$rmse_ms, secs = r$secs_total, stringsAsFactors = FALSE) }))
}
rate <- function(x) mean(x < alpha, na.rm = TRUE)
long <- NULL
for (rho in c("0", "0.3")) { d <- read_dirs(args[if (rho == "0") 1 else 2]); if (is.null(d)) next
  d <- d[!duplicated(d[, c("tau", "cfg", "rep")]), ]
  cat(sprintf("\nrho = %s: %d replications | settings: screen %s, e_input %s, mu %s/%s/%s\n", rho, nrow(d),
              paste(unique(d$screen), collapse = "/"), paste(unique(d$e_input), collapse = "/"),
              paste(unique(d$mu_batch), collapse = "/"), paste(unique(d$mu_epoch), collapse = "/"), paste(unique(d$mu_pat), collapse = "/")))
  key <- d[, c("tau", "n", "p", "sigma")]
  a <- aggregate(d[, c("anal_u", "anal_r", "perm_u", "perm_r", "sign_u", "sign_r", "permstr_u", "permstr_r")], by = key, FUN = rate)
  m <- aggregate(d[, c("rmse_e", "rmse_mu", "rmse_ms", "n_sel", "n_feat", "secs")], by = key, FUN = mean)
  a$reps <- aggregate(d$anal_u, by = key, FUN = length)$x
  a <- merge(a, m, by = c("tau", "n", "p", "sigma")); a$rho <- as.numeric(rho); long <- rbind(long, a) }
long <- long[order(long$tau, long$rho, long$sigma, long$p, long$n), ]
write.csv(long, "summary_v18_long.csv", row.names = FALSE)
## Table 3 layout: rows tau x estimator x (n, p, sigma), columns Anal./Perm. under rho 0 and 0.3
tab <- NULL
for (tau in c("tau0", "tau3")) for (est in c("u", "r")) {
  s <- long[long$tau == tau, ]; if (!nrow(s)) next
  w0 <- s[s$rho == 0, ]; w3 <- s[s$rho == 0.3, ]
  cells <- unique(s[, c("n", "p", "sigma")]); cells <- cells[order(cells$sigma, cells$p, cells$n), ]
  pick <- function(w, col) { i <- match(paste(cells$n, cells$p, cells$sigma), paste(w$n, w$p, w$sigma)); w[[col]][i] }
  tab <- rbind(tab, data.frame(tau = tau, estimator = if (est == "u") "Unrevised" else "Revised", cells,
                               anal_rho0 = pick(w0, paste0("anal_", est)), perm_rho0 = pick(w0, paste0("perm_", est)),
                               anal_rho03 = pick(w3, paste0("anal_", est)), perm_rho03 = pick(w3, paste0("perm_", est)),
                               reps_rho0 = pick(w0, "reps"), reps_rho03 = pick(w3, "reps"))) }
cat(sprintf("\nTable 3 (rejection rate at alpha = %.2f)\n", alpha)); print(tab, digits = 3, row.names = FALSE)
write.csv(tab, "summary_v18_table3.csv", row.names = FALSE)
