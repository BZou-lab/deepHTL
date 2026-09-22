# cm_aggregate_v2.R -- rejection rates (alpha = 0.05) of the 2026-09-21 tuning round (out/test_v2_rho00, out/test_v2_rho03) next to the
# original runs (out/test_rho00, out/test) on the same seeds. tau0/tau3 rows are type I error, S2 rows are power.
base <- "/nas/longleaf/home/shuaiy/project/corrmix_design/out"
rd <- function(dir, tau, cfg) { f <- list.files(file.path(base, dir), pattern = sprintf("^test_%s_c%d_r[0-9]+\\.RData$", tau, cfg), full.names = TRUE)
  if (!length(f)) return(NULL); do.call(rbind, lapply(f, function(x) { e <- new.env(); load(x, e); r <- e$res
    for (v in c("p_score_u", "p_score_r")) if (is.null(r[[v]])) r[[v]] <- NA_real_
    r[, c("rep_id", "p_davies_u", "p_davies_r", "p_score_u", "p_score_r", "p_perm_u", "p_perm_r")] })) }
rate <- function(d, v) if (is.null(d) || all(is.na(d[[v]]))) NA else mean(d[[v]] < 0.05, na.rm = TRUE)
out <- NULL
for (rho in c("00", "03")) for (tau in c("tau0", "tau3", "S2")) for (cfg in 1:8) {
  new <- rd(paste0("test_v2_rho", rho), tau, cfg); old <- rd(if (rho == "00") "test_rho00" else "test", tau, cfg)
  out <- rbind(out, data.frame(rho = ifelse(rho == "00", 0, 0.3), tau = tau, cfg = cfg, n = rep(c(1000, 2000), 4)[cfg], p = rep(c(20, 20, 40, 40), 2)[cfg], sigma = rep(c(1, 3), each = 4)[cfg],
    reps_new = if (is.null(new)) 0 else nrow(new), reps_old = if (is.null(old)) 0 else nrow(old),
    kern_u_new = rate(new, "p_davies_u"), kern_u_old = rate(old, "p_davies_u"), kern_r_new = rate(new, "p_davies_r"), kern_r_old = rate(old, "p_davies_r"),
    score_u_new = rate(new, "p_score_u"), score_r_new = rate(new, "p_score_r"),
    perm_u_new = rate(new, "p_perm_u"), perm_u_old = rate(old, "p_perm_u"), perm_r_new = rate(new, "p_perm_r"), perm_r_old = rate(old, "p_perm_r"))) }
num <- sapply(out, is.numeric); out[num] <- lapply(out[num], function(v) round(v, 3))
write.csv(out, "summary_test_v2.csv", row.names = FALSE); print(out[out$reps_new > 0, ], row.names = FALSE)
