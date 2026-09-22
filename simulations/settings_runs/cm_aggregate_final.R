# cm_aggregate_final.R -- 2026-09-21. Rejection rates at alpha = 0.05 per cell:
#   A. cells 1,3,5-8: original settings, reps 1-500 (out/test_rho00, out/test), reps 501-1000 (out/test_ext_*), and all 1000
#   B. cells 2,4: original settings reps 1-500 vs tuned settings reps 1-500 (out/test_v2_*), same seeds
#   C. probe in cell 3 (n1000 p40 sigma1, rho 0): l1 = 1e-7, 1e-5 (original), 1e-3
base <- "/nas/longleaf/home/shuaiy/project/corrmix_design/out"; V <- c("p_davies_u", "p_perm_u", "p_davies_r", "p_perm_r")
rd <- function(dir, tau, cfg) { f <- list.files(file.path(base, dir), pattern = sprintf("^test_%s_c%d_r[0-9]+\\.RData$", tau, cfg), full.names = TRUE)
  if (!length(f)) return(NULL); do.call(rbind, lapply(f, function(x) { e <- new.env(); load(x, e); e$res[, c("rep_id", V)] })) }
rt <- function(d) if (is.null(d)) rep(NA, 4) else sapply(V, function(v) mean(d[[v]] < 0.05, na.rm = TRUE))
nn <- rep(c(1000, 2000), 4); pp <- rep(c(20, 20, 40, 40), 2); ss <- rep(c(1, 3), each = 4); out <- NULL
for (rho in c("00", "03")) for (tau in c("tau0", "tau3", "S2")) for (cfg in 1:8) {
  old <- rd(if (rho == "00") "test_rho00" else "test", tau, cfg)
  if (cfg %in% c(2, 4)) { new <- rd(paste0("test_v2_rho", rho), tau, cfg); sets <- list(`orig 1-500` = old, `tuned 1-500` = new)
  } else { ext <- rd(paste0("test_ext_rho", rho), tau, cfg); sets <- list(`orig 1-500` = old, `orig 501-1000` = ext, `orig 1-1000` = if (is.null(ext)) NULL else rbind(old, ext)) }
  for (s in names(sets)) { r <- rt(sets[[s]]); out <- rbind(out, data.frame(rho = ifelse(rho == "00", 0, 0.3), tau = tau, n = nn[cfg], p = pp[cfg], sigma = ss[cfg], set = s,
    reps = if (is.null(sets[[s]])) 0 else nrow(sets[[s]]), kernel_unrev = r[1], perm_unrev = r[2], kernel_rev = r[3], perm_rev = r[4])) } }
num <- sapply(out, is.numeric); out[num] <- lapply(out[num], function(v) round(v, 3)); rownames(out) <- NULL
write.csv(out, "summary_test_final_20260921.csv", row.names = FALSE)
cat("=== C. probe, cell n1000 p40 sigma1, rho 0 ===\n")
for (tau in c("tau0", "tau3")) for (L in c("1e-7", "1e-5", "1e-3")) { d <- if (L == "1e-5") rd("test_rho00", tau, 3) else if (L == "1e-3" && tau == "tau3") rd("test_l1e3_rho00", tau, 3) else rd(paste0("probe_c3_l", L), tau, 3)
  r <- rt(d); cat(sprintf("%s l1=%-5s reps=%3d | unrev kernel %.3f perm %.3f | rev kernel %.3f perm %.3f\n", tau, L, if (is.null(d)) 0 else nrow(d), r[1], r[2], r[3], r[4])) }
cat("\nsummary_test_final_20260921.csv written,", nrow(out), "rows\n")
