# cm_aggregate_l1.R -- rejection rates (alpha = 0.05) of the tau = 3, sigma = 1 cells rerun with l1 = 1e-3,
# next to the original runs (l1 = 1e-5) on the same seeds. Usage: Rscript cm_aggregate_l1.R
base <- "/nas/longleaf/home/shuaiy/project/corrmix_design/out"
rd <- function(dir, cfg) { f <- list.files(file.path(base, dir), pattern = sprintf("^test_tau3_c%d_r[0-9]+\\.RData$", cfg), full.names = TRUE)
  if (!length(f)) return(NULL); do.call(rbind, lapply(f, function(x) { e <- new.env(); load(x, e); r <- e$res; r[, c("cfg_id","rep_id","n","p","sigma","p_davies_u","p_davies_r","p_perm_u","p_perm_r")] })) }
out <- NULL
for (rho in c("00", "03")) for (cfg in 1:4) {
  new <- rd(paste0("test_l1e3_rho", rho), cfg); old <- rd(if (rho == "00") "test_rho00" else "test", cfg)
  rate <- function(d, v) if (is.null(d)) NA else mean(d[[v]] < 0.05, na.rm = TRUE)
  row <- data.frame(rho = ifelse(rho == "00", 0, 0.3), cfg = cfg, n = c(1000,2000,1000,2000)[cfg], p = c(20,20,40,40)[cfg], reps_new = if (is.null(new)) 0 else nrow(new),
    unrev_kernel_new = rate(new, "p_davies_u"), unrev_perm_new = rate(new, "p_perm_u"), rev_kernel_new = rate(new, "p_davies_r"), rev_perm_new = rate(new, "p_perm_r"),
    reps_old = if (is.null(old)) 0 else nrow(old),
    unrev_kernel_old = rate(old, "p_davies_u"), unrev_perm_old = rate(old, "p_perm_u"), rev_kernel_old = rate(old, "p_davies_r"), rev_perm_old = rate(old, "p_perm_r"))
  out <- rbind(out, row) }
num <- sapply(out, is.numeric); out[num] <- lapply(out[num], function(v) round(v, 3))
print(out, row.names = FALSE); write.csv(out, "summary_test_l1e3.csv", row.names = FALSE)
