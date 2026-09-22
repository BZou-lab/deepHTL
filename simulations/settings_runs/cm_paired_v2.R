# Paired interim comparison: new tuning round vs original runs on the SAME replicate ids (same seeds), revised tests only plus agreement.
base <- "/nas/longleaf/home/shuaiy/project/corrmix_design/out"
rd <- function(dir, tau, cfg) { f <- list.files(file.path(base, dir), pattern = sprintf("^test_%s_c%d_r[0-9]+\\.RData$", tau, cfg), full.names = TRUE)
  if (!length(f)) return(NULL); do.call(rbind, lapply(f, function(x) { e <- new.env(); load(x, e); e$res[, c("rep_id", "p_davies_r", "p_perm_r", "p_davies_u", "p_perm_u")] })) }
cat(sprintf("%-4s %-5s %5s %3s %3s %5s | %-15s | %-15s | %s\n", "rho", "tau", "n", "p", "sig", "reps", "kernel rev new/old", "perm rev new/old", "cor(new,old) of kernel p"))
for (rho in c("00", "03")) for (tau in c("tau0", "tau3")) for (cfg in 1:8) {
  new <- rd(paste0("test_v2_rho", rho), tau, cfg); if (is.null(new) || nrow(new) < 50) next
  old <- rd(if (rho == "00") "test_rho00" else "test", tau, cfg); m <- merge(new, old, by = "rep_id", suffixes = c(".n", ".o"))
  cat(sprintf("%-4s %-5s %5d %3d %3d %5d |   %.3f / %.3f   |   %.3f / %.3f   | %.2f\n", ifelse(rho == "00", "0", "0.3"), tau, rep(c(1000, 2000), 4)[cfg], rep(c(20, 20, 40, 40), 2)[cfg], rep(c(1, 3), each = 4)[cfg], nrow(m),
    mean(m$p_davies_r.n < .05), mean(m$p_davies_r.o < .05), mean(m$p_perm_r.n < .05), mean(m$p_perm_r.o < .05), cor(m$p_davies_r.n, m$p_davies_r.o))) }
