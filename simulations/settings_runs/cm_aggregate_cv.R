# cm_aggregate_cv.R -- 2026-09-21. Rejection rates of the CV-tuned rerun (out/cv006_rho00, out/cv006_rho03) next to the
# original fixed-setting runs on the same seeds, plus the settings the cross-validation picked in each cell.
base <- "/nas/longleaf/home/shuaiy/project/corrmix_design/out"; V <- c("p_davies_u","p_perm_u","p_davies_r","p_perm_r")
rd <- function(dir, tau, cfg, tune = FALSE) { f <- list.files(file.path(base, dir), pattern = sprintf("^test_%s_c%d_r[0-9]+\\.RData$", tau, cfg), full.names = TRUE)
  if (!length(f)) return(NULL); cols <- c("rep_id", V, if (tune) c("n_batch","n_epoch","l1"))
  do.call(rbind, lapply(f, function(x) { e <- new.env(); load(x, e); e$res[, cols] })) }
old_dir <- function(rho) if (rho == "00") c("test_rho00","test_ext_rho00") else c("test","test_ext_rho03")
nn <- rep(c(1000,2000),4); pp <- rep(c(20,20,40,40),2); ss <- rep(c(1,3), each=4)
cells <- rbind(data.frame(rho="00", cfg=c(1,3,4,5,6,8)), data.frame(rho="03", cfg=1:8))
res <- NULL; tun <- NULL
for (i in seq_len(nrow(cells))) { rho <- cells$rho[i]; cfg <- cells$cfg[i]
  for (tau in c("tau0","tau3","S2")) {
    new <- rd(paste0("cv006_rho", rho), tau, cfg, TRUE); if (is.null(new)) next
    old <- do.call(rbind, lapply(old_dir(rho), function(d) rd(d, tau, cfg)))
    rt <- function(d) if (is.null(d)) rep(NA,4) else sapply(V, function(v) mean(d[[v]] < .05, na.rm = TRUE))
    a <- rt(new); b <- rt(old)
    res <- rbind(res, data.frame(rho = ifelse(rho=="00",0,0.3), tau = tau, n = nn[cfg], p = pp[cfg], sigma = ss[cfg], reps = nrow(new),
      Ku_cv = a[1], Ku_old = b[1], Pu_cv = a[2], Pu_old = b[2], Kr_cv = a[3], Kr_old = b[3], Pr_cv = a[4], Pr_old = b[4]))
    md <- function(v) { t <- table(new[[v]]); paste0(names(t)[which.max(t)], " (", round(100*max(t)/nrow(new)), "%)") }
    tun <- rbind(tun, data.frame(rho = ifelse(rho=="00",0,0.3), tau = tau, n = nn[cfg], p = pp[cfg], sigma = ss[cfg], reps = nrow(new),
      batch = md("n_batch"), epoch = md("n_epoch"), l1 = md("l1"))) } }
num <- sapply(res, is.numeric); res[num] <- lapply(res[num], function(v) round(v, 3))
write.csv(res, "summary_test_cv006.csv", row.names = FALSE); write.csv(tun, "summary_tuning_cv006.csv", row.names = FALSE)
cat("=== rejection rates, CV-tuned vs original (same seeds)\n"); print(res[res$tau != "S2", ], row.names = FALSE)
cat("\n=== power\n"); print(res[res$tau == "S2", ], row.names = FALSE)
cat("\n=== settings chosen by the cross-validation (mode across replicates)\n"); print(tun, row.names = FALSE)
