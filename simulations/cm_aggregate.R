# Rscript cm_aggregate.R -- summarise out/{test,varsel,est} of the correlated-mixed grid
source("cm_dgp.R")
rd <- function(f, nm) { e <- new.env(); load(f, envir = e); get(nm, envir = e) }
## tests: rejection at alpha = .05 per cell x tau_type
fs <- list.files("out/test", "^test_.*\\.RData$", full.names = TRUE)
if (length(fs)) { tst <- do.call(rbind, lapply(fs, rd, "res"))
  s <- aggregate(cbind(davies_unrev = p_davies_u <= .05, davies_rev = p_davies_r <= .05,
                       perm_unrev = p_perm_u <= .05, perm_rev = p_perm_r <= .05) ~ tau_type + cfg_id + n + p + sigma, tst, mean)
  s$reps <- aggregate(rep_id ~ tau_type + cfg_id, tst, length)$rep_id
  s <- s[order(s$tau_type, s$cfg_id), ]; print(s, digits = 3, row.names = FALSE); write.csv(s, "summary_test.csv", row.names = FALSE) }
## screen: out/varsel (rho 0.3, the design's default) plus out/varsel_rho50 / _rho80
vdirs <- c(varsel = "0.3", varsel_rho50 = "0.5", varsel_rho80 = "0.8")
vs <- do.call(rbind, lapply(names(vdirs), function(d) {
  fs <- list.files(file.path("out", d), "^varsel_.*\\.RData$", full.names = TRUE)
  if (!length(fs)) return(NULL)
  v <- do.call(rbind, lapply(fs, rd, "varsel")); v$rho <- vdirs[[d]]; v }))
if (!is.null(vs)) {
  s <- aggregate(cbind(rej_marg = p_marg <= .05, rej_cond = p_cond <= .05) ~ rho + cfg_id + variable + type + group, vs, mean)
  s <- s[order(s$rho, s$cfg_id, as.integer(sub("X", "", s$variable))), ]
  nr <- aggregate(rep_id ~ rho + cfg_id + variable, vs, length)
  s$reps <- nr$rep_id[match(paste(s$rho, s$cfg_id, s$variable), paste(nr$rho, nr$cfg_id, nr$variable))]
  print(s, digits = 3, row.names = FALSE); write.csv(s, "summary_varsel.csv", row.names = FALSE)
  print(aggregate(cbind(rej_marg = p_marg <= .05, rej_cond = p_cond <= .05) ~ rho + cfg_id + group, vs, mean), digits = 3, row.names = FALSE) }
## estimation: out/est (mixed design) and out/est_gauss (paper's Gaussian design),
## each optionally joined with the R-learner base-learner variants in *_rl dirs
for (des in c("est", "est_gauss")) {
  fs <- list.files(file.path("out", des), "^est_.*\\.RData$", full.names = TRUE)
  if (!length(fs)) next
  est <- do.call(rbind, lapply(fs, rd, "df_est"))
  for (sub in c("_rl", "_xgb")) {                       # Lasso/KRR and XGBoost variant files
    frl <- list.files(file.path("out", paste0(des, sub)), "^est_rl_.*\\.RData$", full.names = TRUE)
    if (length(frl)) est <- rbind(est, do.call(rbind, lapply(frl, rd, "df_est_rl"))) }
  s <- aggregate(logmse ~ cfg_id + n + p + sigma + method, est, function(v) c(mean = mean(v), sd = sd(v), reps = length(v)))
  s <- data.frame(s[, 1:5], s$logmse); s <- s[order(s$cfg_id, s$mean), ]
  cat("\n==", des, "==\n"); print(s, digits = 3, row.names = FALSE); write.csv(s, sprintf("summary_%s.csv", des), row.names = FALSE) }
cat("DONE_CM_AGGREGATE\n")
