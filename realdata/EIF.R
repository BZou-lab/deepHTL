library(deepTL); library(MASS);library(glmnet)
source("dnn")

setwd("/nas/longleaf/home/shuaiy/project/htdnn")
load("TLD_Selected.RData")
keep <- with(TLD, trimws(as.character(DIABETES_DON)) %in% c("0","1") &
               trimws(as.character(CMV_DON)) %in% c("0","1"))
TLD <- TLD[keep, , drop = FALSE]

confounders <- setdiff(names(TLD), c("TYPE", "FEV1_follow"))
z <- as.numeric(TLD$TYPE)
y <- TLD$FEV1_follow
x <- TLD[, confounders]
x[,which(colnames(x) == "GENDER")] <- ifelse(x[,which(colnames(x) == "GENDER")] == "M", 1, 0)
x[,which(colnames(x) == "GENDER_DON")] <- ifelse(x[,which(colnames(x) == "GENDER_DON")] == "M", 1, 0)
x[,which(colnames(x) == "DIABETES_DON")] <- ifelse(x[,which(colnames(x) == "DIABETES_DON")] == "1", 1, 0)
x[,which(colnames(x) == "CMV_DON")] <- ifelse(x[,which(colnames(x) == "CMV_DON")] == " 1", 1, 0)
x <- as.matrix(x)
x <- apply(x, 2, function(col) as.numeric(as.character(col)))
cc <- complete.cases(x,y,z)
x <- x[cc, , drop=FALSE]
y <- y[cc]
z <- z[cc]
n <- nrow(x)

en_dnn_ctrl <- list(
  n.ensemble = 20, verbose = TRUE,  
  esCtrl = list(
    n.hidden = c(256, 256, 128, 64),
    n.batch = 512,             
    n.epoch = 400,               
    norm.x = TRUE, norm.y = TRUE,
    activate = "relu", accel = "rcpp",
    l1.reg = 5e-5,              
    plot = FALSE,
    learning.rate.adaptive = "adam",
    early.stop.det = 30       
  )
)

object <- importTrt(x, y, z)
fit <- weight_dnn(object, en_dnn_ctrl) 
tau_hat <- predict.weight_dnn(fit, x, which = "revised")

K <- 5
fold_id <- sample(rep(1:K, length.out = n))
ehat <- numeric(n)
m1hat <- numeric(n)
m0hat <- numeric(n)

.dnn_fit_pred <- function(x_tr, y_tr, x_te, ctrl = en_dnn_ctrl) {
  obj <- importDnnet(x = x_tr, y = y_tr)
  mod <- do.call("ensemble_dnnet", c(list(object = obj), ctrl))
  as.numeric(predict(mod, x_te))
}

for (k in 1:K) {
  idx_tr <- fold_id != k
  idx_te <- !idx_tr
  
  z_fac_tr <- factor(ifelse(z[idx_tr] == 1, "A", "B"), levels = c("A","B"))
  obj_e <- importDnnet(x = x[idx_tr, , drop=FALSE], y = z_fac_tr)
  mod_e <- do.call("ensemble_dnnet", c(list(object = obj_e), en_dnn_ctrl))
  pz_te <- predict(mod_e, x[idx_te, , drop=FALSE])
  ehat[idx_te] <- if (is.null(dim(pz_te))) as.numeric(pz_te) else as.numeric(pz_te[, "A"])
  
  tr1 <- idx_tr & z == 1
  obj_m1 <- importDnnet(x = x[tr1, , drop=FALSE], y = y[tr1])
  mod_m1 <- do.call("ensemble_dnnet", c(list(object = obj_m1), en_dnn_ctrl))
  m1hat[idx_te] <- as.numeric(predict(mod_m1, x[idx_te, , drop=FALSE]))
  
  tr0 <- idx_tr & z == 0
  obj_m0 <- importDnnet(x = x[tr0, , drop=FALSE], y = y[tr0])
  mod_m0 <- do.call("ensemble_dnnet", c(list(object = obj_m0), en_dnn_ctrl))
  m0hat[idx_te] <- as.numeric(predict(mod_m0, x[idx_te, , drop=FALSE]))
}

Delta_hat <- mean(m1hat - m0hat)
phi_ATE <- (m1hat - m0hat) + z/ehat * (y - m1hat) -
  (1 - z)/(1 - ehat) * (y - m0hat) - Delta_hat
se_ATE <- sd(phi_ATE) / sqrt(n)
ci_ATE <- c(Delta_hat - 1.96 * se_ATE, Delta_hat + 1.96 * se_ATE)

LAS <- x[, "CALC_LAS_LISTDATE"]
las_cat <- cut(LAS,
               breaks = c(-Inf, 35, 45, Inf),
               labels = c("<35", "35–45", ">45"), right = TRUE, include.lowest = TRUE)

isch_hr <- x[, "ISCHTIME"]                       
isch_cat <- cut(isch_hr,
                breaks = c(-Inf, 4, 6, Inf),
                labels = c("<4h", "4–6h", ">6h"),
                right = TRUE, include.lowest = TRUE)

walk <- x[, "SIX_MIN_WALK"]
walk_cat <- cut(walk,
                breaks = c(-Inf, 250, 350, Inf),
                labels = c("<250 m", "250–350 m", ">350 m"),
                right = TRUE, include.lowest = TRUE)

bmi <- x[, "BMI_CALC"]
bmi_cat <- cut(bmi,
               breaks = c(-Inf, 18.5, 25, 30, Inf),
               labels = c("<18.5", "18.5–24.9", "25–29.9", "≥30"),
               right = FALSE, include.lowest = TRUE)

age <- x[, "AGE"]
age_cat <- cut(age,
               breaks = c(-Inf, 50, 65, 70, 75, Inf),
               labels = c("<50", "50–64", "65–69", "70–74", "≥75"),
               right = FALSE, include.lowest = TRUE)

age_don <- x[, "AGE_DON"]
age_don_cat <- cut(age_don,
                   breaks = quantile(age_don, probs = seq(0, 1, 0.25), na.rm = TRUE),
                   include.lowest = TRUE,
                   labels = paste0("Q", 1:4))

cmv_don <- factor(x[, "CMV_DON"], levels = c(0,1), labels = c("Donor CMV–", "Donor CMV+"))

gate_eif <- function(group_factor, m1hat, m0hat, ehat, y, z, label = deparse(substitute(group_factor))) {
  g <- as.integer(!is.na(group_factor))
  if (!all(g == 1)) stop("group_factor has NAs; clean before using.")
  levs <- levels(group_factor)
  out <- lapply(levs, function(L) {
    idx <- as.integer(group_factor == L)
    p_hat <- mean(idx)
    Delta_S_hat <- mean( idx * ((m1hat - m0hat) + z/ehat*(y - m1hat) - (1 - z)/(1 - ehat)*(y - m0hat)) ) / p_hat
    phi_S <- (idx / p_hat) * ( (m1hat - m0hat) + z/ehat*(y - m1hat) - (1 - z)/(1 - ehat)*(y - m0hat) - Delta_S_hat )
    se_S <- sd(phi_S) / sqrt(length(y))
    ci_S <- c(Delta_S_hat - 1.96*se_S, Delta_S_hat + 1.96*se_S)
    c(group = L, n = sum(idx), gate = Delta_S_hat, se = se_S, lo = ci_S[1], hi = ci_S[2])
  })
  do.call(rbind, out)
}

gate_las <- gate_eif(las_cat, m1hat, m0hat, ehat, y, z, "LAS")
gate_isch <- gate_eif(isch_cat, m1hat, m0hat, ehat, y, z, "Ischemic time")
gate_walk <- gate_eif(walk_cat, m1hat, m0hat, ehat, y, z, "6MWD")
gate_bmi <- gate_eif(bmi_cat, m1hat, m0hat, ehat, y, z, "BMI")
gate_age <- gate_eif(age_cat, m1hat, m0hat, ehat, y, z, "Recipient age")
gate_age_don <- gate_eif(age_don_cat, m1hat, m0hat, ehat, y, z, "Donor age")
gate_cmv_don <- gate_eif(cmv_don, m1hat, m0hat, ehat, y, z, "Donor CMV")

res_dnn_oof <- list(
  ATE = list(est = Delta_hat, se = se_ATE, ci = ci_ATE, n = n, K = K),
  GATE = list(LAS = gate_las, Ischemic = gate_isch, Walk = gate_walk,
              BMI = gate_bmi, Age = gate_age, DonorAge = gate_age_don, DonorCMV = gate_cmv_don),
  nuisances = list(e_hat_oof = ehat, m1hat_oof = m1hat, m0hat_oof = m0hat, folds = fold_id),
  tau = list(revised = tau_hat),
  meta = list(time = as.character(Sys.time()), ctrl = en_dnn_ctrl)
)

saveRDS(res_dnn_oof, "unos_hte_eif_dnn_oof.rds")
