library(deepTL); library(MASS); library(CompQuadForm); library(caret)

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

object <- importTrt(x, y, z)

semi_dnn_null <- function(object, ctrl = NULL) {
  if (!is.factor(object@z))
   object@z <- factor(ifelse(object@z==1, "A","B"), levels=c("A","B"))
  x <- object@x; y <- object@y; z_nmrc <- as.numeric(object@z=="A")
  n <- NROW(x)
  
  if (is.null(ctrl)) ctrl <- list(
    n.ensemble=100, verbose=FALSE,
    esCtrl=list(n.hidden=10:5*2, n.batch=100, n.epoch=250,
                norm.x=TRUE, norm.y=TRUE, activate="relu",
                accel="rcpp", l1.reg=1e-4, plot=FALSE,
                learning.rate.adaptive="adam", early.stop.det=100)
  )
  
  z_obj <- importDnnet(x = x, y = object@z)
  z_mod <- do.call("ensemble_dnnet", c(list(object = z_obj), ctrl))
  e_hat <- predict(z_mod, x)[, "A"]
  e_hat <- pmin(pmax(e_hat, 0.01), 1 - 0.01)
  
  y_obj <- importDnnet(x=x, y=y)
  y_mod <- do.call("ensemble_dnnet", c(list(object=y_obj), ctrl))
  mu_hat <- predict(y_mod, x)
  
  Ztilde <- z_nmrc - e_hat
  w_i <- Ztilde^2
  Ytilde <- (y - mu_hat) / Ztilde
  tau0 <- sum(w_i * Ytilde) / sum(w_i)
  r <- Ytilde - tau0
  
  eps_hat <- y - mu_hat - tau0 * Ztilde
  sig2_hat <- mean(eps_hat^2)
  v_hat <- sig2_hat / w_i
  std_r <- r / sqrt(v_hat)
  
  P <- diag(n) - tcrossprod(sqrt(w_i)) / sum(w_i)
  
  list(r = r, std_r = std_r, sig2_hat = sig2_hat, w = w_i, P = P,
       v_hat = v_hat, e_hat = e_hat, mu_hat = mu_hat, tau0 = tau0)
}

fit <- semi_dnn_null(object) 

Dmat <- as.matrix(dist(x))
band <- median(Dmat[upper.tri(Dmat, diag = FALSE)])
K <- exp(-(Dmat^2) / (2 * band^2))
P <- fit$P            
Kp <- P %*% K %*% P
Kp <- 0.5 * (Kp + t(Kp)) 
s <- fit$std_r
Q_dav <- as.numeric(t(s) %*% Kp %*% s)
eig <- eigen(Kp, symmetric = TRUE, only.values = TRUE)$values
eig <- eig[eig > 1e-8]

Q_obs <- as.numeric(t(s) %*% K %*% s)
B <- 1000
Q_perm <- replicate(B, {
  s_perm <- sample(s, replace = FALSE)
  as.numeric(t(s_perm) %*% K %*% s_perm)
})
p_perm <- mean(Q_perm >= Q_obs)

p_davies <- if (length(eig)) {
  p <- tryCatch(CompQuadForm::davies(Q_dav, lambda = eig)$Qq,
                error = function(e) NA_real_)
  if (!is.finite(p)) {
    tryCatch(CompQuadForm::liu(Q_dav, lambda = eig), error = function(e2) NA_real_)
  } else p
} else NA_real_

out_path <- sprintf("/nas/longleaf/home/shuaiy/project/davies_test/real_data/semi.RData")
save(fit, Q_dav, Q_perm, p_perm, p_davies, file = out_path)
