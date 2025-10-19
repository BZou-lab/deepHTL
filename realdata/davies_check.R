library(MASS); library(CompQuadForm); library(caret)

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

setwd("/nas/longleaf/home/shuaiy/project/davies_test/real_data")
load("revise.RData")

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

Q_obs <- as.numeric(t(s) %*% Kp %*% s)

set.seed(1)
B <- 20000
Q_perm <- replicate(B, {
  s_perm <- sample(s, replace = FALSE)
  as.numeric(t(s_perm) %*% Kp %*% s_perm)
})
p_perm <- mean(Q_perm >= Q_obs)
se_perm <- sqrt(p_perm * (1 - p_perm) / B)

S1 <- sum(eig)
S2 <- sum(eig^2)
k_eff <- (S1^2) / S2
c_scale <- S2 / S1

set.seed(2)
Bmix <- 50000L
Q_mix <- replicate(Bmix, sum(eig * rchisq(length(eig), df = 1)))

probs <- ppoints(Bmix)
Q_theo <- c_scale * qchisq(probs, df = k_eff)

save(Q_perm, p_perm, se_perm, Q_mix, Q_theo,
     Q_obs, p_davies, k_eff, c_scale, S1, S2,
     file = "davies_check.RData")
