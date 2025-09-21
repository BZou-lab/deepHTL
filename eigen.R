args <- commandArgs(trailingOnly = TRUE)
sim_id <- as.numeric(args[1])

library(deepTL); library(MASS); library(CompQuadForm); library(caret)

revised_dnn_null <- function(object, ctrl = NULL) {
  if (!is.factor(object@z))
    object@z <- factor(ifelse(object@z==1, "A", "B"), levels = c("A","B"))
  x <- object@x; y <- object@y; z_nmrc <- as.numeric(object@z=="A")
  n <- NROW(x)
  
  if (is.null(ctrl)) ctrl <- list(
    n.ensemble = 100, verbose = FALSE,
    esCtrl = list(
      n.hidden = 10:5*2, n.batch = 100, n.epoch = 250,
      norm.x = TRUE, norm.y = TRUE,
      activate = "relu", accel = "rcpp",
      l1.reg = 1e-4, plot = FALSE,
      learning.rate.adaptive = "adam",
      early.stop.det = 100
    )
  )
  
  z_obj <- importDnnet(x = x, y = object@z)
  z_mod <- do.call("ensemble_dnnet", c(list(object = z_obj), ctrl))
  e_hat <- predict(z_mod, x)[, "A"]
  e_hat <- pmin(pmax(e_hat, 0.01), 1 - 0.01)
  
  y_obj <- importDnnet(x = x, y = y)
  y_mod <- do.call("ensemble_dnnet", c(list(object = y_obj), ctrl))
  mu_hat <- predict(y_mod, x)
  
  Ztilde <- z_nmrc - e_hat
  w_i <- Ztilde^2
  Ytilde <- (y - mu_hat) / Ztilde
  tau0 <- sum(w_i * Ytilde) / sum(w_i)
  
  Ystar <- y - tau0 * z_nmrc
  ystar_obj <- importDnnet(x = x, y = Ystar)
  ystar_mod <- do.call("ensemble_dnnet", c(list(object = ystar_obj), ctrl))
  mu_star_hat <- as.numeric(predict(ystar_mod, x))
  
  tildeYstar <- (Ystar - mu_star_hat) / Ztilde
  tau_star <- sum(w_i * tildeYstar) / sum(w_i)
  tau_rev <- tau0 + tau_star   
  
  r <- tildeYstar - tau_star
  eps_star <- Ystar - mu_star_hat - tau_star * Ztilde
  sig2_hat <- mean(eps_star^2)
  v_hat <- sig2_hat / w_i               
  std_r <- r / sqrt(v_hat)
  
  P <- diag(n) - tcrossprod(sqrt(w_i)) / sum(w_i)
  
  list(
    r = r,
    std_r = std_r,
    sig2_hat = sig2_hat,
    v_hat = v_hat,
    w = w_i,
    P = P,
    e_hat = e_hat,
    tau0 = tau0,                
    tau_star = tau_star,      
    tau_rev = tau_rev,  
    mu_hat = mu_hat,
    mu_star_hat = mu_star_hat
  )
}

n <- 2000
d <- 20
sigma_x <- 3

set.seed(10 + sim_id)
X <- mvrnorm(n, mu = rep(0, d), Sigma = diag(d))
b <- log(abs(X[, 1]) + 1) - X[, 2]^2 + sin(X[, 3]) + 0.5 * X[, 4] * X[, 5]
tau <- 5
eps <- rnorm(n, 0, sigma_x)
eX <- plogis(0.3 * X[, 1]^2 - 0.2 * sin(X[, 2]) + 0.3 * X[, 3] * X[, 4] - 0.1 * X[, 5]^2)
Z <- rbinom(n, 1, eX)
Y <- b + Z * tau + eps
object <- importTrt(X, Y, Z)

fit <- revised_dnn_null(object)

Dmat <- as.matrix(dist(X))
band <- median(Dmat[upper.tri(Dmat, diag = FALSE)])
K <- exp(-(Dmat^2) / (2 * band^2))

P <- fit$P            
Kp <- P %*% K %*% P
Kp <- 0.5 * (Kp + t(Kp)) 
s <- fit$std_r
Q_raw <- as.numeric(t(s) %*% K %*% s)
Q_proj <- as.numeric(t(s) %*% Kp %*% s)
eig_raw <- eigen(K, symmetric = TRUE, only.values = TRUE)$values
eig_proj <- eigen(Kp, symmetric = TRUE, only.values = TRUE)$values

x <- log10(eig_raw[which(eig_raw > 0)])[1:100]
y <- log10(eig_proj[which(eig_proj > 0)])[1:100]
out_dir <- "/nas/longleaf/home/shuaiy/project/davis_test/tau5"

png(file.path(out_dir, sprintf("eigen%03d.png", sim_id)), width = 900, height = 600)
plot(y, x)
abline(0, 1, lty = 2)
dev.off()  

out_path <- sprintf("/nas/longleaf/home/shuaiy/project/davis_test/tau5/proj%d.RData", sim_id)
save(fit = fit, file = out_path)


