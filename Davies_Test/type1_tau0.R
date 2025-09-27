args <- commandArgs(trailingOnly = TRUE)
sim_id <- as.numeric(args[1])

library(deepTL); library(rlearner); library(MASS); library(CompQuadForm); library(caret); library(glmnet)

deepTL_null <- function(object, ctrl = NULL) {
  if (!is.factor(object@z))
    object@z <- factor(ifelse(object@z == 1, "A", "B"), levels = c("A","B"))
  
  x <- object@x
  y <- object@y
  z_num <- as.numeric(object@z == "A")
  n <- NROW(x)
  
  if (is.null(ctrl)) ctrl <- list(
    n.ensemble = 100, verbose = FALSE,
    esCtrl = list(
      n.hidden = 10:5*2, n.batch = 100, n.epoch = 200,
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
  e_hat <- pmin(pmax(e_hat, 1e-2), 1 - 1e-2) 
  
  y_obj <- importDnnet(x = x, y = y)
  y_mod <- do.call("ensemble_dnnet", c(list(object = y_obj), ctrl))
  mu_hat <- as.numeric(predict(y_mod, x))
  
  w <- (z_num - e_hat)^2
  
  Ytilde <- (y - mu_hat) / (z_num - e_hat)
  tau0 <- tau_semi <- sum((y - mu_hat) * (z_num - e_hat)) / sum(w)
  
  r_semi <- Ytilde - tau_semi
  eps_semi <- (y - mu_hat) - tau_semi * (z_num - e_hat)
  sig2_semi <- mean(w * r_semi^2)
  v_semi <- sig2_semi / w
  std_r_semi <- r_semi / sqrt(v_semi)
  
  beta1 <- coef(lm(y ~ z_num + e_hat))[2]
  lambdas <- seq(0, 1, length.out = 21)
  best <- list(score = Inf, lam = NA)
  fold <- sample(rep(1:2, length.out = length(z_num)))
  
  for (lam in lambdas) {
    c_lam <- lam * tau0 + (1 - lam) * beta1
    score <- 0
    for (k in 1:2) {
      idx_tr <- fold != k; idx_te <- fold == k
      ystar_tr <- y[idx_tr] - c_lam * z_num[idx_tr]
      xi_mod <- glmnet::cv.glmnet(as.matrix(x[idx_tr, , drop = FALSE]), ystar_tr, alpha = 1)
      xi_hat_te <- as.numeric(predict(xi_mod, as.matrix(x[idx_te, , drop = FALSE]), s = "lambda.min"))
      zr_te <- z_num[idx_te] - e_hat[idx_te]
      lab_te <- (y[idx_te] - c_lam * z_num[idx_te] - xi_hat_te) / zr_te
      score <- score + var(lab_te, na.rm = TRUE)
    }
    if (score < best$score) best <- list(score = score, lam = lam)
  }
  tau0 <- best$lam * tau0 + (1 - best$lam) * beta1
  
  Ystar <- y - tau0 * z_num
  ystar_obj <- importDnnet(x = x, y = Ystar)
  ystar_mod <- do.call("ensemble_dnnet", c(list(object = ystar_obj), ctrl))
  mu_star_hat <- as.numeric(predict(ystar_mod, x))
  
  tildeYstar <- (Ystar - mu_star_hat) / (z_num - e_hat)
  tau_star <- sum(w * tildeYstar) / sum(w)
  tau_rev <- tau0 + tau_star
  
  r_rev <- tildeYstar - tau_star
  eps_rev <- Ystar - mu_star_hat - tau_star * (z_num - e_hat)
  sig2_rev <- mean(w * r_rev^2)
  v_rev <- sig2_rev / w
  std_r_rev <- r_rev / sqrt(v_rev)
  
  P <- diag(n) - tcrossprod(sqrt(w)) / sum(w)
  
  list(
    w = w,
    P = P,
    tau0 = tau0,
    tau_semi = tau_semi,
    tau_rev = tau_rev,
    std_r_semi = std_r_semi,
    sig2_semi = sig2_semi,
    v_semi = v_semi,
    std_r_rev = std_r_rev,
    sig2_rev = sig2_rev,
    v_rev = v_rev, 
    lam = best$lam
  )
}

proj_pvals <- function(s, K, P, B = 2000) {
  Kp <- P %*% K %*% P
  Kp <- 0.5 * (Kp + t(Kp))         
  eig <- eigen(Kp, symmetric = TRUE, only.values = TRUE)$values
  eig <- eig[eig > 1e-8]
  Q_obs <- as.numeric(t(s) %*% Kp %*% s) 
  p_davies <- if (length(eig)) {
    p <- tryCatch(CompQuadForm::davies(Q_obs, lambda = eig)$Qq, error = function(e) NA_real_)
    if (!is.finite(p)) tryCatch(CompQuadForm::liu(Q_obs, lambda = eig), error = function(e2) NA_real_) else p
  } else NA_real_
  
  Q_perm <- replicate(B, {
      s_perm <- sample(s, replace = FALSE)
      as.numeric(t(s_perm) %*% Kp %*% s_perm)
    })
  p_perm <- (sum(Q_perm >= Q_obs) + 1) / (B + 1)
    
  return(c(p_davies, p_perm))
}

n_vec <- c(1000,2000)
d_vec <- c(20,40)
sigma_vec <- c(1,3)

h0_test <- list()
res_idx <- 1

for (sigma in sigma_vec) {
    for (n in n_vec) {
      for (d in d_vec) {
        set.seed(10 * sim_id + n + d + sigma)
        X <- mvrnorm(n, mu = rep(0, d), Sigma = diag(d))
        b <- log(abs(X[, 1]) + 1) - X[, 2]^2 + sin(X[, 3]) + 0.5 * X[, 4] * X[, 5]
        e <- plogis(0.8 * sin(pi * X[,1] * X[,2]) + 0.6 * X[,3] * X[,4] + 0.5 * tanh(X[,5]))
        Z <- rbinom(n, 1, e)
        tau <- rep(0, n)
        eps <- rnorm(n, 0, sigma)
        Y <- b + (Z - 0.5) * tau + eps
        
        object <- importTrt(X, Y, Z)
        
        fit <- deepTL_null(object)
        
        Dmat <- as.matrix(dist(X))
        band <- median(Dmat[upper.tri(Dmat, diag = FALSE)])
        K <- exp(-(Dmat^2) / (2 * band^2))
        
        p_rev <- proj_pvals(fit$std_r_rev, K, fit$P)
        p_semi <- proj_pvals(fit$std_r_semi, K, fit$P)
        
        h0_test[[res_idx]] <- data.frame(
          n = n,
          d = d,
          sigma = sigma,
          p_davies_rev = p_rev[1], 
          p_perm_rev = p_rev[2],
          p_davies_semi = p_semi[1], 
          p_perm_semi = p_semi[2],
          tau_rev = fit$tau_rev,
          tau_semi = fit$tau_semi,
          sig_rev = sqrt(fit$sig2_rev),
          sig_semi = sqrt(fit$sig2_semi)
        )
        res_idx <- res_idx + 1
      }
    }
}
    
h0_res <- do.call(rbind, h0_test)

out_path <- sprintf("/nas/longleaf/home/shuaiy/project/davies_test/H0/tau0_%d.RData", sim_id)
save(h0_res, file = out_path)