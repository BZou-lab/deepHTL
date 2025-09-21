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

n_vec <- c(2000, 4000)
d_vec <- c(20,40)
sigma_vec <- c(1,3)

h1_test <- list()
res_idx <- 1

for (sigma_x in sigma_vec) {
  for (n in n_vec) {
    for (d in d_vec) {
      set.seed(10 * sim_id + n + d + sigma_x)
      X <- mvrnorm(n, mu = rep(0, d), Sigma = diag(d))
      b <- log(abs(X[, 1]) + 1) - X[, 2]^2 + sin(X[, 3]) + 0.5 * X[, 4] * X[, 5]
      eX <- plogis(0.3 * X[, 1]^2 - 0.2 * sin(X[, 2]) + 0.3 * X[, 3] * X[, 4] - 0.1 * X[, 5]^2)
      tau <- 5 + rowSums(((-1)^(1:3 + 1)) * X[, 1:3])
      eps <- rnorm(n, 0, sigma_x)
      Z <- rbinom(n, 1, eX)
      Y <- b + (Z - eX) * tau + eps
      
      object <- importTrt(X, Y, Z)
      
      fit <- revised_dnn_null(object)
      
      Dmat <- as.matrix(dist(X))
      band <- median(Dmat[upper.tri(Dmat, diag = FALSE)])
      K <- exp(-(Dmat^2) / (2 * band^2))
      P <- fit$P            
      Kp <- P %*% K %*% P
      Kp <- 0.5 * (Kp + t(Kp)) 
      s <- fit$std_r
      Q_dav <- as.numeric(t(s) %*% Kp %*% s)
      eig <- eigen(Kp, symmetric = TRUE, only.values = TRUE)$values
      eig <- eig[eig > 1e-8]
      
      B <- 1000
      Q_perm <- replicate(B, {
        s_perm <- sample(s, replace = FALSE)
        as.numeric(t(s_perm) %*% Kp %*% s_perm)
      })
      p_perm <- (sum(Q_perm >= Q_dav) + 1) / (B + 1)
      
      p_davies <- if (length(eig)) {
        p <- tryCatch(CompQuadForm::davies(Q_dav, lambda = eig)$Qq,
                      error = function(e) NA_real_)
        if (!is.finite(p)) {
          tryCatch(CompQuadForm::liu(Q_dav, lambda = eig), error = function(e2) NA_real_)
        } else p
      } else NA_real_
      
      h1_test[[res_idx]] <- data.frame(
        n = n,
        d = d,
        sigma = sigma_x,
        p_perm = p_perm,
        p_davies = p_davies
      )
      res_idx <- res_idx + 1
    }
  }
}

h1_res <- do.call(rbind, h1_test)

out_path <- sprintf("/nas/longleaf/home/shuaiy/project/davis_test/H1/Linear%d.RData", sim_id)
save(h1_res, file = out_path)