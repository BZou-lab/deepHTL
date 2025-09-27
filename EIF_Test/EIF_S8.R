args <- commandArgs(trailingOnly = TRUE)
sim_id <- as.numeric(args[1])

library(glmnet); library(deepTL); library(MASS); library(caret)
source("dnn.R")

# ==== Cross-fitted EIF test for H0: E[tau(X)] = 0 ====

ctrl <- list(
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

EIF_Test <- function(object, K = 5, which = c("revised","unrevised"), clip_e = 1e-3) {
  which <- match.arg(which)
  X <- object@x
  y <- object@y
  z <- if (is.factor(object@z)) {
    as.numeric(object@z == levels(object@z)[1])
  } else {
    as.numeric(object@z)
  }
  n <- nrow(X)
  
  # -- K folds for cross-fitting
  folds <- caret::createFolds(z, k = K, list = TRUE, returnTrain = FALSE)
  
  tau_hat <- numeric(n)
  e_hat <- numeric(n)
  m_hat <- numeric(n)
  
  for (k in seq_along(folds)) {
    te <- folds[[k]]
    tr <- setdiff(seq_len(n), te)
    
    obj_tr <- importTrt(X[tr, , drop = FALSE], y[tr], z[tr])
    fit <- weight_dnn(obj_tr, ctrl)
    
    tau_hat[te] <- as.numeric(predict(fit, X[te, , drop = FALSE], which = which))
    
    pz <- predict(fit$e_mod, X[te, , drop = FALSE])
    e_te <- if (is.null(dim(pz))) as.numeric(pz) else {
      as.numeric(pz[, "A"])
    }
    e_hat[te] <- pmin(pmax(e_te, clip_e), 1 - clip_e)
    
    my <- predict(fit$mu_mod, X[te, , drop = FALSE])
    m_hat[te] <- as.numeric(my)
  }
  
  phi <- tau_hat + ((z - e_hat) / (e_hat * (1 - e_hat))) * (y - m_hat - tau_hat * (z - e_hat))
  
  tau_bar <- mean(tau_hat)
  s2_phi <- mean((phi - mean(phi))^2 )  
  se <- sqrt(s2_phi / n)
  z_stat <- tau_bar / se
  pval <- 2 * (1 - pnorm(abs(z_stat)))
  
  list(
    tau_bar = tau_bar,       
    se = se,           
    z = z_stat,     
    p_value = pval,         
    phi = phi    
  )
}

n <- 2000
d <- 40
sigma <- 3

set.seed(10 * sim_id + n + d + sigma)
X <- mvrnorm(n, mu = rep(0, d), Sigma = diag(d))
b <- log(abs(X[, 1]) + 1) - X[, 2]^2 + sin(X[, 3]) + 0.5 * X[, 4] * X[, 5]
eX <- plogis(0.8 * sin(pi * X[,1] * X[,2]) + 0.6 * X[,3] * X[,4] + 0.5 * tanh(X[,5]))
eps <- rnorm(n, 0, sigma)
Z <- rbinom(n, 1, eX)
tau <- X[,1] + X[,2] + X[,3] 
Y <- b + (Z - 0.5) * tau + eps
object <- importTrt(X, Y, Z)

res_unrev <- EIF_Test(object, K = 5, which = "unrevised")
res_rev <- EIF_Test(object, K = 5, which = "revised")

res <- list(unrev = res_unrev, rev = res_rev)

out_path <- sprintf("/nas/longleaf/home/shuaiy/project/EIF_test/H0/s8/res%d.RData", sim_id)
save(res, file = out_path)