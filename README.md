# deepHTL

Deep Heterogeneous Treatment Learning: An efficient semiparametric framework for estimating and testing heterogeneous treatment effects (HTE). It integrates Robinson transformations with bias-correction steps using ensemble DNNs, XGBoost, Kernel Ridge Regression, and Lasso.

## Installation

You can install the development version of deepHTL from GitHub using the following commands:

``` r
devtools::install_github("shuaiy125/deepHTL")
```

## Simulation Setup

``` r
suppressMessages(library(deepHTL))
suppressMessages(library(deepTL))
library(MASS)

set.seed(2025)
n <- 2000
p <- 20
sigma <- 1

x <- matrix(rnorm(n * p), n, p)
bx <- log(abs(x[,1]) + 1) - x[,2]^2 + sin(x[,3]) + 0.5*x[,4]*x[,5]
ex <- plogis(0.8 * sin(pi * x[,1] * x[,2]) + 0.6 * x[,3] * x[,4] + 0.5 * tanh(x[,5]))
eps <- rnorm(n, 0, sigma)
z <- rbinom(n, 1, ex)
tx <- -1 + x[,1] * x[,2] + cos(x[, 3])^2 + max(x[,4] - x[,5], 0)
y <- bx + (z - 0.5) * tx + eps
obj_tr <- importTrt(x, y, z)

nt <- 2000
xt <- matrix(rnorm(nt * p), nt, p)
tt <- -1 + xt[,1] * xt[,2] + cos(xt[, 3])^2 + pmax(xt[,4] - xt[,5], 0)
```

## Hyper-parameters for DNN and ensemble

``` r
en_dnn_ctrl <- list(
    n.ensemble = 30, verbose = FALSE,
    esCtrl = list(
        n.hidden = c(128, 64, 32),
        n.batch = 256,
        n.epoch = 120,
        norm.x = TRUE, norm.y = TRUE,
        activate = "relu", accel = "rcpp",
        l1.reg = 1e-3,
        plot = FALSE,
        learning.rate.adaptive = "adam",
        early.stop.det = 20
      )
    )
```

## Estimating HTE using deepHTL

``` r
set.seed(4231)
fit_deepHTL <- weight_dnn(obj_tr, en_dnn_ctrl = en_dnn_ctrl)
tau_deepHTL <- predict(fit_deepHTL, xt, which = "both")

set.seed(4231)
fit_xgb <- weight_xgboost(obj_tr, k = 3)
tau_xgb <- predict(fit_xgb, xt, which = "both")

set.seed(4231)
fit_kern <- weight_kern(obj_tr, k_folds = 3)
tau_kern <- predict(fit_kern, xt, which = "both")

set.seed(4231)
fit_lasso <- weight_lasso(obj_tr)
tau_lasso <- predict(fit_lasso, xt, which = "both")

mse_dnn <- mean((tau_deepHTL - tt)^2)
mse_xgb <- mean((tau_xgb - tt)^2)
mse_kern <- mean((tau_kern - tt)^2)
mse_lasso <- mean((tau_lasso - tt)^2)

log_mse_results <- data.frame(
  Method = c("deepHTL (DNN)", "Weighted XGBoost", "Weighted Kernel", "Weighted Lasso"),
  MSE = c(mse_dnn, mse_xgb, mse_kern, mse_lasso),
  Log_MSE = log(c(mse_dnn, mse_xgb, mse_kern, mse_lasso))
)

print(log_mse_results)
```

## Testing HTE using deepHTL

``` r
n <- 2000
d <- 20
sigma <- 1
set.seed(4231)
X <- mvrnorm(n, mu = rep(0, d), Sigma = diag(d))
b <- log(abs(X[, 1]) + 1) - X[, 2]^2 + sin(X[, 3]) + 0.5 * X[, 4] * X[, 5]
e <- plogis(0.8 * sin(pi * X[,1] * X[,2]) + 0.6 * X[,3] * X[,4] + 0.5 * tanh(X[,5]))
Z <- rbinom(n, 1, e)
eps <- rnorm(n, 0, sigma)
Y <- b + (Z - 0.5) * 3 + eps ## Assumae tau = 3
object <- importTrt(X, Y, Z)

fit <- davies_test(object)
print(fit)
```

## References

Mi, X. et al. (2021). A deep learning semiparametric regression for adjusting complex confounding structures. The Annals of Applied Statistics, 15(3):1086–1100.

Nie, X. and Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2):299–319.