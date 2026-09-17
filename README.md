# deepHTL

Deep Heterogeneous Treatment Learning: An efficient semiparametric framework for estimating and testing heterogeneous treatment effects (HTE). It integrates Robinson transformations with bias-correction steps using ensemble Bagged-DNNs, XGBoost, Kernel Ridge Regression, and Lasso.

## Installation

You can install the development version of deepHTL from GitHub using the following commands:

``` r
devtools::install_github("BZou-lab/deepHTL")
```

## Simulation Setup

``` r
library(deepHTL)
library(MASS)

set.seed(2025)
n <- 1000
p <- 20
sigma <- 1

x <- matrix(rnorm(n * p), n, p)
fx <- log(abs(x[,1]) + 1) - x[,2]^2 + sin(x[,3]) + 0.5*x[,4]*x[,5]
ex <- plogis(0.8 * sin(pi * x[,1] * x[,2]) + 0.6 * x[,3] * x[,4] + 0.5 * tanh(x[,5]))
eps <- rnorm(n, 0, sigma)
z <- rbinom(n, 1, ex)
tx <- -1 + x[,1] * x[,2] + cos(x[, 3])^2 + max(x[,4] - x[,5], 0)
y <- fx + z * tx + eps
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
## Select the L1 penalty by cross-validating the R-loss
set.seed(4231)
nuis <- weight_dnn(obj_tr, en_dnn_ctrl = en_dnn_ctrl)$nuisance   # cross-fitted e_hat and mu_hat
K <- 3
folds <- sample(rep(seq_len(K), length.out = n))
cv_rloss_dnn <- function(l1) {
  ctrl <- en_dnn_ctrl
  ctrl$esCtrl$l1.reg <- l1
  loss <- 0
  for (k in seq_len(K)) {
    tr <- folds != k
    te <- folds == k
    fit <- weight_dnn(importTrt(x[tr, ], y[tr], z[tr]), en_dnn_ctrl = ctrl)
    tau_te <- predict(fit, x[te, ], which = "revised")
    loss <- loss + sum((y[te] - nuis$mu_hat[te] - tau_te * (z[te] - nuis$e_hat[te]))^2)
  }
  loss / n
}
l1_grid <- c(1e-5, 1e-4, 1e-3)
cv_l1 <- sapply(l1_grid, cv_rloss_dnn)
en_dnn_ctrl$esCtrl$l1.reg <- l1_grid[which.min(cv_l1)]

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
n <- 1000
d <- 20
sigma <- 1
set.seed(4231)
X <- mvrnorm(n, mu = rep(0, d), Sigma = diag(d))
f <- log(abs(X[, 1]) + 1) - X[, 2]^2 + sin(X[, 3]) + 0.5 * X[, 4] * X[, 5]
e <- plogis(0.8 * sin(pi * X[,1] * X[,2]) + 0.6 * X[,3] * X[,4] + 0.5 * tanh(X[,5]))
Z <- rbinom(n, 1, e)
eps <- rnorm(n, 0, sigma)
Y <- f + Z  * 3 + eps ## Assumae tau = 3
object <- importTrt(X, Y, Z)

fit <- davies_test(object)
fit2 <- cv_perm_test(object)
print(fit)
```

## Screening for effect modifiers

Once the global test rejects, `cv_perm_vsel()` asks *which* covariates the
heterogeneity runs along. For each covariate it permutes that column of the
held-out design and re-evaluates the cross-fitted stage-2 model, reusing the
fits rather than refitting anything. Two p-values are reported per covariate:

- `p_marg` permutes the covariate marginally. Under correlated covariates this
  scheme over-rejects nulls that are merely correlated with true modifiers,
  because the permuted rows fall outside the support of the data and the model
  extrapolates.
- `p_cond` permutes only within quantile strata of a random-forest fit of the
  covariate on its correlated neighbours (`|cor| > cor_threshold`), which
  approximates sampling from `P(X_j | X_-j)` and restores type I error
  control. This is the recommended reading when covariates are correlated;
  with independent covariates the two schemes coincide exactly.

The returned data frame also carries the conditioning-set size `n_cond`,
inflation ratios, and importance ranks.

``` r
## X1..X10 equicorrelated at rho = 0.5; only X1..X5 modify tau,
## so X6..X10 are correlated nulls and X11..X20 independent nulls.
n <- 2000; d <- 20; rho <- 0.5
Sigma <- diag(d); Sigma[1:10, 1:10] <- rho; diag(Sigma) <- 1
set.seed(4231)
X <- mvrnorm(n, rep(0, d), Sigma)
f <- log(abs(X[,1]) + 1) - X[,2]^2 + sin(X[,3]) + 0.5 * X[,4] * X[,5]
e <- plogis(0.8 * sin(pi * X[,1] * X[,2]) + 0.6 * X[,3] * X[,4] + 0.5 * tanh(X[,5]))
Z <- rbinom(n, 1, e)
tau <- -1 + X[,1] * X[,2] + cos(X[,3])^2 + pmax(X[,4] - X[,5], 0)
Y <- f + Z * tau + rnorm(n, 0, 1)
object <- importTrt(X, Y, Z)

vs <- cv_perm_vsel(object, k_folds = 5, B = 500,
                   cor_threshold = 0.2, n_strata = 5, n_cores = 4)
vs$screen[order(vs$screen$p_cond), ]
```

Compute note: the screen evaluates `B` permutations per covariate and scheme
against an ensemble DNN, so it is the most expensive step of the pipeline.
`chunk` batches permuted copies into single `predict()` calls and `n_cores`
parallelizes over covariates (forking; not available on Windows).

## References

Mi, X. et al. (2021). A deep learning semiparametric regression for adjusting complex confounding structures. The Annals of Applied Statistics, 15(3):1086–1100.

Nie, X. and Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2):299–319.

Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T. and Zeileis, A. (2008). Conditional variable importance for random forests. BMC Bioinformatics, 9:307.

Berrett, T. B., Wang, Y., Barber, R. F. and Samworth, R. J. (2020). The conditional permutation test for independence while controlling for confounders. Journal of the Royal Statistical Society, Series B, 82(1):175–197.

## Simulation code

The scripts that reproduce the simulation studies of the paper are in
[`simulations/`](simulations/), with their own README.
