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

`dnn_ctrl()` returns the control lists used in the paper. Every network is a bagged
ensemble of 30 networks with hidden layers of 128, 64 and 32 ReLU units, Adam with
learning rate 1e-3, standardized inputs and outputs, bootstrap training with the weights
of the best out-of-bag epoch kept, and an L1 penalty (1e-3 by default, the simulations
used 1e-5 at noise level sigma = 1 and 1e-3 at sigma = 3). Two training budgets are
recommended:

| networks | control list | mini-batch | max epochs | patience |
|---|---|---|---|---|
| outcome models mu(X) and mu*(X) | `dnn_ctrl("outcome")` | 128 | 500 | 50 |
| propensity model, second-stage effect models | `dnn_ctrl("propensity")` | 256 | 120 | 20 |

The outcome budget matters: with the default budget of 120 epochs the outcome network
was undertrained in the simulations of the paper, and the longer budget together with
the input screen below reduced its root mean squared error from 1.25 to 0.58 in the
hardest configuration (n = 1000, p = 40, sigma = 1).

``` r
ctrl_e  <- dnn_ctrl("propensity")     # propensity and stage-2 networks
ctrl_mu <- dnn_ctrl("outcome")        # outcome networks

# ctrl_e expanded, for reference
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
        learning.rate = 1e-3,
        early.stop.det = 20
      )
    )
```

The optional `tune` argument of `weight_dnn()`, `davies_test()`, `cv_perm_test()` and
`cv_perm_vsel()` selects the L1 penalty and the mini-batch size of every network from
`dnn_tune_grid()` within each training fold by out-of-bag loss. The paper uses the fixed
settings above (no `tune`).

## Nuisance inputs: hierarchical lasso screen

With 20 to 40 covariates the nuisance networks are fitted on screened,
interaction-augmented inputs. Within each training fold `screen_augment()` fits a lasso
for Y (gaussian) and for Z (binomial) on the covariates, their squares and all pairwise
products, with the penalty chosen by ten-fold cross-validation (`lambda.min`). Every
covariate that appears in a selected feature is kept (at least five). The outcome models
mu(X) and mu*(X) take the kept covariates together with the selected squares and products
as inputs, the propensity model the kept covariates only. The kernel test and the
second-stage effect model use all p covariates. A main-effect screen misses covariates
that act only through squares or products, which is why the dictionary is hierarchical.

``` r
tr  <- sample(n, 800)
sel <- screen_augment(x[tr, ], y[tr], z[tr])
sel                                              # kept covariates, selected features
X_mu <- nuisance_input(sel, x)                   # outcome networks: kept + squares/products
X_e  <- nuisance_input(sel, x, augmented = FALSE) # propensity network: kept covariates
```

`davies_test()` and `cv_perm_test()` run the screen inside their cross-fitting folds
(`screen = TRUE`, the default) and report the per-fold input counts in `$screen`.

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

Both global tests share the nuisance pipeline of the paper: five folds stratified by
treatment arm, the hierarchical lasso screen within each training fold, the two training
budgets, propensity estimates clipped to [0.05, 0.95], and the revised constant effect
removed before the outcome model is refitted. `davies_test()` is the analytic test, the
product-form kernel score test with the Davies reference distribution, and
`cv_perm_test()` the permutation test on the cross-fitted second-stage loss (predictions
shuffled within fold and treatment arm). Both report the unrevised and the revised
version.

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
Y <- f + Z  * 3 + eps ## Assume tau = 3
object <- importTrt(X, Y, Z)

fit <- davies_test(object, ctrl = dnn_ctrl("propensity", l1 = 1e-5),
                   ctrl_mu = dnn_ctrl("outcome", l1 = 1e-5))   # kernel score test, Davies reference
fit$unrevised; fit$revised                                   # kernel_score_test objects
fit2 <- cv_perm_test(object, B = 2000)                       # cross-fitted permutation test
c(fit$p_davies, fit2$revised$p_value)
```

### The kernel score test

`kernel_score_test()` computes the analytic statistic from any cross-fitted nuisance
estimates. With d_i = Z_i - e_hat(X_i) and the residual R_i = Y_i - tau0 Z_i - mu_hat(X_i)
(tau0 = 0 for the unrevised residual, the estimated constant with mu_hat fitted to
Y - tau0 Z for the revised one), theta_hat = sum_i d_i R_i / sum_i d_i^2, the scores
q_i = d_i (R_i - theta_hat d_i) sum to zero exactly, sigma_hat^2 = n^{-1} sum_i (R_i -
theta_hat d_i)^2, and Q = q' K q / sigma_hat^2 with K the Gaussian kernel matrix of the
standardized covariates (median pairwise distance bandwidth). Under the null hypothesis of
a constant conditional average treatment effect Q converges to sum_j lambda_j chi^2_{1,j},
where lambda_j are the eigenvalues of M_d D K D M_d with M_d = I - d d'/(d'd) and
D = diag(d), and the p-value is computed by the Davies method (Liu approximation as a
fallback). This is the kernel machine score test of Liu, Lin and Ghosh (2007) applied to
the working model R = theta d + d r(X) + eps with r ~ GP(0, sigma_r^2 K), testing
sigma_r^2 = 0. Nuisance error enters the scores only through products and squares of the
propensity and outcome errors (Neyman orthogonality), which replaces the earlier sign-form
statistic (available as `form = "sign"` for comparison).

``` r
ks_u <- kernel_score_test(Y, Z, e_hat, mu_hat, X, return_kernel = TRUE)        # unrevised
ks_r <- kernel_score_test(Y, Z, e_hat, mu_star_hat, X, tau0 = tau0,             # revised
                          K = ks_u$K, lambda = ks_u$lambda)
ks_r$Q; ks_r$p_value; ks_r$lambda
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
## X1..X10 equicorrelated at rho = 0.3; only X1..X5 modify tau,
## so X6..X10 are correlated nulls and X11..X20 independent nulls.
n <- 2000; d <- 20; rho <- 0.3
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

Liu, D., Lin, X. and Ghosh, D. (2007). Semiparametric regression of multidimensional genetic pathway data: least-squares kernel machines and linear mixed models. Biometrics, 63(4):1079–1088.

Davies, R. B. (1980). The distribution of a linear combination of chi-squared random variables. Journal of the Royal Statistical Society, Series C, 29(3):323–333.

Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T. and Zeileis, A. (2008). Conditional variable importance for random forests. BMC Bioinformatics, 9:307.

Berrett, T. B., Wang, Y., Barber, R. F. and Samworth, R. J. (2020). The conditional permutation test for independence while controlling for confounders. Journal of the Royal Statistical Society, Series B, 82(1):175–197.

## Simulation code

The scripts that reproduce the simulation studies of the paper are in
[`simulations/`](simulations/), with their own README.
