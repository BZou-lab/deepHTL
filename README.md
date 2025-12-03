# deepHTL

Deep Heterogeneous Treatment Learning, a revised deep learning semiparametric regression for **testing** and **estimating** the heterogeneous treatment effects in observational studies.

## Simulations (Testing) 

``` r
set.seed(4231)
suppressMessages(library(deepHTL))
n <- 1000
d <- 20
sigma <- 1
X <- mvrnorm(n, mu = rep(0, d), Sigma = diag(d))
b <- log(abs(X[, 1]) + 1) - X[, 2]^2 + sin(X[, 3]) + 0.5 * X[, 4] * X[, 5]
e <- plogis(0.8 * sin(pi * X[,1] * X[,2]) + 0.6 * X[,3] * X[,4] + 0.5 * tanh(X[,5]))
Z <- rbinom(n, 1, e)
tau <- rep(3, n)
eps <- rnorm(n, 0, sigma)
Y <- b + (Z - 0.5) * tau + eps
object <- importTrt(X, Y, Z)
set.seed(4231)
fit <- deepTL_null(object)
```
