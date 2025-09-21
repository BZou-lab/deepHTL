args <- commandArgs(trailingOnly = TRUE)
batch_id <- as.numeric(args[1])

###########################################################
#'
#' The algorithm for double deep treatment learning in comparative effectiveness analysis.
#'
#' @param object A \code{dnnetInput} or a \code{dnnetSurvInput} object, the training set.
#' @param en_dnn_ctrl1 A list of parameters to be passed to \code{ensemble_dnnet} for model 1.
#' @param en_dnn_ctrl2 A list of parameters to be passed to \code{ensemble_dnnet} for model 2.
#' @param methods Methods included in this analysis.
#' @param ... Parameters passed to both en_dnn_ctrl1 and en_dnn_ctrl2.
#'
#' @return Returns a \code{list} of results.
#'
#' @importFrom stats lm
#' @importFrom stats coefficients
#' @importFrom stats vcov
#'
#' @export
# Load necessary libraries
# Load necessary libraries

# install.packages(c("e1071", "xgboost", "caret"))
# install_github("https://github.com/xnie/rlearner")
# install_github("https://github.com/SkadiEye/deepTL")
library(xgboost); library(dplyr); library(caret); library(rlearner); library(deepTL); library(MASS);library(glmnet)

setwd("/nas/longleaf/home/shuaiy/project/htdnn")
source("dnn.R")
source("xgboost.R")
source("lasso.R")

###### Scenario 15 ######

n <- 2000
p <- 20
sigma <- 3
method_vec <- c("rlearner-lasso", "rlearner-xgboost",
                "weight-xgboost","weight-lasso", "weight-dnn")

set.seed(n + p + sigma + 10 * batch_id)
x <- matrix(rnorm(n * p), n, p)
bx <- log(abs(x[,1]) + 1) - x[,2]^2 + sin(x[,3]) + 0.5*x[,4]*x[,5]
ex <- plogis(0.3*x[,1]^2 - 0.2*sin(x[,2]) + 0.3*x[,3]*x[,4] - 0.1*x[,5]^2)
eps <- rnorm(n, 0, sigma)
z <- rbinom(n, 1, ex)
softplus <- function(u, a = 5) log1p(exp(a*u))/a
R <- qr.Q(qr(matrix(rnorm(p*p), p, p))) 
u <- x %*% R
tx <- 2 + 2*plogis(2*sin(3*pi*u[,1]*u[,2]) + (u[,3]-0.5)^2 + softplus(u[,4]+u[,5]-1, a=6)
                  + 0.7*cos(u[,6]+u[,7]) + 0.25*tanh(u[,8]*u[,9])*u[,10])
y <- bx + (z - 0.5) * tx + eps

xt <- matrix(rnorm(n * p), n, p)
ut <- xt %*% R
tt <- 2 + 2*plogis(2*sin(3*pi*ut[,1]*ut[,2]) + (ut[,3]-0.5)^2 + softplus(ut[,4]+ut[,5]-1, a = 6)
                   + 0.7*cos(ut[,6]+ut[,7]) + 0.25*tanh(ut[,8]*ut[,9])*ut[,10])

obj_tr <- importTrt(x, y, z)

sce <- list()
iter <- 1

for (method in method_vec) {
  if (method == "rlearner-lasso") {
    fit <- rlasso(obj_tr@x, obj_tr@z, obj_tr@y)
    tau_hat <- predict(fit, newx = xt)
  } else if (method == "rlearner-xgboost") {
    fit <- rboost(obj_tr@x, obj_tr@z, obj_tr@y)
    tau_hat <- predict(fit, newx = xt)
  } else if (method == "weight-lasso") {
    fit <- weight_lasso(obj_tr)
    tau_hat <- predict.weight_lasso(fit, xt, "both")
  } else if (method == "weight-xgboost") {
    fit <- weight_xgboost(obj_tr)
    tau_hat <- predict.weight_xgboost(fit, xt, "both")
  } else if (method == "weight-dnn") {
    fit <- weight_dnn(obj_tr)
    tau_hat <- predict.weight_dnn(fit, xt, "both")
  } 
  
  if (grepl("^rlearner", method)) {
    logmse <- log(mean((tau_hat - tt)^2))
    sce[[iter]] <- data.frame(n = n, p = p, sigma = sigma,
                              method = method, logmse = logmse)
  } else{
    logmse <- c(log(mean((tau_hat[,1] - tt)^2)), log(mean((tau_hat[,2] - tt)^2)))
    logmse_u <- logmse[1]
    logmse_r <- logmse[2]
    
    sce[[iter]] <- data.frame(n = n, p = p, sigma = sigma,
                              method = paste0("unrev-", method),
                              logmse = logmse_u)
    iter <- iter + 1
    
    sce[[iter]] <- data.frame(n = n, p = p, sigma = sigma,
                              method = paste0("rev-", method),
                              logmse = logmse_r)
  }
  iter <- iter + 1
}

df_sce15 <- do.call(rbind, lapply(sce, as.data.frame))

output_file <- sprintf("/nas/longleaf/home/shuaiy/project/simulation/sce15/res%d.Rdata", batch_id)
save(df_sce15, file = output_file)