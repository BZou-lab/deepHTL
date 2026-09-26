# cm_dgp2.R -- 2026-09-25 (night). cm_dgp.R plus controlled SIMPLIFICATIONS of the nuisance functions (ChatGPT's 4-way design):
#   CM_FSIMPLE=1  f(X) = X1 + X2 + 0.5 X3 + 0.5 X4 + 0.5 X5   (linear, Var ~ 2.56, matching the original f's ~2.5)
#   CM_ESIMPLE=1  logit e(X) = 0.48 X1 + 0.48 X2 + 0.40 X4 + 0.32 X5   (linear, Var(logit) ~ 0.57, matching the original ~0.56)
# Both unset reproduces cm_dgp.R exactly (gen_cm resolves cm_f / cm_e at call time, so the overrides take effect).
source("cm_dgp.R")
CM_FSIMPLE <- identical(Sys.getenv("CM_FSIMPLE"), "1"); CM_ESIMPLE <- identical(Sys.getenv("CM_ESIMPLE"), "1"); CM_EMAIN <- identical(Sys.getenv("CM_EMAIN"), "1")
if (CM_FSIMPLE) cm_f <- function(X) X[,1] + X[,2] + 0.5 * X[,3] + 0.5 * X[,4] + 0.5 * X[,5]
if (CM_ESIMPLE) cm_e <- function(X) plogis(0.48 * X[,1] + 0.48 * X[,2] + 0.40 * X[,4] + 0.32 * X[,5])
#   CM_EMAIN=1    logit e(X) = 0.4 X1 + 0.4 X2 + 0.6 sin(pi X1 X2) + 0.5 X3 X4 + 0.4 tanh(X5): the original interactions PLUS main effects
#                 (Var(logit) ~ 0.6, sd ~ 0.78, close to the original 0.72); the candidate "learnable but still nonlinear" propensity
if (CM_EMAIN) cm_e <- function(X) plogis(0.4 * X[,1] + 0.4 * X[,2] + 0.6 * sin(pi * X[,1] * X[,2]) + 0.5 * X[,3] * X[,4] + 0.4 * tanh(X[,5]))
CM_DGPVAR <- paste0(if (CM_FSIMPLE) "fS" else "fO", if (CM_ESIMPLE) "eS" else if (CM_EMAIN) "eM" else "eO")
