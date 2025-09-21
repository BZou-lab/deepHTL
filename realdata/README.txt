#READ ME#
#The functions required for running the code in this document are in the file GLiDeR_Rfunctions.txt

#To demonstrate how to run the code for GLiDeR, we use the following example:
#################################################################################################
#This example generates data according to Scenario 10 from the manuscript using p = 10 covariates
set.seed(151)

#sample size
n <- 500

#number of covariates
p <- 10

#Generate matrix of covariates
Xorig <- matrix(rnorm(n*p, mean = 1, sd = 2), n, p)

#Generate probability that A = 1 given covariates
P <-  exp(0.2*Xorig[,1] - 2*Xorig[,2] + Xorig[,5] - Xorig[,6] + Xorig[,7] - Xorig[,8])/
	(1+exp(0.2*Xorig[,1] - 2*Xorig[,2] + Xorig[,5] - Xorig[,6] + Xorig[,7] - Xorig[,8]))

#Generate treatment indicators
Aorig <- rbinom(n, 1, P)

#Generate outcomes
Yorig <-  Aorig + 2*Xorig[,1] + 0.2*Xorig[,2] + 5*Xorig[,3] + 5*Xorig[,4] + rnorm(n, mean=0, sd=2)
#################################################################################################


#################################################################################################
##Note the library "boot" is required to run the bootstrap with the "GLiDeR_bootstrap" function

#The following line of code calculates GLiDeR parameter estimates at "optimal" lambda (see comments below for details)
param.est <- GLiDeR(Xorig, Yorig, Aorig, lambda = NULL)

########################################################################################
#The code above obtains the following parameter estimates in order:
#delta = estimated average causal treatment effect
#alpha = outcome model covariate coefficient estimates
#gamma = treatment model covariate coefficient estimates
#alpha0 = outcome model intercept
#gamma0 = treatment model intercept
#alpha_A = outcome model treatment main effect term
#lambda_star = "optimal" lambda value chosen by GCV
##IMPORTANT##
########################################################################################
##By default, GLiDeR calculates estimates over a uniform sequence of lambda from 
##0.00005 to the largest lambda such that all covariate coefficients are zero,
##If selected lambda is therefore equal to 0.00005, it is suggested to try a sequence
##with smaller values of lambda. Moreover, because the default sequence of lambda is 
##uniform and not concentrated near the "optimal" lambda, performance may be improved
##by considering a finer sequence surrounding the "optimal" lambda after running GLiDeR
##over the default sequence of lambda (this can be done by setting lambda equal to a vector - see below)
########################################################################################

#The following runs GLiDeR for a given sequence of lambda 
lambda_seq <- seq(0.005, 0.0001, length.out=100)
est.given.lambda <- GLiDeR(Xorig, Yorig, Aorig, lambda=lambda_seq)

#The following obtains B bootstrap samples of the ACE (note "optimal" lambda is needed)
B=10
boot.samp <- GLiDeR_bootstrap(Xorig, Yorig, Aorig, lambda_star = param.est$lambda_star, B=B)