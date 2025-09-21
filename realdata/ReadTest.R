library(haven)
library(MatchIt)
library(dplyr)
library(corrplot)
library(glmnet)
library(dplyr)
library(broom)
library(grf)
library(rlearner)
library(deepTL)
library(parallel)
source("GLiDeR_Rfunctions.txt")

TLD <- read_dta("/Users/shuaiyuan/Desktop/heterdnn/translung_complete_data.dta")
TLD <- TLD %>% mutate(TYPE = ifelse(TYPE == "BLT", 1, 0))
confounders <- setdiff(colnames(TLD), c("TYPE", "FEV1_follow"))

## GLiDeR for selectinig confounders
Xorig <- as.matrix(TLD[, confounders]) 
Xorig <- apply(Xorig, 2, function(col) {
  if (is.character(col) || is.factor(col)) {
    return(as.numeric(as.factor(col)) - 1) 
  }
  return(as.numeric(col)) 
})
Yorig <- TLD$FEV1_follow
Aorig <- TLD$TYPE
param.est <- GLiDeR(as.matrix(Xorig), Yorig, Aorig)
confounders <- colnames(Xorig)[param.est$gamma != 0]
cat("Key confounders driving HTE:", confounders)

TLD <- TLD[,c(confounders,"TYPE", "FEV1_follow")]
save(TLD, file = "/Users/shuaiyuan/Desktop/heterdnn/TLD_Selected.RData")



