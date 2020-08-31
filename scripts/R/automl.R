rm(list=ls())
library(h2o)
library(data.table)
library(bit64)
library(dplyr)
library(lubridate)
library(openxlsx)
library(tidyverse)
library(keras)
library(RSNNS)
library(rlang)
library(lares)
library (readr)
library(dlookr)
h2o.shutdown()


h2o.init()

# data
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\"
setwd(dsn)

IN_df <- read.xlsx(paste0(dsn,"MOD_DF_0723.xlsx"))
# create html file. file name is EDA_Report.html
eda_report(IN_mod, "mean_claim", output_format = "html")

IN_mod <- IN_df[ , !(names(IN_df) %in% c("subwatershed"))]

IN_mod <- IN_mod %>% mutate_if(is.character,as.numeric) 
df <- IN_mod[complete.cases(IN_mod), ]

response <- "mean_claim"
predictors <- names(IN_mod[,which(!names(IN_mod) %in% c("mean_claim", "subwatershed"))])


###############
## NORMALIZE ##
###############
IN_norm <- normalizeData(df, type='0_1')
colnames(IN_norm) <- names(IN_mod)
IN_norm_x <- normalizeData(df[,predictors], type='0_1')
IN_norm_y <- normalizeData(df[,response], type='0_1')

IN_norm_index <- as.data.frame(IN_norm)
#IN_norm_index$subwatershed <- IN_df$subwatershed
head(IN_norm_index)

IN_h2o <- as.h2o(IN_norm_index)

## Splits datasets into train, valid and test
splits <- h2o.splitFrame(data=IN_h2o, ratios=c(0.8, 0.1), seed=1236)
names(splits) <- c("train", "valid", "test")

train <- splits$train
test <- splits$test
head(train)


# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml <- h2o.automl(x = predictors, y = response,
                  training_frame = train,
                  validation_frame = valid,
                  max_models = 20,
                  seed = 1)
# View the AutoML Leaderboard
lb <- aml@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
pred <- h2o.predict(aml@leader, test)
pred
h2o.r2(aml@leader, valid = TRUE)
