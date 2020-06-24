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

h2o.shutdown()
# data
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk"
setwd(dsn)
IN_df <- read.xlsx(paste0(dsn,"\\data\\model_data\\IN_Risk_Model.xlsx"))
dim(IN_df)

IN_mod <- IN_df[ , !(names(IN_df) %in% c("subwatershed"))]

IN_mod <- IN_mod %>% mutate_if(is.character,as.numeric) 

# Start local host with given number of threads plus give memory size
h2o.init(ip='localhost', port=54321, nthreads=-1, max_mem_size = '20g')

# Define response and predictors #
response <- "claims_total_building_insurance_coverage_avg"
predictors <- names(IN_mod[,which(!names(IN_mod) %in% c("claims_total_building_insurance_coverage_avg", "subwatershed"))])


###############
## NORMALIZE ##
###############
IN_norm <- normalizeData(IN_mod, type='0_1')
colnames(IN_norm) <- names(IN_mod)
IN_norm_x <- normalizeData(IN_mod[,predictors], type='0_1')
IN_norm_y <- normalizeData(IN_mod[,response], type='0_1')

IN_norm_index <- as.data.frame(IN_norm)
IN_norm_index$subwatershed <- IN_df$subwatershed
head(IN_norm_index)
# convert into H2O frame
IN_h2o <- as.h2o(IN_norm_index)


## Splits datasets into train, valid and test
splits <- h2o.splitFrame(data=IN_h2o, ratios=c(0.8, 0.1), seed=1236)
names(splits) <- c("train","valid","test")

train <- splits$train
valid <- splits$valid
test <- splits$test


# Identify predictors and response
y <- response
x <- predictors


# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

# There are a few ways to assemble a list of models to stack toegether:
# 1. Train individual models and put them in a list
# 2. Train a grid of models
# 3. Train several grids of models
# Note: All base models must have the same cross-validation folds and
# the cross-validated predicted values must be kept.


# 1. Generate a 4-model ensemble (GLM + GBM + DRF + DPL)


## Generalized Linear Model
glm_IN_22c <- h2o.glm(model_id = "glm_IN_22c", 
                      family= "gaussian",
                      x= predictors,
                      y=response,
                      training_frame = train,
                      validation_frame = valid, 
                      nfolds=nfolds,
                      fold_assignment = "Modulo",
                      seed = 23123,
                      lambda = 0,
                      early_stopping = TRUE,
                      keep_cross_validation_predictions = TRUE) # Early stopping               


##  Random Forest
drf_IN_22c <- h2o.randomForest(model_id = "rf_IN_22c",             
                          training_frame = train,        
                          validation_frame = valid,      
                          x=predictors,                 
                          y=response, 
                          nfolds=nfolds,
                          fold_assignment = "Modulo",               
                          ntrees = 200,                 
                          max_depth = 10,
                          stopping_metric = "RMSE",
                          stopping_rounds = 10,
                          stopping_tolerance = 0.05,     
                          score_each_iteration = T,      
                          seed = 23123,
                          keep_cross_validation_predictions = TRUE)                    


## Gradeint Boosting
gbm_IN_22c <-  h2o.gbm(model_id = "gbm_IN_22c",       
                       x = predictors, 
                       y = response,
                       training_frame = train,
                       nfolds=nfolds,
                       fold_assignment = "Modulo",
                       ntrees = 200,                 
                       max_depth = 10,
                       validation_frame = valid, 
                       stopping_metric = "RMSE",
                       stopping_rounds = 10,
                       stopping_tolerance = 0.05,   
                       score_each_iteration = T,  
                       seed = 23123,
                       keep_cross_validation_predictions = TRUE)

## Deep Learning
## Hyper Parameter Tuning 

hyper_params <- list(
  activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20),c(50,50),c(30,30,30),c(25,25,25,25)),
  input_dropout_ratio=c(0,0.05),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)
hyper_params

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria = list(strategy = "RandomDiscrete", 
                       max_runtime_secs = 360, 
                       max_models = 1000, 
                       seed=1234567, 
                       stopping_rounds=5, 
                       stopping_tolerance=1e-2)

dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=train,
  validation_frame=valid, 
  x=predictors, 
  y=response,
  nfolds=nfolds,
  epochs=1,
  stopping_metric="rmse",
  stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  score_validation_samples=1000,   ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  keep_cross_validation_predictions = TRUE
)                                
grid <- h2o.getGrid("dl_grid_random",sort_by="rmse",decreasing=FALSE)
grid

grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest rmse

best_model
dpl_IN_22c

#########
dpl_IN_22c <- h2o.deeplearning(model_id = "dpl_IN_22c",
                       x = predictors,
                       y = response,
                       nfolds=nfolds,
                       rho = 0.95,
                       epsilon = 1e-5,
                       fold_assignment = "Modulo",
                       distribution = "gaussian",
                       hidden = c(25,25,25,25,25,1),
                       epochs = 50,
                       train_samples_per_iteration = 16,
                       reproducible = TRUE,
                       activation = "Tanh",
                       loss = "Quadratic",
                       l1=1e-5,
                       l2=1e-5,
                       seed = 23123,
                       # stopping_metric = "RMSE",
                       # stopping_tolerance=1e-4,        ## stop when misclassification does not improve by >=1% for 2 scoring events
                       # stopping_rounds=4,
                       training_frame = train,
                       validation_frame = valid,
                       keep_cross_validation_predictions = TRUE)

plot(dpl_IN_22c)

# Train a stacked ensemble using the GBM and RF above
# ensemble <- h2o.stackedEnsemble(model_id = "ensemble_IN_22c",
#                                 x = predictors,
#                                 y = response,
#                                 training_frame = train,
#                                 validation_frame = valid,
#                                 metalearner_algorithm = "deeplearning",
#                                 metalearner_nfolds = nfolds,
#                                 metalearner_fold_assignment = "Modulo",
#                                 seed = 23123,
#                                 base_models = list(glm_IN_22c,
#                                                    drf_IN_22c,
#                                                    gbm_IN_22c,
#                                                    best_model))

h2o.varimp_plot(dpl_IN_22c)

# Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                base_models = gbm_grid@model_ids)


# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)

# Compare to base learner performance on the test set
perf_glm_test <- h2o.performance(glm_IN_22c, newdata = test)
perf_drf_test <- h2o.performance(drf_IN_22c, newdata = test)
perf_gbm_test <- h2o.performance(gbm_IN_22c, newdata = test)
perf_dpl_test <- h2o.performance(dpl_IN_22c, newdata = test)

baselearner_best_rmse_test <- min(h2o.rmse(perf_glm_test), 
                                  h2o.rmse(perf_drf_test), 
                                  h2o.rmse(perf_gbm_test), 
                                  h2o.rmse(perf_dpl_test))

ensemble_rmse_test <- h2o.rmse(perf)
print(sprintf("Best Base-learner Test RMSE:  %s", baselearner_best_rmse_test))
print(sprintf("Ensemble Test RMSE:  %s", ensemble_rmse_test))

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(ensemble, newdata = test)

summary(ensemble)
pred_ensemble <- as.data.frame(denormalizeData(pred, getNormParameters(IN_norm_y)))
results <- as.data.frame(cbind(as.data.frame(test$subwatershed), pred_ensemble))
colnames(results) <- c("subwatershed", "predicted")

IN_test_observed <- IN_df[,c("claims_total_building_insurance_coverage_avg","subwatershed")]

model_results <- results %>% left_join(IN_test_observed, by="subwatershed")
head(model_results)

# test_denorm <- as.data.frame(denormalizeData(test, getNormParameters(IN_norm_x)))
# 
# test_df <- as.data.frame(denormalizeData(pred, getNormParameters(IN_norm_y)))
# 
# results <- as.data.frame(cbind(test_df$subwatershed, 
#                                pred_ensemble, 
#                                test_df$claims_total_building_insurance_coverage_avg))
# 
# colnames(results) <- c("subwatershed", "predicted claims", "observed claims")
# head(results)
############################

# # GBM Hyperparamters
# learn_rate_opt <- c(0.01, 0.03)
# max_depth_opt <- c(3, 4, 5, 6, 9)
# sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
# col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# hyper_params <- list(learn_rate = learn_rate_opt,
#                      max_depth = max_depth_opt,
#                      sample_rate = sample_rate_opt,
#                      col_sample_rate = col_sample_rate_opt)
# 
# search_criteria <- list(strategy = "RandomDiscrete",
#                         max_models = 3,
#                         seed = 1)
# 
# 
# gbm_grid <- h2o.grid(algorithm = "gbm",
#                      grid_id = "gbm_grid_binomial",
#                      x = x,
#                      y = y,
#                      training_frame = train,
#                      ntrees = 10,
#                      seed = 1,
#                      nfolds = nfolds,
#                      keep_cross_validation_predictions = TRUE,
#                      hyper_params = hyper_params,
#                      search_criteria = search_criteria)
# 
# # Train a stacked ensemble using the GBM grid
# ensemble <- h2o.stackedEnsemble(x = x,
#                                 y = y,
#                                 training_frame = train,
#                                 base_models = gbm_grid@model_ids)
# 
# # Eval ensemble performance on a test set
# perf <- h2o.performance(ensemble, newdata = test)
