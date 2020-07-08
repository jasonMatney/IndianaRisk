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
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\"
setwd(dsn)
IN_df <- read.xlsx(paste0(dsn,"model_data.xlsx"))
dim(IN_df)

IN_mod <- IN_df[ , !(names(IN_df) %in% c("subwatershed"))]

IN_mod <- IN_mod %>% mutate_if(is.character,as.numeric) 
df <- IN_mod[complete.cases(IN_mod), ]

head(df)

# Start local host with given number of threads plus give memory size
h2o.init(ip='localhost', port=54321, nthreads=-1, max_mem_size = '20g')

# Define response and predictors #
response <- "claims_total_building_insurance_coverage_avg"
predictors <- names(IN_mod[,which(!names(IN_mod) %in% c("claims_total_building_insurance_coverage_avg", "subwatershed"))])


###############
## NORMALIZE ##
###############
IN_norm <- normalizeData(df, type='0_1')
colnames(IN_norm) <- names(IN_mod)
IN_norm_x <- normalizeData(df[,predictors], type='0_1')
IN_norm_y <- normalizeData(df[,response], type='0_1')

IN_norm_index <- as.data.frame(IN_norm)
IN_norm_index$subwatershed <- df$subwatershed
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


#################
## Grid search ##
#################


# GBM hyperparameters
learn_rate = seq(0.01,0.1,0.01)
max_depth = seq(1,10,1)
sample_rate = seq(0.1,1,0.1)
col_sample_rate = seq(0.1, 0.9, 0.01)

gbm_params <- list(learn_rate = learn_rate,
                   max_depth = max_depth,
                   sample_rate = sample_rate,
                   col_sample_rate = col_sample_rate)

## DRF 
ntrees = seq(1,100,10)
max_depth = seq(1,100,10)

drf_params = list(ntrees = ntrees, 
                  max_depth = max_depth)

## GLM 
alpha=seq(0.01, 1, 0.01)
lambda=seq(0.00000001,0.0001, 0.000001)

glm_params = list(alpha = alpha,
                  lambda = lambda)

## DEEP LEARNING 
activation = c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout")
hidden = list(c(5, 5, 5, 5, 5), c(10, 10, 10, 10), c(50, 50, 50), c(100, 100, 100))
epochs = c(50, 100, 200)
l1 = seq(0, 0.00001, 0.0001)
l2 = seq(0, 0.00001, 0.0001)
rate = seq(0, 0.1, 0.01)
rate_annealing = c(1e-8, 1e-7, 1e-6)
rho = seq(0.9, 0.999)
epsilon = c(1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
momentum_start = c(0, 0.5)
momentum_stable = c(0.99, 0.5, 0)
input_dropout_ratio = seq(0, 0.5, 0.1)
max_w2 = c(10, 100, 1000, 1000000)

dpl_params = list(
  activation = activation, 
  hidden = hidden,
  epochs = epochs,
  l1 = l1, 
  l2 = l2,
  rate = rate,
  rate_annealing = rate_annealing,
  rho = rho,
  epsilon = epsilon,
  momentum_start = momentum_start,
  momentum_stable = momentum_stable,
  input_dropout_ratio = input_dropout_ratio,
  max_w2 = max_w2
)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 100,
                        seed = 1)

##########################################
######## GRID Search Models ##############
##########################################

glm_grid <- h2o.grid(algorithm = "glm",
                     grid_id = "glm_grid",
                     x = x,
                     y = y,
                     training_frame = train,
                     seed = 1,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = glm_params,
                     search_criteria = search_criteria)

drf_grid <- h2o.grid(algorithm = "drf",
                     grid_id = "drf_grid",
                     x = x,
                     y = y,
                     training_frame = train,
                     seed = 1,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = drf_params,
                     search_criteria = search_criteria)

gbm_grid <- h2o.grid(algorithm = "gbm",
                     grid_id = "gbm_grid",
                     x = x,
                     y = y,
                     training_frame = train,
                     seed = 1,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = gbm_params,
                     search_criteria = search_criteria)

dl_grid <- h2o.grid(algorithm = "deeplearning",
                     grid_id = "dl_grid",
                     x = x,
                     y = y,
                     training_frame = train,
                     seed = 1,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = dpl_params,
                     search_criteria = search_criteria,
                     parallelism = 4)

models = c(glm_grid, gbm_grid, drf_grid, dl_grid)

############################################
### Get the grid results, sorted by RMSE ###
############################################
glm_gridperf <- h2o.getGrid(grid_id = "glm_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)

gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)

drf_gridperf <- h2o.getGrid(grid_id = "drf_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)

dl_gridperf <- h2o.getGrid(grid_id = "dl_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)
print(glm_gridperf)
print(gbm_gridperf)
print(drf_gridperf)
print(dl_gridperf)


# Grab the top GLM model, chosen by RMSE
best_glm <- h2o.getModel(glm_gridperf@model_ids[[1]])
best_glm



# Grab the top GBM model, chosen by RMSE
best_gbm <- h2o.getModel(gbm_gridperf@model_ids[[1]])
best_gbm

# Grab the top DRF model, chosen by RMSE
best_drf <- h2o.getModel(drf_gridperf@model_ids[[1]])
best_drf

# Train a stacked ensemble using the GBM grid
glm_ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                metalearner_algorithm = "glm",
                                training_frame = train,
                                base_models = models)

gbm_ensemble <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    metalearner_algorithm = "gbm",
                                    training_frame = train,
                                    base_models = models)


drf_ensemble <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    metalearner_algorithm = "drf",
                                    training_frame = train,
                                    base_models = models)

dl_ensemble <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    metalearner_algorithm = "deeplearning",
                                    training_frame = train,
                                    base_models = models)


# Compare to base learner performance on the test set
glm_ensemble_test <- h2o.performance(glm_ensemble, newdata = test)
gbm_ensemble_test <- h2o.performance(gbm_ensemble, newdata = test)
drf_ensemble_test <- h2o.performance(drf_ensemble, newdata = test)
dl_ensemble_test <- h2o.performance(dl_ensemble, newdata = test)


# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)

# Compare to base learner performance on the test set
perf_glm_test <- h2o.performance(h2o.getModel(glm_grid@model_ids[[1]]), newdata = test)
perf_drf_test <- h2o.performance(h2o.getModel(drf_grid@model_ids[[1]]), newdata = test)
perf_gbm_test <- h2o.performance(h2o.getModel(gbm_grid@model_ids[[1]]), newdata = test)
perf_dpl_test <- h2o.performance(h2o.getModel(dl_grid@model_ids[[1]]), newdata = test)

baselearner_best_rmse_test <- min(h2o.rmse(perf_glm_test), 
                                  h2o.rmse(perf_drf_test), 
                                  h2o.rmse(perf_gbm_test), 
                                  h2o.rmse(perf_dpl_test))

ensemble_rmse_test <- h2o.rmse(dl_ensemble_test)
print(sprintf("Best Base-learner Test RMSE:  %s", baselearner_best_rmse_test))
print(sprintf("Ensemble Test RMSE:  %s", ensemble_rmse_test))

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(dl_ensemble, newdata = test)

summary(ensemble)
pred_ensemble <- as.data.frame(denormalizeData(pred, getNormParameters(IN_norm_y)))
test_backtransform <- round(as.data.frame(denormalizeData(test$claims_total_building_insurance_coverage_avg, getNormParameters(IN_norm_y))),0)
results <- as.data.frame(cbind(test_backtransform, round(pred_ensemble,0)))
colnames(results) <- c("claims", "predicted")

IN_test_observed <- IN_df[,c("claims_total_building_insurance_coverage_avg","subwatershed")]

model_results <- results %>% left_join(IN_test_observed, by="subwatershed")
colnames(model_results) <- c("subwatershed", "predicted", "observed")
head(model_results)

plot(results$claims, results$predicted, xlim=c(0,50000))
abline(x=y,col="blue")

h2o.varimp_plot(h2o.getModel(drf_grid@model_ids[[1]]))

lares::mplot_full(tag = results$claims,
                  score = results$predicted,
                  splits = 10,
                  subtitle = "Ensemble DL Metalearner Results",
                  model_name = "simple_model_02",
                  save = T)

# # test_denorm <- as.data.frame(denormalizeData(test, getNormParameters(IN_norm_x)))
# # 
# # test_df <- as.data.frame(denormalizeData(pred, getNormParameters(IN_norm_y)))
# # 
# # results <- as.data.frame(cbind(test_df$subwatershed, 
# #                                pred_ensemble, 
# #                                test_df$claims_total_building_insurance_coverage_avg))
# # 
# # colnames(results) <- c("subwatershed", "predicted claims", "observed claims")
# # head(results)
# ############################
# 
# # # GBM Hyperparamters
# # learn_rate_opt <- c(0.01, 0.03)
# # max_depth_opt <- c(3, 4, 5, 6, 9)
# # sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
# # col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# # hyper_params <- list(learn_rate = learn_rate_opt,
# #                      max_depth = max_depth_opt,
# #                      sample_rate = sample_rate_opt,
# #                      col_sample_rate = col_sample_rate_opt)
# # 
# # search_criteria <- list(strategy = "RandomDiscrete",
# #                         max_models = 3,
# #                         seed = 1)
# # 
# # 
# # gbm_grid <- h2o.grid(algorithm = "gbm",
# #                      grid_id = "gbm_grid_binomial",
# #                      x = x,
# #                      y = y,
# #                      training_frame = train,
# #                      ntrees = 10,
# #                      seed = 1,
# #                      nfolds = nfolds,
# #                      keep_cross_validation_predictions = TRUE,
# #                      hyper_params = hyper_params,
# #                      search_criteria = search_criteria)
# # 
# # # Train a stacked ensemble using the GBM grid
# # ensemble <- h2o.stackedEnsemble(x = x,
# #                                 y = y,
# #                                 training_frame = train,
# #                                 base_models = gbm_grid@model_ids)
# # 
# # # Eval ensemble performance on a test set
# # perf <- h2o.performance(ensemble, newdata = test)
# 
# # There are a few ways to assemble a list of models to stack toegether:
# # 1. Train individual models and put them in a list
# # 2. Train a grid of models
# # 3. Train several grids of models
# # Note: All base models must have the same cross-validation folds and
# # the cross-validated predicted values must be kept.
# 
# 
# # 1. Generate a 4-model ensemble (GLM + GBM + DRF + DPL)
# 
# 
# ## Generalized Linear Model
# # glm_IN_22c <- h2o.glm(model_id = "glm_IN_22c", 
# #                       family= "gaussian",
# #                       x= predictors,
# #                       y=response,
# #                       training_frame = train,
# #                       validation_frame = valid, 
# #                       nfolds=nfolds,
# #                       fold_assignment = "Modulo",
# #                       seed = 23123,
# #                       lambda = 0,
# #                       early_stopping = TRUE,
# #                       keep_cross_validation_predictions = TRUE) # Early stopping               
# # 
# # 
# # ##  Random Forest
# # drf_IN_22c <- h2o.randomForest(model_id = "rf_IN_22c",             
# #                           training_frame = train,        
# #                           validation_frame = valid,      
# #                           x=predictors,                 
# #                           y=response, 
# #                           nfolds=nfolds,
# #                           fold_assignment = "Modulo",               
# #                           ntrees = 200,                 
# #                           max_depth = 10,
# #                           stopping_metric = "RMSE",
# #                           stopping_rounds = 10,
# #                           stopping_tolerance = 0.05,     
# #                           score_each_iteration = T,      
# #                           seed = 23123,
# #                           keep_cross_validation_predictions = TRUE)                    
# # 
# # 
# # ## Gradeint Boosting
# # gbm_IN_22c <-  h2o.gbm(model_id = "gbm_IN_22c",       
# #                        x = predictors, 
# #                        y = response,
# #                        training_frame = train,
# #                        nfolds=nfolds,
# #                        fold_assignment = "Modulo",
# #                        ntrees = 200,                 
# #                        max_depth = 10,
# #                        validation_frame = valid, 
# #                        stopping_metric = "RMSE",
# #                        stopping_rounds = 10,
# #                        stopping_tolerance = 0.05,   
# #                        score_each_iteration = T,  
# #                        seed = 23123,
# #                        keep_cross_validation_predictions = TRUE)
# 
# ## Deep Learning
# ## Hyper Parameter Tuning 
# 
# hyper_params <- list(
#   activation=c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
#   hidden=list(c(20,20),c(50,50),c(30,30,30),c(25,25,25,25)),
#   input_dropout_ratio=c(0,0.05),
#   l1=seq(0,1e-4,1e-6),
#   l2=seq(0,1e-4,1e-6)
# )
# 
# hyper_params
# 
# ## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
# search_criteria = list(strategy = "RandomDiscrete", 
#                        max_runtime_secs = 360, 
#                        max_models = 1000, 
#                        seed=1234567, 
#                        stopping_rounds=5, 
#                        stopping_tolerance=1e-2)
# 
# dl_models <- h2o.grid(
#   algorithm="deeplearning",
#   grid_id = "dl_grid_random",
#   training_frame=train,
#   validation_frame=valid, 
#   x=predictors, 
#   y=response,
#   nfolds=nfolds,
#   epochs=1,
#   stopping_metric="rmse",
#   stopping_tolerance=1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
#   stopping_rounds=2,
#   score_validation_samples=1000,   ## downsample validation set for faster scoring
#   score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
#   max_w2=10,                      ## can help improve stability for Rectifier
#   hyper_params = hyper_params,
#   search_criteria = search_criteria,
#   keep_cross_validation_predictions = TRUE
# )                                
# # grid <- h2o.getGrid("dl_grid_random",sort_by="rmse",decreasing=FALSE)
# # grid
# # 
# # grid@summary_table[1,]
# # best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest rmse
# # 
# # best_model
# # dpl_IN_22c
# 
# #########
# # dpl_IN_22c <- h2o.deeplearning(model_id = "dpl_IN_22c",
# #                        x = predictors,
# #                        y = response,
# #                        nfolds=nfolds,
# #                        rho = 0.95,
# #                        epsilon = 1e-5,
# #                        fold_assignment = "Modulo",
# #                        distribution = "gaussian",
# #                        hidden = c(25,25,25,25,25,1),
# #                        epochs = 50,
# #                        train_samples_per_iteration = 16,
# #                        reproducible = TRUE,
# #                        activation = "Tanh",
# #                        loss = "Quadratic",
# #                        l1=1e-5,
# #                        l2=1e-5,
# #                        seed = 23123,
# #                        # stopping_metric = "RMSE",
# #                        # stopping_tolerance=1e-4,        ## stop when misclassification does not improve by >=1% for 2 scoring events
# #                        # stopping_rounds=4,
# #                        training_frame = train,
# #                        validation_frame = valid,
# #                        keep_cross_validation_predictions = TRUE)
# 
# # plot(dpl_IN_22c)
# # 
# # # Train a stacked ensemble using the GBM and RF above
# # # ensemble <- h2o.stackedEnsemble(model_id = "ensemble_IN_22c",
# # #                                 x = predictors,
# # #                                 y = response,
# # #                                 training_frame = train,
# # #                                 validation_frame = valid,
# # #                                 metalearner_algorithm = "deeplearning",
# # #                                 metalearner_nfolds = nfolds,
# # #                                 metalearner_fold_assignment = "Modulo",
# # #                                 seed = 23123,
# # #                                 base_models = list(glm_IN_22c,
# # #                                                    drf_IN_22c,
# # #                                                    gbm_IN_22c,
# # #                                                    best_model))
# # 
# # h2o.varimp_plot(dpl_IN_22c)