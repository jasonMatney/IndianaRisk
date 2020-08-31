options(scipen=999)
pacman::p_load(h2o, data.table, bit64, dplyr, lubridate,
               openxlsx, tidyverse, keras, RSNNS, rlang,
               lares, readr, sparklyr, tidyverse,
               caret, car, ggcorrplot, car, sparklyr)


##, h2o.shutdown()

dsn = "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\"

# data
load_data <- function(in_path){
  dsn <- in_path
  setwd(dsn)
  IN_df <- read.xlsx(paste0(dsn,"MOD_DF_08_27.xlsx"))
  IN_mod <- IN_df[ , !(names(IN_df) %in% c("subwatershed"))]
  IN_mod <- IN_mod %>% mutate_if(is.character,as.numeric) 
  # calulate the correlations
  r <- cor(IN_mod[,1:20], use="complete.obs")
  round(r,2)
  ggcorrplot(r)
  
  IN_mod <- IN_mod %>% mutate_if(is.character,as.numeric) 
  ml_df <- IN_mod[complete.cases(IN_mod), ] #### VIF NAMES IF FEATURE SELECTION ON
  
  currentData <- ml_df#[, vif_names]
  linear_reg <- lm(mean_claim ~ . ,data=currentData)
  
  outlierResult <-outlierTest(linear_reg)
  exclusionRows <-names(outlierResult[[1]])
  inclusionRows <- !(rownames(currentData) %in% exclusionRows)
  currentData <- currentData[inclusionRows,]
  
  model_data <- filter(currentData, mean_claim >= 0 )
  return(model_data)
}

model_data <- load_data(dsn)

dim(model_data)



# Start local host with given number of threads plus give memory size
h2o.init(max_mem_size = '32G')

# Define response and predictors #
response <- "mean_claim"
predictors <- names(model_data[,which(!names(model_data) %in% c("mean_claim", "subwatershed"))])


###############
## NORMALIZE ##
###############

h2o_normalize <- function(input_df){
  IN_norm <- normalizeData(model_data, type='0_1')
  colnames(IN_norm) <- names(model_data)
  IN_norm_x <- normalizeData(model_data[,predictors], type='0_1')
  IN_norm_y <- normalizeData(model_data[,response], type='0_1')
  
  norm_df <- as.data.frame(IN_norm)
  norm_df$subwatershed <- model_data$subwatershed
  return(norm_df)
}

norm_df <- h2o_normalize(model_data)

IN_h2o <- as.h2o(norm_df)


## Splits datasets into train, valid and test
splits <- h2o.splitFrame(data=IN_h2o, ratios=c(0.8), seed=1236)

train <- splits[[1]]
valid <- splits[[2]]

# Number of CV folds (to generate level-one data for stacking)
nfolds <- 10


glm = h2o.glm(x = predictors,
              y = response,
              training_frame = train,
              validation_frame = valid,
              seed = 1975)

gbm = h2o.gbm(x = predictors,
              y = response,
              training_frame = train,
              validation_frame = valid,
              seed = 1975)

drf = h2o.randomForest(x = predictors,
              y = response,
              training_frame = train,
              validation_frame = valid,
              seed = 1975)

ann = h2o.deeplearning(x = predictors,
              y = response,
              training_frame = train,
              validation_frame = valid,
              seed = 1975)

h2o.r2(glm, valid=TRUE)
h2o.r2(gbm, valid=TRUE)
h2o.r2(drf, valid=TRUE)
h2o.r2(ann, valid=TRUE)

# aml_norm <- h2o.automl(x = predictors, y = response, 
#                        training_frame = train, 
#                        validation_frame = valid,
#                        seed = 1975,
#                        max_runtime_secs = 3600)
# 
# # When it finishes, check out the leaderboard
# print(aml@leaderboard)
# 
# h2o.r2(aml_norm@leader, valid=TRUE)
# 
# pred = h2o.predict(aml@leader, valid)
# pred_df <- as.data.frame(pred)
# test_df <- as.data.frame(valid$mean_claim)
# compare <- as.data.frame(cbind(test_df, pred_df))
# colnames(compare) <- c("observed", "predicted")
# head(compare)
# 
# 
# plot(compare$predicted,compare$observed,
#      xlab = "Predicted", ylab = "Actual",
#      main = "Predicted vs Actual: AML, Test Data",
#      col = "dodgerblue", pch = 20)
# grid()
# abline(0, 1, col = "darkorange", lwd = 2)

#################
## Grid search ##
#################

## GLM hyperparameters
alpha=seq(0.01, 1, 0.01)
lambda=seq(0.00000001,0.0001, 0.000001)

hyper_params_glm <- 
  list(
    alpha  = alpha,
    lambda = lambda
  )

# GBM hyperparameters
learn_rate = seq(0.01,0.1,0.01)
sample_rate = seq(0.1,1,0.1)
col_sample_rate = seq(0.1, 1, 0.01)
max_depth = seq(1, 30, 3)
min_rows = seq(1,10,1)

hyper_params_gbm <- 
  list(
    learn_rate               = learn_rate,
    sample_rate              = sample_rate,
    col_sample_rate_per_tree = col_sample_rate,
    max_depth                = max_depth,
    min_rows                 = min_rows
  )

## DRF 
mtries = seq(2, 5, by = 1)
sample_rate = seq(0.1, 1, 0.1)
col_sample_rate_per_tree = seq(0.1, 1, 0.1)
max_depth = seq(1, 30, 3)
ntrees = seq(1,100,10)
min_rows = c(1, 2, 5, 10)

hyper_params_drf <- 
  list(
    mtries                   = mtries, 
    sample_rate              = sample_rate,
    col_sample_rate_per_tree = col_sample_rate_per_tree,
    max_depth                = max_depth,
    ntrees                   = ntrees, 
    min_rows                 = min_rows
  )


## DEEP LEARNING 
hyper_params_ann <- list(
  activation          = c("Rectifier","Tanh","Maxout","RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden              = list(c(5, 5, 5, 5, 5), 
                             c(10, 10, 10, 10), 
                             c(20,20),
                             c(25, 25, 25, 25),
                             c(32, 32, 32), 
                             c(50,50), 
                             c(50, 50, 50), 
                             c(64, 64, 64, 64, 64),
                             c(100, 100, 100)),
  l1                  = seq(0,1e-4,1e-6),
  l2                  = seq(0,1e-4,1e-6),
  rate                = c(0, 0.1, 0.005, 0.001),
  rate_annealing      = c(1e-8, 1e-7, 1e-6),
  rho                 = c(0.9, 0.95, 0.99, 0.999),
  epsilon             = c(1e-10, 1e-8, 1e-6, 1e-4),
  momentum_start      = c(0, 0.5),
  momentum_stable     = c(0.99, 0.5, 0),
  input_dropout_ratio = c(0, 0.1, 0.2),
  max_w2              = c(10, 100, 1000, 1000000)
)


search_criteria_max_runtime = list(strategy = "RandomDiscrete",
                                   max_models = 50)




#################
## GRID SEARCH ##
#################

# GLM model
glm_model_grid <- 
  h2o.grid(
    algorithm        = "glm", 
    x                = predictors, 
    y                = response,
    training_frame   = train,
    validation_frame = valid,
    nfolds           = nfolds,
    keep_cross_validation_predictions = TRUE,
    grid_id          = "glm_grid",
    hyper_params     = hyper_params_glm,
    search_criteria  = search_criteria_max_runtime,
    seed             = 1975
  )

# GBM model
gbm_model_grid <- 
  h2o.grid(
    algorithm        = "gbm", 
    x                = predictors, 
    y                = response,
    training_frame   = train,
    validation_frame = valid,
    nfolds           = nfolds,
    keep_cross_validation_predictions = TRUE,
    grid_id          = "gbm_grid",
    hyper_params     = hyper_params_gbm,
    search_criteria  = search_criteria_max_runtime,
    seed             = 1975
  )


# Random Forest model
drf_model_grid <- 
  h2o.grid(
    algorithm        = "randomForest", 
    x                = predictors, 
    y                = response,
    training_frame   = train,
    validation_frame = valid,
    nfolds           = nfolds,
    keep_cross_validation_predictions = TRUE,
    grid_id          = "drf_grid",
    hyper_params     = hyper_params_drf,
    search_criteria  = search_criteria_max_runtime,
    seed             = 1975
  )

# ANN model
ann_model_grid <- 
  h2o.grid(
    algorithm                         = "deeplearning", 
    x                                 = predictors, 
    y                                 = response,
    training_frame                    = train,
    validation_frame                  = valid,
    nfolds                            = nfolds,
    keep_cross_validation_predictions = TRUE,
    grid_id                           = "ann_grid",
    epochs                            = 20,
    hyper_params                      = hyper_params_ann,
    search_criteria                   = search_criteria_max_runtime,
    seed                              = 1975
  )



# AUTO

# model_h2o_automl <- h2o.automl(y = y, 
#                                x = x,
#                                nfolds = nfolds,
#                                training_frame = train, 
#                                validation_frame = valid,
#                                max_models = 10)
# 
# lb <- h2o.get_leaderboard(model_h2o_automl)
# h2o.r2(model_h2o_automl@leader, valid=TRUE)

############################################
### Get the grid results, sorted by RMSE ###
############################################

# GLM
glm_gridperf <- h2o.getGrid(grid_id = "glm_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)
best_glm <- h2o.getModel(glm_gridperf@model_ids[[1]])

# GBM
gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)

best_gbm <- h2o.getModel(gbm_gridperf@model_ids[[1]])

# DRF
drf_gridperf <- h2o.getGrid(grid_id = "drf_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)

best_drf <- h2o.getModel(drf_gridperf@model_ids[[1]])

# ANN
ann_gridperf <- h2o.getGrid(grid_id = "ann_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)
best_ann <- h2o.getModel(ann_gridperf@model_ids[[1]])



best_glm
h2o.r2(best_glm, valid = TRUE)

best_gbm
h2o.r2(best_gbm, valid = TRUE)

best_drf
h2o.r2(best_drf, valid = TRUE)

best_ann
h2o.r2(best_ann, valid = TRUE)


# Coefficients that can be applied to the non-standardized data
h2o.coef(best_glm)

# Coefficients fitted on the standardized data (requires standardize=TRUE, which is on by default)
h2o.coef_norm(best_glm)

# Print the coefficients table
best_glm@model$coefficients_table


ensemble_models = c(glm_model_grid, gbm_model_grid, drf_model_grid)

# Train a stacked ensemble using the GBM grid
glm_ensemble <- h2o.stackedEnsemble(x = predictors,
                                    y = response,
                                    metalearner_algorithm = "glm",
                                    training_frame = train,
                                    validation_frame = valid,
                                    base_models = ensemble_models)

gbm_ensemble <- h2o.stackedEnsemble(x = predictors,
                                    y = response,
                                    metalearner_algorithm = "gbm",
                                    training_frame = train,
                                    validation_frame = valid,
                                    base_models = ensemble_models)


drf_ensemble <- h2o.stackedEnsemble(x = predictors,
                                    y = response,
                                    metalearner_algorithm = "drf",
                                    training_frame = train,
                                    validation_frame = valid,
                                    base_models = ensemble_models)

ann_ensemble <- h2o.stackedEnsemble(x = predictors,
                                    y = response,
                                    metalearner_algorithm = "deeplearning",
                                    training_frame = train,
                                    validation_frame = valid,
                                    base_models = ensemble_models)

# R squared
h2o.r2(glm_ensemble, valid=TRUE)
h2o.r2(gbm_ensemble, valid=TRUE)
h2o.r2(drf_ensemble, valid=TRUE)
h2o.r2(ann_ensemble, valid=TRUE)

# save the model
#h2o.saveModel(object = glm_ensemble, path = getwd(), force = TRUE)
model_path = "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\model_path\\StackedEnsemble_model_R_1598374652612_1"

# load the model
saved_model <- h2o.loadModel(model_path)
h2o.r2(saved_model, valid=TRUE)




h2o.varimp(best_drf)



# convert feature variables to a data frame - tibble is also a data frame 
x_valid <- as.data.frame(valid) %>% select(-mean_claim) %>% as_tibble()

# change response variable to a numeric binary vector
y_valid <- as.vector(as.numeric(as.character(valid$mean_claim)))



###########
## DALEX ##
###########

# The explain() function
# The first step of using the DALEX package 
# is to wrap-up the black-box model with 
# meta-data that unifies model interfacing.

custom_predict <- function(model, newdata)  {
  newdata_h2o <- as.h2o(newdata)
  res <- as.data.frame(h2o.predict(model, newdata_h2o))
  return(as.numeric(res$predict))
}

library(DALEX)

explainer_h2o_gbm_ensemble <- DALEX::explain(model = gbm_ensemble, 
                             data = x_valid,  
                             y = y_valid,
                             predict_function = custom_predict,
                             label = "h2o glm",
                             colorize = FALSE)

explainer_h2o_gbm <- DALEX::explain(model = best_gbm, 
                             data = x_valid,  
                             y = y_valid,
                             predict_function = custom_predict,
                             label = "h2o gbm",
                             colorize = FALSE)

explainer_h2o_drf <- DALEX::explain(model = best_drf, 
                             data = x_valid,  
                             y = y_valid,
                             predict_function = custom_predict,
                             label = "h2o drf",
                             colorize = FALSE)

explainer_h2o_automl <- DALEX::explain(model = saved_model, 
                                data = x_valid,  
                                y = y_valid,
                                predict_function = custom_predict,
                                label = "h2o automl",
                                colorize = FALSE)

# Model performance
# Function model_performance() calculates predictions and residuals 
# for validation dataset.

mp_h2o_gbm_ensemble <- DALEX::model_performance(explainer_h2o_gbm_ensemble)
mp_h2o_gbm <- model_performance(explainer_h2o_gbm)
mp_h2o_drf <- model_performance(explainer_h2o_drf)
mp_h2o_automl <- model_performance(explainer_h2o_automl)

plot(mp_h2o_gbm_ensemble)
plot(mp_h2o_glm, mp_h2o_automl, geom = "boxplot")


outlierTest(lm(mean_claim ~ ., data=ml_df))


#########################
## Variable importance ##
#########################

# Using the DALEX package 
# we are able to better understand which variables are important.

vi_h2o_glm <- ingredients::feature_importance(explainer_h2o_gbm_ensemble, type="difference")
vi_h2o_gbm <- ingredients::feature_importance(explainer_h2o_gbm, type="difference")
vi_h2o_gbm <- ingredients::feature_importance(explainer_h2o_drf, type="difference")
vi_h2o_automl <- ingredients::feature_importance(explainer_h2o_automl, type="difference")
plot(vi_h2o_glm, vi_h2o_gbm, vi_h2o_automl)

# Length of the interval coresponds to a variable importance. 
# Longer interval means larger loss, so the variable is more important.

#############################
## Partial Dependence Plot ##
#############################

# Partial Dependence Plots (PDP) are one of the most popular methods 
# for exploration of the relation between a continuous variable 
# and the model outcome.

pdp_h2o_glm <- DALEX::variable_effect(explainer_h2o_glm, variable = "mean_claim")
pdp_h2o_gbm <- DALEX::variable_effect(explainer_h2o_gbm, variable = "mean_claim")
pdp_h2o_drf <- DALEX::variable_effect(explainer_h2o_drf, variable = "mean_claim")
pdp_h2o_automl <- DALEX::variable_effect(explainer_h2o_automl, variable = "mean_claim")

plot(pdp_h2o_glm, pdp_h2o_gbm, pdp_h2o_drf, pdp_h2o_automl)


###################################
## Acumulated Local Effects plot ##
###################################

# Acumulated Local Effects (ALE) plot is 
# the extension of PDP, 
# that is more suited for highly correlated variables.

pdp_h2o_glm <- DALEX::variable_effect(explainer_h2o_glm, variable = c("area","mean_policy"), type="accumulated_dependency")
pdp_h2o_gbm <- DALEX::variable_effect(explainer_h2o_gbm, variable = "mean_policy", type="accumulated_dependency")
pdp_h2o_automl <- DALEX::variable_effect(explainer_h2o_automl, variable = c("area","mean_policy"), type="accumulated_dependency")

plot(pdp_h2o_glm, pdp_h2o_automl)


mpp_h2o_glm <- DALEX::variable_effect(explainer_h2o_glm, variable = "district", type = "factor")
mpp_h2o_gbm <- DALEX::variable_effect(explainer_h2o_gbm, variable = "district", type = "factor")
mpp_h2o_automl <- DALEX::variable_effect(explainer_h2o_automl, variable = "district", type = "factor")

plot(mpp_h2o_glm, mpp_h2o_gbm, mpp_h2o_automl)

DALEX::predict_parts_break_down_interactions(explainer_h2o_automl, new_observation = valid)


### VIF

###########################
## VIF FEATURE SELECTION ##
###########################
# VIF
# training.samples <- IN_mod$mean_claim %>%
#   createDataPartition(p = 0.8, list = FALSE)
# train.data.index  <- IN_mod[training.samples, ]
# test.data.index <- IN_mod[-training.samples, ]
# 
# train.data <- train.data.index[ , !(names(train.data.index) %in% c("subwatershed"))]
# test.data <- test.data.index[ , !(names(test.data.index) %in% c("subwatershed"))]
# 
# # Build the model
# model1 <- lm(mean_claim ~., data = train.data)
# # Make predictions
# predictions <- model1 %>% predict(test.data)
# # Model performance
# data.frame(
#   RMSE = RMSE(predictions, test.data$mean_claim),
#   R2 = R2(predictions, test.data$mean_claim)
# )
# 
# vif_model1 <- car::vif(model1)
# sort(vif_model1)
# 
# # new_cols <- names(vif_model1[which( vif_model1 < 10 )])
# new_cols <- names(vif_model1[which(!names(vif_model1) %in% c("subwatershed",
#                                                              "orb100yr06h","orb100yr12h",
#                                                              "orb25yr06h","orb25yr12h","orb25yr24h",
#                                                              "orb2yr06h","orb2yr12h","orb2yr24h",
#                                                              "orb50yr06h","orb50yr12h","orb50yr24h",
#                                                              "orb100yr06ha_am","orb100yr12ha_am","orb100yr24ha_am",
#                                                              "orb25yr06ha_am","orb25yr12ha_am","orb25yr24ha_am",
#                                                              "orb2yr06ha_am","orb2yr12ha_am","orb2yr24ha_am",
#                                                              "orb50yr06ha_am","orb50yr12ha_am","orb50yr24ha_am"))])
# #
# #
# new_cols <- append(new_cols, "mean_claim")
# 
# ############################
# 
# vif.train.data.index  <- IN_mod[training.samples, new_cols]
# vif.test.data.index <- IN_mod[-training.samples, new_cols]
# 
# vif.train.data <- vif.train.data.index[ , !(names(vif.train.data.index) %in% c("subwatershed"))]
# vif.test.data <- vif.test.data.index[ , !(names(vif.test.data.index) %in% c("subwatershed"))]
# 
# # Build the model
# vif.model <- lm(mean_claim ~., data = vif.train.data)
# # Make predictions
# vif.predictions <- vif.model %>% predict(vif.test.data)
# # Model performance
# data.frame(
#   RMSE = RMSE(vif.predictions, vif.test.data$mean_claim),
#   R2 = R2(vif.predictions, vif.test.data$mean_claim)
# )
# 
# vif.vif_model <- car::vif(vif.model)
# sort(vif.vif_model)
# new_names <- names(vif.vif_model)
# vif_names <- append(new_names, "mean_claim")

############
## MODEL ###
############




#
# currentData <- model_data
# linear_reg <- lm(mean_claim ~ . ,data=currentData)
# 
# outlierResult <-outlierTest(linear_reg)
# outlierResult
# exclusionRows <-names(outlierResult[[1]])
# inclusionRows <- !(rownames(currentData) %in% exclusionRows)
# currentData <- currentData[inclusionRows,]
# 
# library(dplyr)
# model_data <- filter(currentData, mean_claim > 1 )
