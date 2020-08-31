rm(list=ls())
options(scipen=999)
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
library(sparklyr)
library(tidyverse)
library(caret)
h2o.shutdown()
# data
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\"
setwd(dsn)
IN_df <- read.xlsx(paste0(dsn,"MOD_DF_08_21.xlsx"))

IN_mod <- IN_df[ , !(names(IN_df) %in% c("subwatershed"))]
IN_mod <- IN_mod %>% mutate_if(is.character,as.numeric) 

# calulate the correlations
r <- cor(IN_mod[,1:20], use="complete.obs")
round(r,2)
ggcorrplot(r)

### VIF
training.samples <- IN_mod$mean_claim %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data.index  <- IN_mod[training.samples, ]
test.data.index <- IN_mod[-training.samples, ]

train.data <- train.data.index[ , !(names(train.data.index) %in% c("subwatershed"))]
test.data <- test.data.index[ , !(names(test.data.index) %in% c("subwatershed"))]

# Build the model
model1 <- lm(mean_claim ~., data = train.data)
# Make predictions
predictions <- model1 %>% predict(test.data)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test.data$mean_claim),
  R2 = R2(predictions, test.data$mean_claim)
)

vif_model1 <- car::vif(model1)
sort(vif_model1)

# new_cols <- names(vif_model1[which( vif_model1 < 10 )])
new_cols <- names(vif_model1[which(!names(vif_model1) %in% c("subwatershed",
                                                             "orb100yr06h","orb100yr12h",                 
                                                             "orb25yr06h","orb25yr12h","orb25yr24h",                  
                                                             "orb2yr06h","orb2yr12h","orb2yr24h",                
                                                             "orb50yr06h","orb50yr12h","orb50yr24h",                  
                                                             "orb100yr06ha_am","orb100yr12ha_am","orb100yr24ha_am",            
                                                             "orb25yr06ha_am","orb25yr12ha_am","orb25yr24ha_am",          
                                                             "orb2yr06ha_am","orb2yr12ha_am","orb2yr24ha_am",        
                                                             "orb50yr06ha_am","orb50yr12ha_am","orb50yr24ha_am",
                                                             "flow_0_exceedence_prob","flow_0.1_exceedence_prob",    
                                                             "flow_1_exceedence_prob","flow_10_exceedence_prob",
                                                             "lu_21_area","lu_22_area","lu_23_area",                  
                                                             "lu_24_area","lu_41_area","lu_82_area", 
                                                             "avg_slope",
                                                             "relief", 
                                                             "population", 
                                                             "ruggedness", 
                                                             "population_density", 
                                                             "perimeter",
                                                             "elongation_ratio"

))])

mod_lm_cols <- c("circulatory_ratio",
"relief_ratio",
"water_bodies_area",
"dams_count",
"population_change",
"dependent_population_pct",
"levee_low_crest_elevation_m",
"flow_50_exceedence_prob", 
"mean_policy")

# new_cols <- names(vif_model1[which( vif_model1 < 10 )])
new_cols <- names(IN_df[which(names(IN_df) %in% mod_lm_cols)])
new_cols <- append(new_cols, "mean_claim")

############################

vif.train.data.index  <- IN_mod[training.samples, new_cols]
vif.test.data.index <- IN_mod[-training.samples, new_cols]

vif.train.data <- vif.train.data.index[ , !(names(vif.train.data.index) %in% c("subwatershed"))]
vif.test.data <- vif.test.data.index[ , !(names(vif.test.data.index) %in% c("subwatershed"))]

# Build the model
vif.model <- lm(mean_claim ~., data = vif.train.data)
# Make predictions
vif.predictions <- vif.model %>% predict(vif.test.data)
# Model performance
data.frame(
  RMSE = RMSE(vif.predictions, vif.test.data$mean_claim),
  R2 = R2(vif.predictions, vif.test.data$mean_claim)
)

vif.vif_model <- car::vif(vif.model)
sort(vif.vif_model)
new_names <- names(vif.vif_model)
a_new_names <- append(new_names, "mean_claim")

############
## MODEL ###
############

IN_mod <- IN_mod %>% mutate_if(is.character,as.numeric) 
df <- IN_mod[complete.cases(IN_mod), new_cols]

str(df)

# Start local host with given number of threads plus give memory size
h2o.init(max_mem_size = '16G')

# Define response and predictors #
response <- "mean_claim"
predictors <- names(df[,which(!names(df) %in% c("mean_claim", "subwatershed"))])
predictors

###############
## NORMALIZE ##
###############
# IN_norm <- normalizeData(df, type='0_1')
# colnames(IN_norm) <- names(df)
# IN_norm_x <- normalizeData(df[,predictors], type='0_1')
# IN_norm_y <- normalizeData(df[,response], type='0_1')
# 
# IN_norm_index <- as.data.frame(IN_norm)
# IN_norm_index$subwatershed <- df$subwatershed
# head(IN_norm_index)
# convert into H2O frame
IN_h2o <- as.h2o(df)



## Splits datasets into train, valid and test
splits <- h2o.splitFrame(data=IN_h2o, ratios=c(0.8,0.1), seed=1236)
names(splits) <- c("train","valid", "test")

train <- splits$train
valid <- splits$valid
test <- splits$test
# dim(test)

# write.xlsx(train, "P:\\Temp\\jMatney\\IndianaML\\AutoML\\train.xlsx")
# write.xlsx(test, "P:\\Temp\\jMatney\\IndianaML\\AutoML\\test.xlsx")

# Identify predictors and response
y <- response
x <- predictors


# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

# Train the DRF model
cars_drf <- h2o.deeplearning(x = x, y = y,
                             training_frame = train, 
                             validation_frame = valid,
                             seed = 1234)

h2o.r2(cars_drf, valid=TRUE)

# Histogram with density plot
ggplot(df, aes(x=mean_claim)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666") 

h2o.performance(cars_drf, valid=TRUE)

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
ntrees = seq(1,1000,100)
max_depth = seq(1,1000,100)

drf_params = list(ntrees = ntrees, max_depth = max_depth)

## GLM 
alpha=seq(0.01, 1, 0.01)
lambda=seq(0.00000001,0.0001, 0.000001)

glm_params = list(alpha = alpha,
                  lambda = lambda)

## DEEP LEARNING 
activation = c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout")
hidden = list(c(5, 5, 5, 5, 5), c(10, 10, 10, 10), c(50, 50, 50), c(100, 100, 100))
epochs = c(50, 100, 200)
l1 = c(0, 0.00001, 0.0001)
l2 = c(0, 0.00001, 0.0001)
rate = c(0, 0.1, 0.005, 0.001)
rate_annealing = c(1e-8, 1e-7, 1e-6)
rho = c(0.9, 0.95, 0.99, 0.999)
epsilon = c(1e-10, 1e-8, 1e-6, 1e-4)
momentum_start = c(0, 0.5)
momentum_stable = c(0.99, 0.5, 0)
input_dropout_ratio = c(0, 0.1, 0.2)
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

search_criteria_max_runtime = list(strategy = "RandomDiscrete",
                                   max_runtime_secs = 600,
                                   max_models = 100,
                                   stopping_metric = "RMSE",
                                   stopping_tolerance = 0.001,
                                   stopping_rounds = 20, seed = 1)


# DRF hyperparameters
hyper_params_drf <- 
  list(
    mtries                   = seq(2, 5, by = 1), 
    sample_rate              = c(0.65, 0.8, 0.95),
    col_sample_rate_per_tree = c(0.5, 0.9, 1.0),
    max_depth                = seq(1, 30, by = 3),
    min_rows                 = c(1, 2, 5, 10)
  )

# GBM hyperparameters
hyper_params_gbm <- 
  list(
    learn_rate               = c(0.01, 0.1),
    sample_rate              = c(0.65, 0.8, 0.95),
    col_sample_rate_per_tree = c(0.5, 0.9, 1.0),
    max_depth                = seq(1, 30, by = 3),
    min_rows                 = c(1, 2, 5, 10)
  )

# elastic net model 
glm_model <- 
  h2o.glm(
    x               = x,
    y               = y, 
    training_frame  = train,
    validation_frame = valid,
    balance_classes = TRUE,
    nfolds          = 10,
    seed            = 1975
  )

glm_grid <- h2o.grid(algorithm = "glm",
                     grid_id = "glm_grid",
                     x = x,
                     y = y,
                     training_frame = train,
                     validation_frame = valid,
                     seed = 1,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = glm_params,
                     search_criteria = search_criteria_max_runtime)
# random forest model
drf_model_grid <- 
  h2o.grid(
    algorithm       = "randomForest", 
    x               = x, 
    y               = y,
    training_frame  = train,
    validation_frame = valid,
    balance_classes = TRUE, 
    nfolds          = 10,
    ntrees          = 1000,
    grid_id         = "drf_grid",
    hyper_params    = hyper_params_drf,
    search_criteria = search_criteria_all,
    seed            = 1975
  )

# gradient boosting machine model
gbm_model_grid <- 
  h2o.grid(
    algorithm       = "gbm",
    x               = x, 
    y               = y,
    training_frame  = train,
    validation_frame = valid,
    balance_classes = TRUE, 
    nfolds          = 10,
    ntrees          = 1000,
    grid_id         = "gbm_grid",
    hyper_params    = hyper_params_gbm,
    search_criteria = search_criteria_all,
    seed            = 1975
  )


# mod1 <- h2o.randomForest( x = x, y = y,
#                        training_frame = train,
#                        validation_frame = valid,
#                        seed = 1,
#                        nfolds = nfolds,
#                        max_depth = 100,
#                        ntrees=500)
# 
# h2o.r2(mod1, valid=TRUE)

# convert feature variables to a data frame - tibble is also a data frame 
x_valid <- as.data.frame(valid) %>% select(-mean_claim) %>% as_tibble()

# change response variable to a numeric binary vector
y_valid <- as.vector(as.numeric(as.character(valid$mean_claim)))


test_backtransform <- round(as.data.frame(denormalizeData(test$mean_claim, getNormParameters(IN_norm_y))),0)
# Generate predictions on a test set (if neccessary)
best_drf_pred <- h2o.predict(mod1, newdata = test)

# create custom predict function
pred <- function(model, newdata)  {
  results <- as.data.frame(h2o.predict(model, newdata %>% as.h2o()))
  return(results[[3L]])
}


# generalised linear model explainer
explainer_glm <- explain(
  model            = glm_model, 
  type             = "classification",
  data             = x_valid,
  y                = y_valid,
  predict_function = pred,
  label            = "h2o_glm"
)

# random forest model explainer
explainer_drf <- explain(
  model            = drf_model, 
  type             = "classification",
  data             = x_valid,
  y                = y_valid,
  predict_function = pred,
  label            = "h2o_drf"
)

# gradient boosting machine explainer
explainer_gbm <- explain(
  model            = gbm_model, 
  type             = "classification",
  data             = x_valid,
  y                = y_valid,
  predict_function = pred,
  label            = "h2o_gbm"
)

summary(best_drf_pred)
best_drf_pred_denorm <- as.data.frame(denormalizeData(best_drf_pred, getNormParameters(IN_norm_y)))
best_drf_results <- as.data.frame(cbind(test_backtransform, round(best_drf_pred_denorm,0)))
colnames(best_drf_results) <- c("claims", "predicted")

# compute residuals
residuals = best_drf_results$predicted - best_drf_results$claims
resids=as.data.frame(residuals)

# plot residuals 
compare = cbind(
  as.data.frame(best_drf_results$claims),
  as.data.frame(resids$residuals))

plot( compare[ ,1:2], xlab = "actual", ylab="residuals")

##########################################
######## GRID Search Models ##############
##########################################

glm_grid <- h2o.grid(algorithm = "glm",
                     grid_id = "glm_grid",
                     x = x,
                     y = y,
                     training_frame = train,
                     validation_frame = valid,
                     seed = 1,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = glm_params,
                     search_criteria = search_criteria_max_runtime)

drf_grid <- h2o.grid(algorithm = "drf",
                     grid_id = "drf_grid",
                     x = x,
                     y = y,
                     training_frame = train,
                     validation_frame = valid,
                     seed = 1,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = drf_params,
                     search_criteria = search_criteria_max_runtime)

gbm_grid <- h2o.grid(algorithm = "gbm",
                     grid_id = "gbm_grid",
                     x = x,
                     y = y,
                     training_frame = train,
                     validation_frame = valid,
                     seed = 1,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = gbm_params,
                     search_criteria = search_criteria)

dpl_grid <- h2o.grid(algorithm = "deeplearning",
                     grid_id = "dpl_grid",
                     x = x,
                     y = y,
                     training_frame = train,
                     validation_frame = valid,
                     seed = 1,
                     nfolds = nfolds,
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = dpl_params,
                     search_criteria = search_criteria_max_runtime)

dpl_grid



models = c(glm_grid, gbm_grid, drf_grid)

############################################
### Get the grid results, sorted by RMSE ###
############################################
glm_gridperf <- h2o.getGrid(grid_id = "glm_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)

drf_gridperf <- h2o.getGrid(grid_id = "drf_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)

gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)

dpl_gridperf <- h2o.getGrid(grid_id = "dpl_grid",
                            sort_by = "rmse",
                            decreasing = FALSE)
print(glm_gridperf)
print(gbm_gridperf)
print(drf_gridperf)
print(dpl_gridperf)


h2o.gainsLift(best_glm)


p_glm <- h2o.glm(x = x, y = y, training_frame = train,
                        validation_frame = valid,
                        lambda = 0,
                        remove_collinear_columns = TRUE,
                        compute_p_values = TRUE)

# take a look at the coefficients_table to see the p_values
a <- p_glm@model$coefficients_table
write.xlsx(a, "p_vlaues.xlsx")

mod_lm <- lm(mean_claim ~., data=df)
summary(mod_lm)

# Grab the top GLM model, chosen by RMSE
best_glm <- h2o.getModel(glm_gridperf@model_ids[[1]])
best_glm
gb

# save the model
h2o.saveModel(object = best_glm, path = "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\best_models", force = TRUE)
h2o.r2(best_glm, valid = TRUE)

# Grab the top DRF model, chosen by RMSE
best_drf <- h2o.getModel(drf_gridperf@model_ids[[1]])
best_drf
h2o.varimp_plot(best_drf, num_of_features = 20)

# save the model
h2o.saveModel(object = best_drf, path = "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\best_models", force = TRUE)
h2o.r2(best_drf, train = TRUE, valid = TRUE, xval = TRUE)
plot(best_drf)

a <- lm(mean_claim ~ ., data=df)
plot(a)

# Calculate performance measures at threshold that maximizes precision
drf.pred = h2o.predict(best_drf, test)
drf.perf = h2o.performance(drf.pred)

plot(prostate.perf, type = "cutoffs")     # Plot precision vs. thresholds
plot(prostate.perf, type = "roc")         # Plot ROC curve


# # create custom predict function
# pred <- function(model, newdata)  {
#   results <- as.data.frame(h2o.predict(model, as.h2o(newdata)))
#   return(results[[3L]])
# }
# 
# # random forest explainer
# explainer_rf <- explain(
#   model = best_drf,
#   data = test,
#   y = y,
#   predict_function = pred,
#   label = "h2o rf"
# )
# 
# resids_rf  <- model_performance(explainer_rf)


# Grab the top GBM model, chosen by RMSE
best_gbm <- h2o.getModel(gbm_gridperf@model_ids[[1]])
best_gbm
h2o.varimp_plot(best_gbm, num_of_features = 20)
# save the model
h2o.saveModel(object = best_gbm, path = "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\best_models", force = TRUE)
h2o.r2(best_gbm, train=TRUE, valid=TRUE, xval=TRUE)


# Grab the top DPL model, chosen by RMSE
best_dpl <- h2o.getModel(dl_gridperf@model_ids[[1]])
best_dpl
h2o.varimp_plot(best_dpl, num_of_features = 20)
# save the model
h2o.saveModel(object = best_dpl, path = "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\best_models", force = TRUE)
h2o.r2(best_dpl, valid = TRUE)


# Train a stacked ensemble using the GBM grid
glm_ensemble <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    metalearner_algorithm = "glm",
                                    training_frame = train,
                                    validation_frame = valid,
                                    base_models = models)

gbm_ensemble <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    metalearner_algorithm = "gbm",
                                    training_frame = train,
                                    validation_frame = valid,
                                    base_models = models)


drf_ensemble <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    metalearner_algorithm = "drf",
                                    training_frame = train,
                                    validation_frame = valid,
                                    base_models = models)

dpl_ensemble <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    metalearner_algorithm = "deeplearning",
                                    training_frame = train,
                                    validation_frame = valid,
                                    base_models = models)
# R squared

h2o.r2(glm_ensemble, valid=TRUE)
h2o.r2(drf_ensemble, valid=TRUE)
h2o.r2(gbm_ensemble, valid=TRUE)
h2o.r2(dpl_ensemble, valid=TRUE)

# nash sutcliff coefficient - looks for equal means
# compare to mean model
# how often do we get a band right / categorical variable

# how good is the best model? R2

# Compare to base learner performance on the test set
glm_ensemble_test <- h2o.performance(glm_ensemble, newdata = test)
gbm_ensemble_test <- h2o.performance(gbm_ensemble, newdata = test)
drf_ensemble_test <- h2o.performance(drf_ensemble, newdata = test)
dpl_ensemble_test <- h2o.performance(dpl_ensemble, newdata = test)


# Compare to base learner performance on the test set
perf_glm_test <- h2o.performance(h2o.getModel(glm_grid@model_ids[[1]]), newdata = test)
perf_drf_test <- h2o.performance(h2o.getModel(drf_grid@model_ids[[1]]), newdata = test)
perf_gbm_test <- h2o.performance(h2o.getModel(gbm_grid@model_ids[[1]]), newdata = test)
perf_dpl_test <- h2o.performance(h2o.getModel(dpl_grid@model_ids[[1]]), newdata = test)

baselearner_best_rmse_test <- min(h2o.rmse(perf_glm_test), 
                                  h2o.rmse(perf_drf_test), 
                                  h2o.rmse(perf_gbm_test), 
                                  h2o.rmse(perf_dpl_test))

ensemble_rmse_test <- h2o.rmse(dpl_ensemble_test)
print(sprintf("Best Base-learner Test RMSE:  %s", baselearner_best_rmse_test))
print(sprintf("Ensemble Test RMSE:  %s", ensemble_rmse_test))

# Generate predictions on a test set (if neccessary)
glm_ensemble_pred <- h2o.predict(glm_ensemble, newdata = test)

summary(glm_ensemble_pred)
glm_ensemble_pred_denorm <- as.data.frame(denormalizeData(glm_ensemble_pred, getNormParameters(IN_norm_y)))
test_backtransform <- round(as.data.frame(denormalizeData(test$mean_claim, getNormParameters(IN_norm_y))),0)
glm_ensemble_results <- as.data.frame(cbind(test_backtransform, round(glm_ensemble_pred_denorm,0)))
colnames(glm_ensemble_results) <- c("claims", "predicted")

lares::mplot_full(tag = glm_ensemble_results$claims,
                  score = glm_ensemble_results$predicted,
                  splits = 10,
                  subtitle = "Ensemble GLM Metalearner Results",
                  model_name = "simple_model_02",
                  save = T)

# Generate predictions on a test set (if neccessary)
best_drf_pred <- h2o.predict(best_drf, newdata = test)

summary(best_drf_pred)
best_drf_pred_denorm <- as.data.frame(denormalizeData(best_drf_pred, getNormParameters(IN_norm_y)))
best_drf_results <- as.data.frame(cbind(test_backtransform, round(best_drf_pred_denorm,0)))
colnames(best_drf_results) <- c("claims", "predicted")

lares::mplot_full(tag = best_drf_results$claims,
                  score = best_drf_results$predicted,
                  splits = 10,
                  subtitle = "Best DRF Results",
                  model_name = "simple_model_02",
                  save = T)

# IN_test_observed <- IN_df[,c("mean_claim","subwatershed")]
# 
# model_results <- results %>% left_join(IN_test_observed, by="subwatershed")
# colnames(model_results) <- c("subwatershed", "predicted", "observed")
# head(model_results)

# plot(results$claims, results$predicted, xlim=c(0,50000))
# abline(x=y,col="blue")
# 
# library(ggplot2)
# 
# # compute predicted values on our test dataset
# pred <- h2o.predict(best_drf, newdata = test)
# 
# # extract the true 'mpg' values from our test dataset
# actual <- test$mean_claim
# 
# # produce a data.frame housing our predicted + actual 'mpg' values
# data <- data.frame(
#   predicted = pred,
#   actual    = actual
# )
# # a bug in data.frame does not set colnames properly; reset here 
# names(data) <- c("predicted", "actual")
# 
# # plot predicted vs. actual values
# ggplot(best_drf_results, aes(x = claims, y = predicted)) +
#   geom_abline(lty = "dashed", col = "red") +
#   geom_point() +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   coord_fixed(ratio = 1) +
#   labs(
#     x = "Actual",
#     y = "Predicted",
#     title = "Predicted vs. Actual"
#   )

  sc <- spark_connect(master = "local")
  iris_tbl <- sdf_copy_to(sc, iris, name = "iris_tbl", overwrite = TRUE)
  
  partitions <- iris_tbl %>%
    sdf_random_split(training = 0.7, test = 0.3, seed = 1111)
  
  iris_training <- partitions$training
  iris_test <- partitions$test
  
  rf_model <- iris_training %>%
    ml_random_forest(Species ~ ., type = "classification")
  
  pred <- ml_predict(rf_model, iris_test)
  
  ml_multiclass_classification_evaluator(pred)
