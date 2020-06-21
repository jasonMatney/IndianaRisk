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

# data
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\model\\"
setwd(dsn)
IN_df <- read.xlsx(paste0(dsn,"IN_Risk_Model.xlsx"))
head(IN_df)

IN_mod <- IN_df[ , !(names(IN_df) %in% c("subwatershed"))]

IN_mod <- IN_mod %>% mutate_if(is.character,as.numeric) 

# Start local host with given number of threads plus give memory size
h2o.init(ip='localhost', port=54321, nthreads=-1, max_mem_size = '20g')

###############
## NORMALIZE ##
###############
IN_norm <- normalizeData(IN_mod, type='0_1')
colnames(IN_norm) <- names(IN_mod)
IN_norm_y <- normalizeData(IN_mod$claims_total_building_insurance_coverage_avg, type='0_1')

# convert into H2O frame
IN_h2o <- as.h2o(IN_norm)

## Splits datasets into train, valid and test
splits <- h2o.splitFrame(data=IN_h2o, ratios=c(0.8), seed=1236)
names(splits) <- c("train","test")#valid","test")

train <- splits$train
#valid <- splits$valid
test <- splits$test

# Define response and predictors #
response <- "claims_total_building_insurance_coverage_avg"
predictors <- names(IN_mod[,which(!names(IN_mod) %in% c("claims_total_building_insurance_coverage_avg"))])
gc()


names(train) <- names(IN_mod)
#names(valid) <- names(IN_mod)
names(test) <- names(IN_mod)


###################
## Random Forest ##
###################

## run our random forest  model
drf_IN_22c <- h2o.randomForest(        ## h2o.randomForest function
  model_id = "rf_IN_22c",             ## name the model in H2O
  training_frame = train,        ## the H2O frame for training
  #validation_frame = valid,      ## the H2O frame for validation (not required)
  x=predictors,                  ## the predictor columns, by column index
  y=response,                    ## the target index (what we are predicting)
  nfolds=5,                     ## CROSS-VALIDATION
  
  ##   not required, but helps use Flow
  ntrees = 2000,                  ## use a maximum of 200 trees
  max_depth = 10,
  stopping_metric = "MSE",
  stopping_rounds = 10,
  keep_cross_validation_predictions = T,
  stopping_tolerance = 0.05,     ##
  score_each_iteration = T,      ## Predict against training and validation for
  ## each tree. Default will skip several. fold_assignment = stratified,
  seed = 111)                    ## Set the random seed so that this can be reproduced.

summary(drf_IN_22c)

# Variable Importance
h2o.varimp_plot(drf_IN_22c)

drf_performance <- h2o.performance(drf_IN_22c, test)
drf_performance

##############################
## Generalized Linear Model ##
##############################


## run our second predictive model
glm_IN_22c <- h2o.glm(family= "gaussian",
                      x= predictors,
                      y=response,
                      training_frame = train,
                      #stopping_metric = "MSE",
                      #validation_frame = valid, 
                      nfolds=5, # Cross-Validation
                      seed = 111,
                      lambda = 0,
                      early_stopping = TRUE) # Early stopping               

summary(glm_IN_22c)


# Variable Importance
h2o.varimp_plot(glm_IN_22c)

glm_performance <- h2o.performance(glm_IN_22c, test)
glm_performance

#######################
## Gradient Boosting ##
#######################

# gradient boosting machine model
gbm_IN_22c <-  h2o.gbm(
  x = predictors, 
  y = response,
  training_frame = train,
  nfolds = 5,
  #validation_frame = valid, 
  stopping_metric = "MSE",
  stopping_rounds = 10,
  keep_cross_validation_predictions = T,
  stopping_tolerance = 0.05,     ##
  score_each_iteration = T,  
  seed = 123)


# Variable Importance
h2o.varimp_plot(gbm_IN_22c)

h2o.performance(gbm_IN_22c, test)

gbm_performance <- h2o.performance(gbm_IN_22c, test)
gbm_performance

#################
## Keras Model ##
#################

y <- "claims_total_building_insurance_coverage_avg"

# Build X_train, y_train, X_test, y_test
x_train <- as.matrix(dplyr::select(as.data.frame(train), -c(y)))
y_train <- as.matrix(dplyr::select(as.data.frame(train), c(y)))

x_test <- as.matrix(dplyr::select(as.data.frame(test), -c(y)))
y_test <- as.matrix(dplyr::select(as.data.frame(test), c(y)))

# x_val <- as.matrix(dplyr::select(as.data.frame(valid), -c(y)))
# y_val <- as.matrix(dplyr::select(as.data.frame(valid), c(y)))


# # # Clean slate - just in case the cluster was already running
# h2o.removeAll()
# # 
# # # Cluster Info
# h2o.clusterInfo()
# # 
# # # CLuster Status
# h2o.clusterStatus()
# require("Metrics")
# 
# root_mean_squared_error <- function(y_true, y_pred){ return(rmse(y_pred, y_true) ) } 

# model.compile(optimizer = "rmsprop", loss = root_mean_squared_error, 
#               metrics =["accuracy"])



######################
## Create the model ##
######################
 
build_model <- function(){ 
  
  # add dropout layers
  # add early stopping 
  model <- keras_model_sequential() %>%
    
    # First hidden layer
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu", 
      input_shape        = ncol(x_train))  %>%
    
    # Dropout to prevent overfitting
    #layer_dropout(rate = 0.1) %>%
    # 
    # Second hidden layer
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu") %>% 
    
    # Dropout to prevent overfitting
    #layer_dropout(rate = 0.1) %>%
    # 
    # Third hidden layer
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu") %>% 
    
    # Dropout to prevent overfitting
    #layer_dropout(rate = 0.1) %>%
    # 
    # Output layer
    layer_dense(
      units              = 1, 
      kernel_initializer = "uniform", 
      activation         = "sigmoid")
  
  model %>% compile(
    loss = "mse", #root_mean_squared_error,
    optimizer = optimizer_rmsprop(),
    # metrics <- list(mape = "mean_absolute_percentage_error", mse = "mean_squared_error", rmse_metric)
    model %>% summary()
    
    ##################### list("mean_absolute_error")
  )
  
  model
}

model <- build_model()

#####################
## Train the model ##
#####################

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

epochs <- 150
batch_size <- 32

# Fit the model and store training stats
history <- model %>% fit(
  x = x_train,
  y = y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.1,
  #validation_data = list(x_val, y_val),
  verbose = 1,
  callbacks = list(print_dot_callback,
                   callback_early_stopping(monitor = "val_loss", 
                                           min_delta = 0.05,
                                           patience = 4, 
                                           verbose = 1, 
                                           mode = c("auto", "min", "max"),
                                           baseline = NULL, 
                                           restore_best_weights = FALSE)))

summary(model)
plot(history)

model %>% evaluate(x_test, y_test)

IN_mat <- rbind(x_train, x_test)
dim(IN_mat)


# # Keras Model
# dpl_pred <- model %>% predict(IN_mat)

###############################################################################
####################
## Ensemble Model ##
####################
###############################################################################

names(IN_h2o) <- names(IN_mod)

# H2O models
glm_pred <- h2o.predict(glm_IN_22c, as.h2o(x_test))
drf_pred <- h2o.predict(drf_IN_22c, as.h2o(x_test))
gbm_pred <- h2o.predict(gbm_IN_22c, as.h2o(x_test))

# Keras Model
dpl_pred <- model %>% predict(x_test)

###################
### DENORMALIZE ###
###################

glm_pred_df <- as.data.frame(denormalizeData(glm_pred, getNormParameters(IN_norm_y)))
drf_pred_df <- as.data.frame(denormalizeData(drf_pred, getNormParameters(IN_norm_y)))
gbm_pred_df <- as.data.frame(denormalizeData(gbm_pred, getNormParameters(IN_norm_y)))
dpl_pred_df <- as.data.frame(denormalizeData(dpl_pred, getNormParameters(IN_norm_y)))
y_test_denorm <- as.data.frame(denormalizeData(y_test, getNormParameters(IN_norm_y)))

# glm_pred_df$HC12 <- IN_df$subwatershed
# drf_pred_df$HC12 <- IN_df$subwatershed
# gbm_pred_df$HC12 <- IN_df$subwatershed
# dpl_pred_df$HC12 <- IN_df$subwatershed

#############################
## MAKE ENSEMBLE DATAFRAME ##
#############################


ensemble_df <- as.data.frame(cbind(y_test_denorm, 
                                   glm_pred_df, 
                                   drf_pred_df, 
                                   gbm_pred_df, 
                                   dpl_pred_df))
names(ensemble_df) <- c("observed","glm","drf","gbm","dpl")
head(ensemble_df)

## NORMALIZE ##
ensemble_norm <- normalizeData(ensemble_df, type='0_1')

# dpl_results <- as.data.frame(cbind(y_test_denorm, dpl_pred_df))
# 
# colnames(dpl_results) <- c("Observed","Predicted")

ggplot(aes(observed, glm), data=ensemble_df) + 
  geom_point() +
  xlim(0, 100000) + 
  ylim(0, 100000) + geom_abline(intercept = 0, slope = 1) #(aes(observed,drf))


lares::mplot_full(tag = ensemble_df$observed,
                    score = ensemble_df$dpl,
                    splits = 10,
                    subtitle = "Deep Learning",
                    model_name = "simple_model_02",
                    save = T)

lares::mplot_full(tag = ensemble_df$observed,
                  score = ensemble_df$drf,
                  splits = 10,
                  subtitle = "Random Forest",
                  model_name = "simple_model_02",
                  save = T)

lares::mplot_full(tag = ensemble_df$observed,
                  score = ensemble_df$gbm,
                  splits = 10,
                  subtitle = "GBM Results",
                  model_name = "simple_model_02",
                  save = T)

lares::mplot_full(tag = ensemble_df$observed,
                  score = ensemble_df$glm,
                  splits = 10,
                  subtitle = "GLM Results",
                  model_name = "simple_model_02",
                  save = T)

ensemble_df$resid_glm <- ensemble_df$observed - ensemble_df$glm
ensemble_df$resid_drf <- ensemble_df$observed - ensemble_df$drf
ensemble_df$resid_gbm <- ensemble_df$observed - ensemble_df$gbm
ensemble_df$resid_dpl <- ensemble_df$observed - ensemble_df$dpl

ggplot(aes(x=glm, y=resid_glm), data=ensemble_df) + 
  geom_point() +
  xlim(0, 100000) + 
  ylim(0, 100000) + geom_abline(intercept = 0, slope = 1) #(aes(observed,drf))

ggplot(ensemble_df, aes(x=glm, y=resid_glm)) + 
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=4)

p1 <- boxplot(ensemble_df$resid_glm, main = "GLM")
p2 <- boxplot(ensemble_df$resid_drf, main = "DRF")
p3 <- boxplot(ensemble_df$resid_gbm, main = "GBM")
p4 <- boxplot(ensemble_df$resid_dpl, main = "DPL")

# require(ggplot2)
# require(gridExtra)
# grid.arrange(p1, p2, p3, p4, 
#              layout_matrix = rbind(c(1,1,1),c(2,3,4)))
# 
# 
# ggplot(data = ensemble_df, aes(x=observed, y=resid_glm)) + 
#   geom_boxplot() + facet_wrap(~resid_glm, ncol = 4)
# 
# p <- ggplot(data = ensemble_df, aes(x=variable, y=value)) + geom_boxplot(aes(fill=Label))

#########################
## Split Ensemble data ##
#########################

## Splits datasets into train, valid and test
ensemble_h2o <- as.h2o(ensemble_norm)

## Reduce this to 70 train - add between 5 and 10-fold cross validation 

ensemble_splits <- h2o.splitFrame(data=ensemble_h2o, ratios=c(0.8), seed=1236)
names(ensemble_splits) <- c("train","test")

ensemble_train <- ensemble_splits$train
#ensemble_valid <- ensemble_splits$valid
ensemble_test <- ensemble_splits$test
names(ensemble_train) <- c("observed","glm","drf","gbm","dpl")
#names(ensemble_valid) <- c("observed","glm","drf","gbm","dpl")
names(ensemble_test) <- c("observed","glm","drf","gbm","dpl")


obsv <- "observed"

# Build X_train, y_train, X_test, y_test
ensemble_x_train <- as.matrix(dplyr::select(as.data.frame(ensemble_train), -c(obsv)))
ensemble_y_train <- as.matrix(dplyr::select(as.data.frame(ensemble_train), c(obsv)))

ensemble_x_test <- as.matrix(dplyr::select(as.data.frame(ensemble_test), -c(obsv)))
ensemble_y_test <- as.matrix(dplyr::select(as.data.frame(ensemble_test), c(obsv)))

# ensemble_x_val <- as.matrix(dplyr::select(as.data.frame(ensemble_valid), -c(obsv)))
# ensemble_y_val <- as.matrix(dplyr::select(as.data.frame(ensemble_valid), c(obsv)))

###############################
## Create the ensemble model ##
###############################

ensemble_build_model <- function(){ 
  
  # add dropout layers
  # add early stopping 
  ensemble_model <- keras_model_sequential() %>%
    
    # First hidden layer
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu", 
      input_shape        = ncol(ensemble_x_train))  %>%
    
    # Dropout to prevent overfitting
    layer_dropout(rate = 0.1) %>%
    # 
    # Second hidden layer
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu") %>% 
    
    # Dropout to prevent overfitting
    layer_dropout(rate = 0.1) %>%
    # 
    # Third hidden layer
    layer_dense(
      units              = 16, 
      kernel_initializer = "uniform", 
      activation         = "relu") %>% 
    
    # Dropout to prevent overfitting
    layer_dropout(rate = 0.1) %>%
    # 
    # Output layer
    layer_dense(
      units              = 1, 
      kernel_initializer = "uniform", 
      activation         = "sigmoid")
  
  ensemble_model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    ensemble_model %>% summary()
    
    ##################### list("mean_absolute_error")
  )
  
  ensemble_model
}

ensemble_model <- ensemble_build_model()

#####################
## Train the model ##
#####################

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

ensemble_epochs <- 150
ensemble_batch_size <- 32

# Fit the model and store training stats
ensemble_history <- ensemble_model %>% fit(
    x = ensemble_x_train,
    y = ensemble_y_train,
    epochs = ensemble_epochs,
    batch_size = ensemble_batch_size,
    #validation_data = list(ensemble_x_val, ensemble_y_val),
    validation_split = 0.2,
    verbose = 1,
    callbacks = list(print_dot_callback, 
                     callback_early_stopping(monitor = "val_loss", 
                                             min_delta = 0,
                                             patience = 10, 
                                             verbose = 1, 
                                             mode = c("auto", "min", "max"),
                                             baseline = TRUE, 
                                             restore_best_weights = TRUE))
)

summary(ensemble_model)
plot(ensemble_history)

ensemble_model %>% evaluate(ensemble_x_test, ensemble_y_test)

ensemble_IN_mat <- rbind(ensemble_x_train, ensemble_x_test) #ensemble_x_val, ensemble_x_test)
head(ensemble_IN_mat)

# Keras Model
#head(ensemble_test_data)
ensemble_norm <- normalizeData(ensemble_df, type='0_1')
colnames(ensemble_norm) <- c("observed","glm","drf","gbm","dpl")
ensemble_pred_data <- dplyr::select(as.data.frame(ensemble_norm), -c("observed"))

ensemble_dpl_pred <- ensemble_model %>% predict(as.matrix(ensemble_pred_data))


##################
## DENORMALIZE ###
##################
dpl_ensemble_pred_df <- as.data.frame(denormalizeData(ensemble_dpl_pred, getNormParameters(IN_norm_y)))
observed_ensemble_pred_df <- as.data.frame(denormalizeData(y_test, getNormParameters(IN_norm_y)))


ensemble_results <- cbind(observed_ensemble_pred_df, dpl_ensemble_pred_df)
colnames(ensemble_results) <- c("Observed", "Predicted")
ensemble_results_df <- as.data.frame(ensemble_results)
# 
# lares::mplot_density(tag = ensemble_results_df$Observed, 
#                      score = ensemble_results_df$Predicted,
#                      subtitle = "Ensemble Model",
#                      model_name = "simple_model_02")
# 

lares::mplot_full(tag = ensemble_results_df$Observed, 
                  score = ensemble_results_df$Predicted,
                  splits = 10,
                  subtitle = "Ensemble Model",
                  model_name = "simple_model_02",
                  save = T)

