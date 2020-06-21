rm(list=ls())
library(h2o)
library(data.table)
library(bit64)
library(dplyr)
library(lubridate)
library(openxlsx)

# data
# dsn <- "C:\\Users\\jmatney\\Documents\\ML Research\\data\\"
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\model\\"
# ---Apply h2o library   ------- #
setwd(dsn)

IN_df <- read.xlsx(paste0(dsn,"IN_Risk_Model.xlsx"))

head(IN_df)

# IN_22_scale <- as.data.frame(read.csv(paste0(dsn,"IN_DL_scale.csv")))
# colnames(IN_22_scale) <- names(IN.df)

# IN_df <- IN.df[ , -which(names(IN.df) %in% c("orb25yr24ha_am", "orb2yr24ha_am", "orb50yr24ha_am", "orb25yr24ha_am", "orb2yr24ha_am", "orb50yr24ha_am", "lu_21_area", "lu_22_area", "lu_41_area", "lu_23_area", 
#                                           "lu_82_area", "lu_24_area", "population", "area", "x_area", 
#                                           "housing_density", "watershed_length", "perimeter"))]

#IN_22_scale <- IN_22_scale[ , -which(names(IN_22_scale) %in% c("subwatershed"))]

dim(IN_df)

# # Clean slate - just in case the cluster was already running
h2o.removeAll()
# 
# # Cluster Info
h2o.clusterInfo()
# 
# # CLuster Status
h2o.clusterStatus()

# Start local host with given number of threads plus give memory size
h2o.init(ip='localhost', port=54321, nthreads=-1, max_mem_size = '20g')
# To shutdown cluster 
# h2o.shutdown(prompt=TRUE)

head(IN_df)
# convert into H2O frame
IN_dfc <- as.h2o(IN_df)

## Splits datasets into train, valid and test
splits <- h2o.splitFrame(data=IN_dfc, ratios=c(0.74, 0.117), seed=1236)
names(splits) <- c("train","valid","test")

traintest_path <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\trainTestValid\\"

# x_train <- read.csv(paste0(traintest_path, "x_train_model.csv"))
# y_train <- read.csv(paste0(traintest_path, "y_train_model.csv"))
# x_valid <- read.csv(paste0(traintest_path, "x_valid_model.csv"))
# y_valid <- read.csv(paste0(traintest_path, "y_valid_model.csv"))
# x_test <- read.csv(paste0(traintest_path, "x_test_model.csv"))
# y_test <- read.csv(paste0(traintest_path, "y_test_model.csv"))
# 
# drops <- c("X.x", "X.y","subwatershed")

train_df <-x_train %>% left_join(y_train, "subwatershed")
train_mod <- train_df[ , !(names(train_df) %in% drops)]

valid_df <-x_valid %>% left_join(y_valid, "subwatershed")
valid_mod <- valid_df[ , !(names(valid_df) %in% drops)]

test_df <-x_test %>% left_join(y_test, "subwatershed")
test_mod <- test_df[ , !(names(test_df) %in% drops)]


## assign the first result the R variable train
train <- as.h2o(train_mod) #h2o.assign(splits[[1]], "train.hex")   ## and the H2O name train.hex
valid <- as.h2o(valid_mod) #h2o.assign(splits[[2]], "valid.hex")   ## R valid, H2O valid.hex
test <- as.h2o(test_mod) #h2o.assign(splits[[3]],  "test.hex")     ## R test, H2O test.hex

##############
### VIF ######
##############

# library(tidyverse)
# library(caret)
# 
# IN_df <- IN_22_scale[ , -which(names(IN_22_scale) %in% c("orb25yr24ha_am", "orb2yr24ha_am", "orb50yr24ha_am"))]
# 
# IN_df <- IN_df[ , -which(names(IN_df) %in% c("orb25yr24ha_am", "orb2yr24ha_am", "orb50yr24ha_am", "lu_21_area", "lu_22_area", "lu_41_area", "lu_23_area", 
#                                              "lu_82_area", "lu_24_area", "population", "area", "x_area", 
#                                              "housing_density", "watershed_length", "perimeter"))]
# 
# IN_df[1] <- IN.df[1]
# dim(IN_df)
# write.csv(IN_df, "use_this_data.csv")
# mod <- lm(claims_total_building_insurance_coverage_avg ~., data=IN_df)
# a <- car::vif(mod)
# sort(round(a,2))
# 
# 
# getwd()
# write.csv(as.data.frame(train),"train_reduced.csv")
# write.csv(as.data.frame(test),"test_reduced.csv")
# write.csv(as.data.frame(valid),"valid_reduced.csv")

############################


# Define response and predictors #
response <- "claims_total_building_insurance_coverage_avg"
predictors <- names(train_mod[,which(!names(train_mod) %in% c("claims_total_building_insurance_coverage_avg"))])
gc()


###################
## Random Forest ##
###################

# write.csv(as.data.frame(train),"train_reduced.csv")
# write.csv(as.data.frame(test),"test_reduced.csv")
# write.csv(as.data.frame(valid),"valid_reduced.csv")


## run our first predictive model
rf_IN_22c <- h2o.randomForest(        ## h2o.randomForest function
  model_id = "rf_IN_22c",             ## name the model in H2O
  training_frame = train,        ## the H2O frame for training
  validation_frame = valid,      ## the H2O frame for validation (not required)
  x=predictors,                  ## the predictor columns, by column index
  y=response,                    ## the target index (what we are predicting)
  nfolds=10,
  
  ##   not required, but helps use Flow
  ntrees = 200,                  ## use a maximum of 200 trees
  max_depth = 10,
  stopping_metric = "deviance",
  stopping_rounds = 5,
  keep_cross_validation_predictions = T,
  stopping_tolerance = 0,     ##
  score_each_iteration = T,      ## Predict against training and validation for
  ## each tree. Default will skip several. fold_assignment = stratified,
  seed = 111)                    ## Set the random seed so that this can be reproduced.


###############################################################################
summary(rf_IN_22c)


#p100 <-as.data.frame(train)
#head(p100)


# Variable Importance
h2o.varimp_plot(rf_IN_22c)

# retrieve the model performance
# perf <- h2o.performance(rf861c, valid)
# perf

rf_performance <- h2o.performance(rf_IN_22c, test)
rf_performance

# perf2 <- h2o.performance(rf_IN_22c, train)
# perf2

# results <-  h2o.confusionMatrix(rf_IN_22c, test)

##############################
## Generalized Linear Model ##
##############################


## run our second predictive model
glm_IN_22c <- h2o.glm(family= "gaussian",
              x= predictors,
              y=response,
              training_frame = train,
              validation_frame = valid,  
              seed = 111,
              lambda = 0,
              max_active_predictors = 1,
              early_stopping = TRUE)               

h2o.coef(glm_IN_22c)



# Variable Importance
h2o.varimp_plot(glm_IN_22c)

# retrieve the model performance
# perf <- h2o.performance(rf861c, valid)
# perf

glm_performance <- h2o.performance(glm_IN_22c, test)
glm_performance


#######################
## Gradient Boosting ##
#######################

###########
## DALEX ##
###########
library(DALEX)
x_valid <- as.data.frame(splits$valid)[, predictors]
y_valid <- as.vector(as.numeric(as.character(splits$valid$claims_total_building_insurance_coverage_avg)))
head(y_valid)

h2o.predict(rf_IN_22c, as.h2o(x_valid))
##  predict
# 1 0.25228501
# 2 0.29230132
# 3 0.02336654
# 4 0.20098733
# 5 0.06316784
# 6 0.07213516


# gradient boosting machine model
gbm_IN_22c <-  h2o.gbm(
  x = predictors, 
  y = response,
  training_frame = train,
  validation_frame = valid,  
  seed = 123,
  stopping_metric = "RMSE", 
  #  keep_cross_validation_predictions = T,
  stopping_tolerance = 1e-2,     ##
  score_each_iteration = T,
)


# Variable Importance
h2o.varimp_plot(gbm_IN_22c)

h2o.performance(gbm_IN_22c, test)


gbm_performance <- h2o.performance(gbm_IN_22c, test)
gbm_performance



#########################
### DALEX explainers ####
#########################
# create custom predict function
pred <- function(model, newdata)  {
  results <- as.data.frame(h2o.predict(model, as.h2o(newdata)))
  return(results$predict)
}

pred(rf_IN_22c, x_valid) %>% head()


# elastic net explainer
explainer_glm <- explain(
  model = glm_IN_22c,
  data = x_valid,
  y = y_valid,
  predict_function = pred,
  label = "h2o glm"
)

# random forest explainer
explainer_rf <- explain(
  model = rf_IN_22c,
  data = x_valid,
  y = y_valid,
  predict_function = pred,
  label = "h2o rf"
)

# GBM explainer
explainer_gbm <- explain(
  model = gbm_IN_22c,
  data = train[,predictors],
  y = as.vector(train[,response]),
  predict_function = pred,
  label = "h2o gbm"
)

# example of explainer object
class(explainer_glm)
## [1] "explainer"
summary(explainer_glm)
# compute predictions & residuals
resids_glm <- DALEX::model_performance(explainer_glm)
resids_rf  <- model_performance(explainer_rf)
resids_gbm <- model_performance(explainer_gbm)

# create comparison plot of residuals for each model
p1 <- plot(resids_glm, resids_rf, resids_gbm)
p2 <- plot(resids_glm, resids_rf, resids_gbm, geom = "boxplot")

gridExtra::grid.arrange(p1, p2, nrow = 1)


####################

drf_results <- as.data.frame(h2o.predict(rf_IN_22c, test))
drf_results <- as.data.frame(cbind(as.data.frame(test_df)$subwatershed, drf_results, test_df$claims_total_building_insurance_coverage_avg))
names(drf_results) <- c("subwatershed", "predicted", "observed")

glm_results <- as.data.frame(h2o.predict(glm_IN_22c, test))
glm_results <- as.data.frame(cbind(as.data.frame(test_df)$subwatershed, glm_results, test_df$claims_total_building_insurance_coverage_avg))
names(glm_results) <- c("subwatershed", "predicted", "observed")

gbm_results <- as.data.frame(h2o.predict(gbm_IN_22c, test))
gbm_results <- as.data.frame(cbind(as.data.frame(test_df)$subwatershed, gbm_results, test_df$claims_total_building_insurance_coverage_avg))
names(gbm_results) <- c("subwatershed", "predicted", "observed")

######################
getwd()
write.csv(drf_results, "results/drf_results.csv", row.names = FALSE)
write.csv(glm_results, "results/glm_results.csv", row.names = FALSE)
write.csv(gbm_results, "results/gbm_results.csv", row.names = FALSE)

############


glm_performance
rf_performance
gbm_performance

h2o.varimp_plot(glm_IN_22c)
h2o.varimp_plot(rf_IN_22c)
h2o.varimp_plot(gbm_IN_22c)

names(test)
