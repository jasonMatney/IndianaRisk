rm(list=ls())
library(h2o)
library(data.table)
library(bit64)
library(dplyr)
library(lubridate)
library(openxlsx)

# data
# dsn <- "C:\\Users\\jmatney\\Documents\\ML Research\\data\\"
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\"
# ---Apply h2o library   ------- #
setwd(dsn)

IN_22_scale <- as.data.frame(read.csv(paste0(dsn,"IN_DL_scale.csv")))
IN_22_scale
names(IN_22_scale)

# # Clean slate - just in case the cluster was already running
h2o.removeAll() ''
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

# head(IN_22_sub)
# 
# drops <- c("claims_amt_paid_building_avg", 
#            "claims_amt_paid_contents_avg",  
#            "claims_total_contents_insurance_coverage_avg",
#            "claims_amt_paid_building_sum",
#            "claims_amt_paid_contents_sum",
#            "claims_total_building_insurance_coverage_sum",
#            "claims_total_contents_insurance_coverage_sum",
#            "policy_total_contents_coverage_avg",
#            "policy_total_insurance_premium_avg",
#            "policy_total_building_coverage_sum",
#            "policy_total_building_coverage_sum",
#            "policy_total_contents_coverage_sum",
#           "orb100yr06h", "orb100yr06ha_am", "orb100yr12h",                          
#           "orb100yr12ha_am", "orb100yr24h",                       
#           "orb25yr06h", "orb25yr06ha_am", "orb25yr12h",                           
#           "orb25yr12ha_am", "orb25yr24h",                        
#           "orb2yr06h", "orb2yr06ha_am", "orb2yr12h",                            
#           "orb2yr12ha_am", "orb2yr24h",                         
#           "orb50yr06h", "orb50yr06ha_am", "orb50yr12h",                           
#           "orb50yr12ha_am", "orb50yr24h")
# IN_22 <- IN_22_sub[, !names(IN_22_sub) %in% drops]
# names(IN_22)

# convert into H2O frame
IN_22c <- as.h2o(IN_22_scale)

## Splits datasets into train, valid and test
splits <- h2o.splitFrame(data=IN_22c, ratios=c(0.8, 0.15), seed=1236)
names(splits) <- c("train","valid","test")

## assign the first result the R variable train
train <- h2o.assign(splits[[1]], "train.hex")   ## and the H2O name train.hex
valid <- h2o.assign(splits[[2]], "valid.hex")   ## R valid, H2O valid.hex
test <- h2o.assign(splits[[3]],  "test.hex")     ## R test, H2O test.hex

# Define response and predictors #
response <- "claims_total_building_insurance_coverage_avg"
predictors <- names(IN_22_scale[,which(!names(IN_22_scale) %in% c("subwatershed", "claims_total_building_insurance_coverage_avg"))])
gc()

###################
## Random Forest ##
###################

## run our first predictive model
rf_IN_22c <- h2o.randomForest(        ## h2o.randomForest function
  model_id = "rf_IN_22c",             ## name the model in H2O
  training_frame = train,        ## the H2O frame for training
  validation_frame = valid,      ## the H2O frame for validation (not required)
  x=predictors,                  ## the predictor columns, by column index
  y=response,                    ## the target index (what we are predicting)
  nfolds=4,
  
  ##   not required, but helps use Flow
  ntrees = 200,                  ## use a maximum of 200 trees to create the
  max_depth = 60,
  stopping_metric = "MAE", 
  #  keep_cross_validation_predictions = T,
  stopping_tolerance = 1e-2,     ##
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
  stopping_metric = "MAE", 
  #  keep_cross_validation_predictions = T,
  stopping_tolerance = 1e-2,     ##
  score_each_iteration = T,
)


# Variable Importance
h2o.varimp_plot(gbm_IN_22c)

h2o.performance(gbm_IN_22c, test)

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
