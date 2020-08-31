options(scipen=999)
library(h2o)
library(data.table)
library(bit64)
library(dplyr)
library(lubridate)
library(openxlsx)
library(tidyverse)
library(keras)
library(tfdatasets)
library(RSNNS)
library(rlang)
library(lares)
library(readr)
library(sparklyr)
library(tidyverse)
library(caret)
library(car)
library(ggcorrplot)
library(car)
library(sparklyr)
require(caTools)  # loading caTools library
## h2o.shutdown()

# data
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\"
setwd(dsn)
model_data <- read.xlsx(paste0(dsn,"keras_model.xlsx"))

set.seed(123)   #  set seed to ensure you always have same random numbers generated
sample = sample.split(model_data,SplitRatio = 0.75) # splits the data in the ratio mentioned in SplitRatio. After splitting marks these rows as logical TRUE and the the remaining are marked as logical FALSE
train_df  = subset(model_data,sample ==TRUE) # creates a training dataset named train1 with rows which are marked as TRUE
test_df   = subset(model_data, sample==FALSE)


y_train <- train$mean_claim
x_train <- train %>% select(-mean_claim) 

spec <- feature_spec(train_df, mean_claim ~ . ) %>% 
  step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
  fit()

layer <- layer_dense_features(
  feature_columns = dense_features(spec), 
  dtype = tf$float32
)
layer(train_df)

input <- layer_input_from_dataset(train_df %>% select(-mean_claim))

output <- input %>% 
  layer_dense_features(dense_features(spec)) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1) 

model <- keras_model(input, output)

summary(model)

model %>% 
  compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )

# Define R2 custum metric
r2_metric <- custom_metric("R-squared", function(y_true, y_pred){
  
  SS_res = k_sum(k_square(y_true - y_pred))
  SS_tot = k_sum(k_square(y_true - k_mean(y_true)))
  
  R2 <- (1- SS_res/(SS_tot + k_epsilon()))
  
  return(R2)
}) 

build_model <- function() {
  input <- layer_input_from_dataset(train_df %>% select(-mean_claim))
  
  output <- input %>% 
    layer_dense_features(dense_features(spec)) %>% 
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1) 
  
  model <- keras_model(input, output)
  
  model %>% 
    compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(),
      metrics = list(r2_metric)
    )
  
  model
}

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

model <- build_model()

# The patience parameter is the amount of epochs to check for improvement.
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

history <- model %>% fit(
  x = train_df %>% select(-mean_claim),
  y = train_df$mean_claim,
  epochs = 10,
  validation_split = 0.2,
  verbose = 1,
  callbacks = list(early_stop)
)

plot(history)

c(loss, mae) %<-% (model %>% evaluate(test_df %>% select(-mean_claim), test_df$mean_claim, verbose = 1))

paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))

test_predictions <- model %>% predict(test_df %>% select(-mean_claim))
test_predictions[ , 1]

predictions <- as.data.frame(cbind(test_df$mean_claim, test_predictions))

colnames(predictions) <- c("observed","predicted")



r2_metric(predictions$observed, predictions$predicted)


