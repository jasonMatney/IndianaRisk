library(sparklyr)
library(h2o)
library(rsparkling)
library(dplyr)
library(openxlsx)
library(caret)
library(car)
library(ggcorrplot)
sc <- spark_connect(master = "local", version = "3.0.0")
h2oConf <- H2OConf()
hc <- H2OContext.getOrCreate(h2oConf)


# data
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\"
setwd(dsn)
ml_df <- read.xlsx(paste0(dsn,"MOD_DF_08_21.xlsx"))
ml_df <- ml_df[ , !(names(ml_df) %in% c("subwatershed"))]
ml_df <- ml_df %>% mutate_if(is.character,as.numeric) 
# calulate the correlations
r <- cor(ml_df[,1:20], use="complete.obs")
round(r,2)
ggcorrplot(r)
ml_df <- ml_df[complete.cases(ml_df), ] #### VIF NAMES IF FEATURE SELECTION ON
dim(ml_df)

response <- "mean_claim"
predictors <- names(ml_df[,which(!names(ml_df) %in% c("mean_claim", "subwatershed"))])
predictors

currentData <- ml_df
linear_reg <- lm(mean_claim ~ . ,data=currentData)

outlierResult <-outlierTest(linear_reg)
exclusionRows <-names(outlierResult[[1]])
inclusionRows <- !(rownames(currentData) %in% exclusionRows)
ml_df <- currentData[inclusionRows,]

library(dplyr)
model_data <- filter(ml_df, mean_claim >= 1000 & mean_claim <= 5000)
dim(model_data)
model_tbl <- copy_to(sc, model_data, overwrite = TRUE)
model_tbl

# transform our data set, and then partition into 'training', 'test'
partitions <- model_tbl %>%
  sdf_random_split(training = 0.5, test = 0.5, seed = 1099)


training <- hc$asH2OFrame(partitions$training)
test <- hc$asH2OFrame(partitions$test)

# fit a linear model to the training dataset
glm_model <- h2o.gbm(x = predictors, 
                     y = response, 
                     training_frame = training)



library(ggplot2)

# compute predicted values on our test dataset
pred <- h2o.predict(glm_model, newdata = test)
# convert from H2O Frame to Spark DataFrame
pred_sdf <- hc$asSparkFrame(pred)
head(pred_sdf)

# extract the true 'mpg' values from our test dataset
actual <- partitions$test %>%
  select(mean_claim) %>%
  collect() %>%
  `[[`("mean_claim")

# produce a data.frame housing our predicted + actual 'mpg' values
data <- data.frame(
  predicted = pred_sdf,
  actual    = actual
)
# a bug in data.frame does not set colnames properly; reset here 
names(data) <- c("predicted", "actual")

# plot predicted vs. actual values
ggplot(data, aes(x = actual, y = predicted)) +
geom_smooth(method = "lm", se = FALSE) +
  geom_point() 

h2o.r2(glm_model)
# +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   coord_fixed(ratio = 1) +
#   labs(
#     x = "Actual Fuel Consumption",
#     y = "Predicted Fuel Consumption",
#     title = "Predicted vs. Actual Fuel Consumption"
#   )
