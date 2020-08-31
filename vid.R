rm(list=ls)
library(tidyverse)
library(caret)
options(scipen=999)
# Load the data
dsn <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\"
setwd(dsn)
IN_df <- read.xlsx(paste0(dsn,"MOD_DF_0723.xlsx"))
IN_df <- IN_df %>% mutate_if(is.character,as.numeric) 
# Split the data into training and test set
set.seed(123)

training.samples <- IN_df$mean_claim %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data.index  <- IN_df[training.samples, ]
test.data.index <- IN_df[-training.samples, ]

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
                                                             "relief", 
                                                             "population", 
                                                             "ruggedness", 
                                                             "population_density", 
                                                             "perimeter",
                                                             "elongation_ratio"
                                                             # "housing_density", 
                                                             # "circulatory_ratio", 
                                                             # "dependent_population_pct", 
                                                             # "water_bodies_area",
                                                             # "dams_count", 
                                                             # "median_income",
                                                             # "slope_of_flow_duration_curve"
                                                             # "dist_to_stream_stdev",
                                                             # "dist_to_stream_skewness",
                                                             # "dist_to_stream_kurtosis",
                                                             # "mean_policy",
                                                             # "watershed_length",
                                                             # "shape_factor",
                                                             # "streets_km",
                                                             # "relief_ratio"
                                                             ))])

new_cols <- append(new_cols, "mean_claim")

############################

vif.train.data.index  <- IN_df[training.samples, new_cols]
vif.test.data.index <- IN_df[-training.samples, new_cols]

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
