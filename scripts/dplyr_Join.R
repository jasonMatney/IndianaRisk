rm(list=ls())
library(dplyr)
original <- read.csv("C:/Users/jmatney/Documents/GitHub/IndianaRisk/data/IN_DL_05202020_original.csv")
hotfix <- read.csv("C:/Users/jmatney/Documents/GitHub/IndianaRisk/data/IN_DL_05202020_hotfix.csv")



new <- original %>% left_join(hotfix, by = "subwatershed")
dim(new)
write.csv(new, "modified.csv")
