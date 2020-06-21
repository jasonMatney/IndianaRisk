library(openxlsx)
library(tidyverse)

path = "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\"

final_mod <- read.xlsx(paste0(path,"FINAL_MOD.xlsx"))
hex <- read.xlsx(paste0(path,"all_hexagons_50c_grouped_withCounties.xlsx"))
hotfix <- read.xlsx(paste0(path, "Hotfix.xlsx"))
streams <- read.xlsx(paste0(path, "700_stream_dist.xlsx"))

hex$subwatershed <- paste0("0", hex$ws_id)
dim(hex)
dim(final_mod)
dim(streams)
dim(hotfix)

df_streams <- final_mod %>% left_join(streams, by="subwatershed")
write.xlsx(df_streams, paste0(path, "df_streams.xlsx"))
# -----------------------
df <- read.xlsx(paste0(path, "df_streams.xlsx"))
head(df)

df_hotfix <- df %>% left_join(hotfix, by="subwatershed")
write.xlsx(df_hotfix, paste0(path, "df_hotfix.xlsx"))


df_fixed <- read.xlsx(paste0(path, "df_hotfix.xlsx"))
dim(df_fixed)

# # Remove duplica,tes based on Sepal.Width columns
final_mod <- df_fixed[!duplicated(df_fixed$subwatershed), ]
dim(final_mod)
# dim(final_mod)
# 
# 
dat=final_mod %>% left_join(hex, by="subwatershed")
write.xlsx(dat, paste0(path,"IN_Risk_MOD.xlsx"))


df_final <- read.xlsx(paste0(path, "\\model\\IN_Risk_MOD.xlsx"))
dat=df_final %>% left_join(hex, by="subwatershed")
head(dat)
# oidList = [
#   '051202060310',
#   '051201130903',
#   '051202081402',
#   '051202030505',
#   '051202080703',
#   '051201081204',
#   '051201060505',
#   '051202020605',
#   '051202011702',
#   '051202010902',
#   '051202081004'
#   
#   
#   ]


library(openxlsx)
final <- read.xlsx("C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\results\\RESULTS_MODEL.xlsx")
hex <- read.xlsx("C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\all_hexagons_50c_grouped_withCounties.xlsx")
hex$subwatershed <- paste0("0", hex$ws_id)
dim(hex)

# Remove duplica,tes based on Sepal.Width columns
final_mod <- final[!duplicated(final$subwatershed), ]
dim(final_mod)

library(tidyverse)
dat=final_mod %>% left_join(hex, by="subwatershed")
dim(dat)
head(dat)