library("party")
library("tidyverse")
library("arules")
library("rpart")

features <- readRDS("/u/project/cratsch/haoyudu/rds/features_jan_morning_2015.rds")
targets <- readRDS("/u/project/cratsch/haoyudu/rds/targets_jan_morning_2015.rds")

df <- as.data.frame(scale(features[5:89]))
normalized <- cbind(features[1:4],df)

disc <- discretizeDF(normalized[,39:89], default = list(method = "interval", breaks = 10,
                                                        labels = 1:10))
df_1 <- normalized[,1:38]

df_final <- data.frame(df_1,disc)
ft <- cbind(df_final,targets[5:10])

saveRDS(ft,file = "combined_jan_morning_2015.rds")