library("partykit")
library("arules")
library("rpart")
library("tidyverse")

prefix <- "/u/project/cratsch/matthewc/ft_month_"
suffix <- "_morning_2015.rds"

train <- function(train_months) {
  train_file <- paste(paste(prefix,train_months[1],sep=""),suffix,sep="")
  training <- readRDS(train_file)
  for(i in 2:length(train_months)) {
    train_file <- paste(paste(prefix,train_months[i],sep=""),suffix,sep="")
    training <- merge(training, readRDS(train_file), all.x = TRUE, all.y = TRUE)
  }
  
  training$Unnamed..0_x <- NULL
  training$Unnamed..0.1_x <- NULL
  training$Unnamed..0_y <- NULL
  training$Unnamed..0.1_y <- NULL
  
  df <- as.data.frame(scale(training[,5:38]))
  training <- cbind(cbind(training[,1:4],df),training[,39:95])
  
  disc <- discretizeDF(training[,39:89], default = list(method = "interval", breaks = 10,
                                                        labels = 1:10))
  
  df_1 <- training[,1:38]
  df_2 <- training[,90:95]
  
  training_final <- data.frame(data.frame(df_1,disc),df_2)
  
  ctrl <- mob_control(alpha = 0.05, bonferroni = TRUE, minsplit = 5000, verbose = TRUE)
  
  #results <- mob(y2 ~ x8 + x7 + x6 + x5 + x3 | z8,
  #            data = training_final, control = ctrl, model = linearModel
  #    )
  results <- lmtree(y2 ~ x8 + x7 + x6 + x5 + x3 | z8, data = training_final, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
  saveRDS(results,file = paste(paste("/u/project/cratsch/matthewc/model_",toString(train_months),sep=""),".rds",sep=""))
}

trainmonths <- c(1,2,3)

train(trainmonths)