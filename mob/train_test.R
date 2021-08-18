library("partykit")
library("arules")
library("rpart")
library("tidyverse")

prefix <- "/u/project/cratsch/matthewc/ft_week_"
suffix <- "_early_2015.rds"

# morning only
str <- "/u/project/cratsch/tescala/output_for_mob/top_feats_"
str1 <- "_early_50.csv"

# vector of X's and Z's, constant

train <- function(train_months, month, day, N, M) {
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
  
  results <- lmtree(y2 ~ x8 | z8 + z4 + z2, data = training_final, weights = training_final$weights,minsize = 5000, alpha = 0.05, bonferroni = TRUE, caseweights = FALSE)
  saveRDS(results,file = paste(paste("/u/project/cratsch/matthewc/model_",toString(train_months),sep=""),".rds",sep=""))
}

test <- function(train_months, test_months) {
  model_file <- paste(paste("/u/project/cratsch/matthewc/model_",toString(train_months),sep=""),".rds",sep="")
  model <- readRDS(model_file)
  test_file <- paste(paste(prefix,test_months[1],sep=""),suffix,sep="")
  testing <- readRDS(test_file)
  train_file <- paste(paste(prefix,train_months[1],sep=""),suffix,sep="")
  training <- readRDS(train_file)
  for(i in 2:length(train_months)) {
    train_file <- paste(paste(prefix,train_months[i],sep=""),suffix,sep="")
    training <- merge(training, readRDS(train_file), all.x = TRUE, all.y = TRUE)
  }
  training$Unnamed..0_x <- NULL
  training$Unnamed..0.1_x <- NULL
  training$Unnamed..0_y <- NULL
  training$Unnamed..0_x <- NULL
  testing$Unnamed..0.1_x <- NULL
  testing$Unnamed..0_y <- NULL
  testing$Unnamed..0.1_y <- NULL
  testing$Unnamed..0.1_y <- NULL
  
  # normalizing the test data based on the training mean
  for(col in 5:38) {
    train_sum <- sum(training[col])
    train_mean <- train_sum / nrow(training)
    train_std <- sqrt(sum((training[col]-train_mean)^2)/nrow(training))
    testing[col] <- (testing[col]-train_mean)/train_std
  }
  
  disc <- discretizeDF(testing[,39:89], default = list(method = "interval", breaks = 10,
                                                       labels = 1:10))
  
  df_1 <- testing[,1:38]
  
  df_2 <- testing[,90:95]
  
  testing_final <- data.frame(data.frame(df_1,disc),df_2)
  correct <- 0
  base_return <- 0
  net_return <- 0
  rss <- 0
  tss <- 0
  predictions <- predict(model, newdata = testing_final, type = "response")
  for(i in 1:nrow(testing_final)) {
    ret <- sign(predictions[i])*testing_final$y2[i]*testing_final$weights[i]
    if(!is.na(ret)) {
      if(ret > 0) {
        correct <- correct + 1
      }
      net_return <- net_return + ret
      base_return <- base_return + ret*sign(predictions[i])
      rss <- rss + (predictions[i]-testing_final$y_2[i])^2
      tss <- tss + (testing_final$y_2[i])^2
    }
  }
  print(1-rss/tss)
  print(net_return)
  print(base_return)
  print(correct/nrow(testing_final))
}

#feature <- c(1,5,1,1)
#feature_selection(1,5,1,1)
#do.call(feature_selection, as.list(1,5,1,1))
for(i in 1:8) {
  trainmonths <- c(i,i+1,i+2,i+3)
  testmonths <- c(i+4)
  train(trainmonths, 1, 5, 1, 1)
  test(trainmonths,testmonths)
}


#train(trainmonths, 1, 5, 5, 1)
#test(trainmonths,testmonths)
