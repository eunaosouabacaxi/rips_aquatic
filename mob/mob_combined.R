library("partykit")
library("arules")
library("rpart")
library("tidyverse")

prefix <- "/u/project/cratsch/matthewc/ft_week_"
suffix <- "_early_2015.rds"

# morning only
str <- "/u/project/cratsch/tescala/output_for_mob/top_feats_"
str1 <- "_early_50.csv"

# global variable for mob input
a <- 0
b <- 0
c <- 0
d <- 0
e <- 0

inter_1 <- 0
inter_2 <- 0
inter_3 <- 0
inter_4 <- 0
inter_5 <- 0

# vector of X's and Z's, constant
x <- c()
z <- c()
for (i in 1:34) {
    str2 <- paste("x", i, sep = "")
    x <- append(x, str2)
}
for (i in 1:12) {
    str3 <- paste("z", i, sep = "")
    z <- append(z, str3)
}

feature_selection <- function(month, day, N, M, df) {
    
    # concat file name
    date <- paste(month, paste("_", day, sep = ""), sep = "")
    s <- paste(paste(str, date, sep = ""), str1, sep = "")
    features <- read.csv(s)
    
    # saves how many times each X and Z appear in the feature selection
    x_ctr <- c()
    z_ctr <- c()
    for (i in 1:34) {
      x_list <- nrow(which(features == x[i], arr.ind = TRUE))
      x_ctr <- append(x_ctr, x_list)
    }
    for (i in 1:12) {
      z_list <- nrow(which(features == z[i], arr.ind = TRUE))
      z_ctr <- append(z_ctr, z_list)
    }

    # saves the top N X's and top M Z's into respective vectors
    max_x_list <- c()
    for (i in 1:N) {
      max_x_list <- append(max_x_list, which.max(x_ctr))
      x_ctr[which.max(x_ctr)] <- 0
    }
    
    max_z_list <- c()
    for (i in 1:M) {
      max_z_list <- append(max_z_list, which.max(z_ctr))
      z_ctr[which.max(z_ctr)] <- 0
    }

    x_top <- c()
    for (i in 1:5) {
      x_top <- append(x_top, x[max_x_list[i]])
    }
    
    z_top <- c()
    for (i in 1:5) {
      z_top <- append(z_top, z[max_z_list[i]])
    }
    
    # PROBLEM
    if (!is.na(match(x_top[1], x))) {
      f <- match(x_top[1], x)
    if (f == 1) a <<- df$x1
    if (f == 2) a <<- df$x2
    if (f == 3) a <<- df$x3
    if (f == 4) a <<- df$x4
    if (f == 5) a <<- df$x5
    if (f == 6) a <<- df$x6
    if (f == 7) a <<- df$x7
    if (f == 8) a <<- df$x8
    if (f == 9) a <<- df$x9
    if (f == 10) a <<- df$x10
    if (f == 11) a <<- df$x11
    if (f == 12) a <<- df$x12
    if (f == 13) a <<- df$x13
    if (f == 14) a <<- df$x14
    if (f == 15) a <<- df$x15
    if (f == 16) a <<- df$x16
    if (f == 17) a <<- df$x17
    if (f == 18) a <<- df$x18
    if (f == 19) a <<- df$x19
    if (f == 20) a <<- df$x20
    if (f == 21) a <<- df$x21
    if (f == 22) a <<- df$x22
    if (f == 23) a <<- df$x23
    if (f == 24) a <<- df$x24
    if (f == 25) a <<- df$x25
    if (f == 26) a <<- df$x26
    if (f == 27) a <<- df$x27
    if (f == 28) a <<- df$x28
    if (f == 29) a <<- df$x29
    if (f == 30) a <<- df$x30
    if (f == 31) a <<- df$x31
    if (f == 32) a <<- df$x32
    if (f == 33) a <<- df$x33
    if (f == 34) a <<- df$x34
    }

    if (!is.na(match(x_top[2], x))) {
      f <- match(x_top[2], x)
    if (f == 1) b <<- df$x1
    if (f == 2) b <<- df$x2
    if (f == 3) b <<- df$x3
    if (f == 4) b <<- df$x4
    if (f == 5) b <<- df$x5
    if (f == 6) b <<- df$x6
    if (f == 7) b <<- df$x7
    if (f == 8) b <<- df$x8
    if (f == 9) b <<- df$x9
    if (f == 10) b <<- df$x10
    if (f == 11) b <<- df$x11
    if (f == 12) b <<- df$x12
    if (f == 13) b <<- df$x13
    if (f == 14) b <<- df$x14
    if (f == 15) b <<- df$x15
    if (f == 16) b <<- df$x16
    if (f == 17) b <<- df$x17
    if (f == 18) b <<- df$x18
    if (f == 19) b <<- df$x19
    if (f == 20) b <<- df$x20
    if (f == 21) b <<- df$x21
    if (f == 22) b <<- df$x22
    if (f == 23) b <<- df$x23
    if (f == 24) b <<- df$x24
    if (f == 25) b <<- df$x25
    if (f == 26) b <<- df$x26
    if (f == 27) b <<- df$x27
    if (f == 28) b <<- df$x28
    if (f == 29) b <<- df$x29
    if (f == 30) b <<- df$x30
    if (f == 31) b <<- df$x31
    if (f == 32) b <<- df$x32
    if (f == 33) b <<- df$x33
    if (f == 34) b <<- df$x34
    }

    if (!is.na(match(x_top[3], x))) {
      f <- match(x_top[3], x)
    if (f == 1) c <<- df$x1
    if (f == 2) c <<- df$x2
    if (f == 3) c <<- df$x3
    if (f == 4) c <<- df$x4
    if (f == 5) c <<- df$x5
    if (f == 6) c <<- df$x6
    if (f == 7) c <<- df$x7
    if (f == 8) c <<- df$x8
    if (f == 9) c <<- df$x9
    if (f == 10) c <<- df$x10
    if (f == 11) c <<- df$x11
    if (f == 12) c <<- df$x12
    if (f == 13) c <<- df$x13
    if (f == 14) c <<- df$x14
    if (f == 15) c <<- df$x15
    if (f == 16) c <<- df$x16
    if (f == 17) c <<- df$x17
    if (f == 18) c <<- df$x18
    if (f == 19) c <<- df$x19
    if (f == 20) c <<- df$x20
    if (f == 21) c <<- df$x21
    if (f == 22) c <<- df$x22
    if (f == 23) c <<- df$x23
    if (f == 24) c <<- df$x24
    if (f == 25) c <<- df$x25
    if (f == 26) c <<- df$x26
    if (f == 27) c <<- df$x27
    if (f == 28) c <<- df$x28
    if (f == 29) c <<- df$x29
    if (f == 30) c <<- df$x30
    if (f == 31) c <<- df$x31
    if (f == 32) c <<- df$x32
    if (f == 33) c <<- df$x33
    if (f == 34) c <<- df$x34
    }

    if (!is.na(match(x_top[4], x))) {
      f <- match(x_top[4], x)
    if (f == 1) d <<- df$x1
    if (f == 2) d <<- df$x2
    if (f == 3) d <<- df$x3
    if (f == 4) d <<- df$x4
    if (f == 5) d <<- df$x5
    if (f == 6) d <<- df$x6
    if (f == 7) d <<- df$x7
    if (f == 8) d <<- df$x8
    if (f == 9) d <<- df$x9
    if (f == 10) d <<- df$x10
    if (f == 11) d <<- df$x11
    if (f == 12) d <<- df$x12
    if (f == 13) d <<- df$x13
    if (f == 14) d <<- df$x14
    if (f == 15) d <<- df$x15
    if (f == 16) d <<- df$x16
    if (f == 17) d <<- df$x17
    if (f == 18) d <<- df$x18
    if (f == 19) d <<- df$x19
    if (f == 20) d <<- df$x20
    if (f == 21) d <<- df$x21
    if (f == 22) d <<- df$x22
    if (f == 23) d <<- df$x23
    if (f == 24) d <<- df$x24
    if (f == 25) d <<- df$x25
    if (f == 26) d <<- df$x26
    if (f == 27) d <<- df$x27
    if (f == 28) d <<- df$x28
    if (f == 29) d <<- df$x29
    if (f == 30) d <<- df$x30
    if (f == 31) d <<- df$x31
    if (f == 32) d <<- df$x32
    if (f == 33) d <<- df$x33
    if (f == 34) d <<- df$x34
    }

    if (!is.na(match(x_top[5], x))) {
      f <- match(x_top[5], x)
    if (f == 1) e <<- df$x1
    if (f == 2) e <<- df$x2
    if (f == 3) e <<- df$x3
    if (f == 4) e <<- df$x4
    if (f == 5) e <<- df$x5
    if (f == 6) e <<- df$x6
    if (f == 7) e <<- df$x7
    if (f == 8) e <<- df$x8
    if (f == 9) e <<- df$x9
    if (f == 10) e <<- df$x10
    if (f == 11) e <<- df$x11
    if (f == 12) e <<- df$x12
    if (f == 13) e <<- df$x13
    if (f == 14) e <<- df$x14
    if (f == 15) e <<- df$x15
    if (f == 16) e <<- df$x16
    if (f == 17) e <<- df$x17
    if (f == 18) e <<- df$x18
    if (f == 19) e <<- df$x19
    if (f == 20) e <<- df$x20
    if (f == 21) e <<- df$x21
    if (f == 22) e <<- df$x22
    if (f == 23) e <<- df$x23
    if (f == 24) e <<- df$x24
    if (f == 25) e <<- df$x25
    if (f == 26) e <<- df$x26
    if (f == 27) e <<- df$x27
    if (f == 28) e <<- df$x28
    if (f == 29) e <<- df$x29
    if (f == 30) e <<- df$x30
    if (f == 31) e <<- df$x31
    if (f == 32) e <<- df$x32
    if (f == 33) e <<- df$x33
    if (f == 34) e <<- df$x34
    }
    
    if (!is.na(match(z_top[1], z))) {
      f <- match(z_top[1], z)
    if (f == 1) inter_1 <<- df$z1
    if (f == 2) inter_1 <<- df$z2
    if (f == 3) inter_1 <<- df$z3
    if (f == 4) inter_1 <<- df$z4
    if (f == 5) inter_1 <<- df$z5
    if (f == 6) inter_1 <<- df$z6
    if (f == 7) inter_1 <<- df$z7
    if (f == 8) inter_1 <<- df$z8
    if (f == 9) inter_1 <<- df$z9
    if (f == 10) inter_1 <<- df$z10
    if (f == 11) inter_1 <<- df$z11
    if (f == 12) inter_1 <<- df$z12
    }
    
    if (!is.na(match(z_top[2], z))) {
      f <- match(z_top[2], z)
    if (f == 1) inter_2 <<- df$z1
    if (f == 2) inter_2 <<- df$z2
    if (f == 3) inter_2 <<- df$z3
    if (f == 4) inter_2 <<- df$z4
    if (f == 5) inter_2 <<- df$z5
    if (f == 6) inter_2 <<- df$z6
    if (f == 7) inter_2 <<- df$z7
    if (f == 8) inter_2 <<- df$z8
    if (f == 9) inter_2 <<- df$z9
    if (f == 10) inter_2 <<- df$z10
    if (f == 11) inter_2 <<- df$z11
    if (f == 12) inter_2 <<- df$z12
    }
    
    if (!is.na(match(z_top[3], z))) {
      f <- match(z_top[3], z)
    if (f == 1) inter_3 <<- df$z1
    if (f == 2) inter_3 <<- df$z2
    if (f == 3) inter_3 <<- df$z3
    if (f == 4) inter_3 <<- df$z4
    if (f == 5) inter_3 <<- df$z5
    if (f == 6) inter_3 <<- df$z6
    if (f == 7) inter_3 <<- df$z7
    if (f == 8) inter_3 <<- df$z8
    if (f == 9) inter_3 <<- df$z9
    if (f == 10) inter_3 <<- df$z10
    if (f == 11) inter_3 <<- df$z11
    if (f == 12) inter_3 <<- df$z12
    }
    
    if (!is.na(match(z_top[4], z))) {
      f <- match(z_top[4], z)
    if (f == 1) inter_4 <<- df$z1
    if (f == 2) inter_4 <<- df$z2
    if (f == 3) inter_4 <<- df$z3
    if (f == 4) inter_4 <<- df$z4
    if (f == 5) inter_4 <<- df$z5
    if (f == 6) inter_4 <<- df$z6
    if (f == 7) inter_4 <<- df$z7
    if (f == 8) inter_4 <<- df$z8
    if (f == 9) inter_4 <<- df$z9
    if (f == 10) inter_4 <<- df$z10
    if (f == 11) inter_4 <<- df$z11
    if (f == 12) inter_4 <<- df$z12
    }
    
    if (!is.na(match(z_top[5], z))) {
      f <- match(z_top[5], z)
    if (f == 1) inter_5 <<- df$z1
    if (f == 2) inter_5 <<- df$z2
    if (f == 3) inter_5 <<- df$z3
    if (f == 4) inter_5 <<- df$z4
    if (f == 5) inter_5 <<- df$z5
    if (f == 6) inter_5 <<- df$z6
    if (f == 7) inter_5 <<- df$z7
    if (f == 8) inter_5 <<- df$z8
    if (f == 9) inter_5 <<- df$z9
    if (f == 10) inter_5 <<- df$z10
    if (f == 11) inter_5 <<- df$z11
    if (f == 12) inter_5 <<- df$z12
    }
}

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
  feat <- feature_selection(month, day, N, M, training_final)
  
  ctrl <- mob_control(alpha = 0.05, bonferroni = TRUE, minsplit = 5000, verbose = TRUE)
  
    if (N == 1 & M == 1) {
        results <- lmtree(y2 ~ a | inter_1, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 2 & M == 1) {
        results <- lmtree(y2 ~ a + b | inter_1, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 3 & M == 1) {
        results <- lmtree(y2 ~ a + b + c | inter_1, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 4 & M == 1) {
        results <- lmtree(y2 ~ a + b + c + d | inter_1, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 5 & M == 1) {
        results <- lmtree(y2 ~ a + b + c + d + e | inter_1, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    
    if (N == 1 & M == 2) {
        results <- lmtree(y2 ~ a | inter_1 + inter_2, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 2 & M == 2) {
        results <- lmtree(y2 ~ a + b | inter_1 + inter_2, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 3 & M == 2) {
        results <- lmtree(y2 ~ a + b + c | inter_1 + inter_2, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 4 & M == 2) {
        results <- lmtree(y2 ~ a + b + c + d | inter_1 + inter_2, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 5 & M == 2) {
        results <- lmtree(y2 ~ a + b + c + d + e | inter_1 + inter_2, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    
    if (N == 1 & M == 3) {
        results <- lmtree(y2 ~ a | inter_1 + inter_2 + inter_3, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 2 & M == 3) {
        results <- lmtree(y2 ~ a + b | inter_1 + inter_2 + inter_3, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 3 & M == 3) {
        results <- lmtree(y2 ~ a + b + c | inter_1 + inter_2 + inter_3, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 4 & M == 3) {
        results <- lmtree(y2 ~ a + b + c + d | inter_1 + inter_2 + inter_3, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 5 & M == 3) {
        results <- lmtree(y2 ~ a + b + c + d + e | inter_1 + inter_2 + inter_3, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
  
    if (N == 1 & M == 4) {
        results <- lmtree(y2 ~ a | inter_1 + inter_2 + inter_3 + inter_4, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 2 & M == 4) {
        results <- lmtree(y2 ~ a + b | inter_1 + inter_2 + inter_3 + inter_4, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 3 & M == 4) {
        results <- lmtree(y2 ~ a + b + c | inter_1 + inter_2 + inter_3 + inter_4, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 4 & M == 4) {
        results <- lmtree(y2 ~ a + b + c + d | inter_1 + inter_2 + inter_3 + inter_4, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 5 & M == 4) {
        results <- lmtree(y2 ~ a + b + c + d + e | inter_1 + inter_2 + inter_3 + inter_4, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    
    if (N == 1 & M == 5) {
        results <- lmtree(y2 ~ a | inter_1 + inter_2 + inter_3 + inter_4 + inter_5, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 2 & M == 5) {
        results <- lmtree(y2 ~ a + b | inter_1 + inter_2 + inter_3 + inter_4 + inter_5, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 3 & M == 5) {
        results <- lmtree(y2 ~ a + b + c | inter_1 + inter_2 + inter_3 + inter_4 + inter_5, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 4 & M == 5) {
        results <- lmtree(y2 ~ a + b + c + d | inter_1 + inter_2 + inter_3 + inter_4 + inter_5, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
    if (N == 5 & M == 5) {
        results <- lmtree(y2 ~ a + b + c + d + e | inter_1 + inter_2 + inter_3 + inter_4 + inter_5, data = training_final, weights = training_final$weights, minsize = 5000, alpha = 0.05, bonferroni = TRUE)
    }
  
  saveRDS(results,file = paste(paste("/u/project/cratsch/haoyudu/model_",toString(train_months),sep=""),".rds",sep=""))
}

test <- function(train_months, test_months) {
  model_file <- paste(paste("/u/project/cratsch/haoyudu/model_",toString(train_months),sep=""),".rds",sep="")
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
  predictions <- predict(model, newdata = testing_final, type = "response")
  for(i in 1:nrow(testing_final)) {
    ret <- sign(predictions[i])*testing_final$y1[i]*testing_final$weights[i]
    if(!is.na(ret)) {
      if(ret > 0) {
        correct <- correct + 1
      }
      net_return <- net_return + ret
      base_return <- base_return + ret*sign(predictions[i])
    }
  }
  print(net_return)
  print(base_return)
  print(correct/nrow(testing_final))
}

#feature <- c(1,5,1,1)
#feature_selection(1,5,1,1)
#do.call(feature_selection, as.list(1,5,1,1))
trainmonths <- c(1,2,3,4)
testmonths <- c(5)
train(trainmonths, 1, 5, 1, 1)
test(trainmonths,testmonths)
