library("party")
library("tidyverse")
df_early_2012 <- read.csv(file = 'early_2012.csv')
head(df_early_2012)
df_early_2012$Unnamed..0 <- NULL
df_early_2012$Unnamed..0.1 <- NULL
print(names(df_early_2012)[6])
for(i in 5:39) {
  # renaming predictors with X_i
  names(df_early_2012)[i] <- paste("X_", toString(i - 4), sep = "")
}
print(names(df_early_2012)[39])
for(i in 40:91) {
  print(paste(toString(i), names(df_early_2012)[i]))
}
names(df_early_2012)[39] <- "Z_1"
names(df_early_2012)[43] <- "Z_2"
names(df_early_2012)[47] <- "Z_3"
names(df_early_2012)[51] <- "Z_4"
names(df_early_2012)[55] <- "Z_5"
names(df_early_2012)[59] <- "Z_6"
names(df_early_2012)[63] <- "Z_7"
names(df_early_2012)[67] <- "Z_8"
names(df_early_2012)[71] <- "Z_9"
names(df_early_2012)[75] <- "Z_10"
names(df_early_2012)[79] <- "Z_11"
names(df_early_2012)[83] <- "Z_12"
names(df_early_2012)[87] <- "Z_13"
names(df_early_2012)[88] <- "Z_14"
names(df_early_2012)[89] <- "Z_15"
names(df_early_2012)[90] <- "Y_2"
names(df_early_2012)[91] <- "Y_1"
ctrl <- mob_control(alpha = 0.05, bonferroni = TRUE, minsplit = 5000, verbose = TRUE)
E2012 <- mob(Y_1 ~ X_1 + X_2 + X_3 + X_4 + X_5 + X_6 + X_7 + X_8 + X_9 + X_10
             + X_11 + X_12 + X_13 + X_14 + X_15 + X_16 + X_17 + X_18 + X_19 + X_20
             + X_21 + X_22 + X_23 + X_24 + X_25 + X_26 + X_27 + X_28 + X_29 + X_30
             + X_31 + X_32 + X_33 + X_34 |
             Z_1 + Z_2 + Z_3 + Z_4 + Z_5 + Z_6 + Z_7 + Z_8 + Z_9 + Z_10 + Z_11 + Z_12 + Z_13 + Z_14 + Z_15,
             data = df_early_2012, control = ctrl, model = linearModel
    )
# BostonHousing$lstat <- log(BostonHousing$lstat)
# BostonHousing$rm <- BostonHousing$rm^2
# BostonHousing$chas <- factor(BostonHousing$chas, levels = 0:1, labels = c("no", "yes"))
# BostonHousing$rad <- factor(BostonHousing$rad, ordered = TRUE)
# #setting
# ctrl <- mob_control(alpha = 0.05, bonferroni = TRUE, minsplit = 40, objfun = deviance, verbose = TRUE)
# fmBH <- mob(medv ~ lstat + rm | zn + indus + chas + nox + age + dis + rad + tax + crim + b + ptratio, data = BostonHousing, control = ctrl, model = linearModel)
# fmBH
# coef(fmBH)
# summary(fmBH, node = 7)
# sctest(fmBH, node = 7)
# mean(residuals(fmBH)^2)
# logLik(fmBH)
# AIC(fmBH)
# plot(fmBH)

#set.seed(1234)
#x <- matrix(rnorm(30, 1), ncol = 5)
#y <- c(1, seq(5))

#combining x and y into one matrix
#x <- cbind(x, y)

#converting x into a data frame called x.df
#x.df <- data.frame(x)
#x.df
