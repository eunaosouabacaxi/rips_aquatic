library("party")
library("tidyverse")
library("arules")

df_2015_day <- readRDS('combined_week_1_morning_2015.rds')
head(df_2015_day)

# We want to read only certain columns to avoid memory problems
# It is possible to nullify certain columns
# Do this for ever new rds file that you read into the dataframe
df_2015_day$Unnamed..0_x <- NULL
df_2015_day$Unnamed..0.1_x <- NULL
df_2015_day$Unnamed..0_y <- NULL
df_2015_day$Unnamed..0.1_y <- NULL

# look at column names
head(df_2015_day)

# should be z15
print(names(df_2015_day)[89])

# USE SEPARATE BLOCK! do this if you want to discretize Z
disc <- discretizeDF(df_2015_day[,39:89], default = list(method = "interval", breaks = 10,
                                       labels = 1:10))

df_1 <- df_2015_day[,1:38]

df_2 <- df_2015_day[,90:95]

df_final <- data.frame(data.frame(df_1,disc),df_2)

head(df_final$x8)

ctrl <- mob_control(alpha = 0.05, bonferroni = TRUE, minsplit = 5000, verbose = TRUE)

results <- mob(y2 ~ x8 |
            z3 + z4 + z5 + z6 + z7 + z8 + z10 + z11,
             data = df_final, control = ctrl, model = linearModel
    )

options(repr.plot.width=140, repr.plot.height=30)

plot(results)
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
