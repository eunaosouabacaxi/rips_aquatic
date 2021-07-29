library("partykit")
## Boston housing data
data("BostonHousing", package = "mlbench")
BostonHousing <- transform(BostonHousing,
                           chas = factor(chas, levels = 0:1, labels = c("no", "yes")),
                           rad = factor(rad, ordered = TRUE))

## linear model tree
bh_tree <- lmtree(medv ~ log(lstat) + I(rm^2) | zn +
                    indus + chas + nox + age + dis + rad + tax + crim + b + ptratio,
                  data = BostonHousing, minsize = 40)

## printing whole tree or individual nodes
print(bh_tree)
print(bh_tree, node = 7)

## plotting
plot(bh_tree)
plot(bh_tree, tp_args = list(which = "log(lstat)"))
plot(bh_tree, terminal_panel = NULL)

## estimated parameters
coef(bh_tree)
coef(bh_tree, node = 9)
summary(bh_tree, node = 9)

## various ways for computing the mean squared error (on the training data)
mean((BostonHousing$medv - fitted(bh_tree))^2)
mean(residuals(bh_tree)^2)
deviance(bh_tree)/sum(weights(bh_tree))
deviance(bh_tree)/nobs(bh_tree)

## log-likelihood and information criteria
logLik(bh_tree)
AIC(bh_tree)
BIC(bh_tree)
## (Note that this penalizes estimation of error variances, which
## were treated as nuisance parameters in the fitting process.)

## different types of predictions
bh <- BostonHousing[c(1:50), ]
predict(bh_tree, newdata = bh, type = "node")
#predict(bh_tree, newdata = bh, type = "response")
#predict(bh_tree, newdata = bh, type = function(object) summary(object)$r.squared)
result_1 <- split(bh, predict(bh_tree, newdata = bh, type = "node"))
result_1
nodeapply(bh_tree, nodeids(bh_tree), function(x) info_node(x)$nobs)
data_party(bh_tree, id = 3)

## new tree for comparison
bh_tree_fewer_z <- lmtree(medv ~ log(lstat) + I(rm^2) |
                    indus + tax + crim + ptratio,
                  data = BostonHousing, minsize = 40)
plot(bh_tree)
bh_copy <- BostonHousing[c(1:50), ]
result_2 <- split(bh_copy, predict(bh_tree_fewer_z, newdata = bh_copy, type = "node"))
result_2$`3`

library(dplyr)
x = nrow(inner_join(result_1$`3`, result_2$`3`))
ratio = x / nrow(result_1$`3`)
ratio


library("partykit")
## Load data
# using a week to train
df <- readRDS("/u/project/cratsch/haoyudu/rds/combined_week_1_morning_2015.rds")

library("tidyverse")
library("arules")
library("dplyr")

## preprocess for tree
df$Unnamed..0_x <- NULL
df$Unnamed..0.1_x <- NULL
df$Unnamed..0_y <- NULL
df$Unnamed..0.1_y <- NULL
head(df)

print(names(df)[89])

## mob but linear
# continuous Z's
df_tree_cont <- lmtree(y2 ~ x8 | z4 + z7 + z8,
                  data = df, minsize = 40000)
print(df_tree_cont)

## discretize
disc <- discretizeDF(df[,39:89], default = list(method = "interval", breaks = 10,
                                       labels = 1:10))
df_1 <- df[,1:38]
df_2 <- df[,90:95]
df_final <- data.frame(data.frame(df_1,disc),df_2)

## mob but linear
# discrete Z's
df_tree_disc <- lmtree(y2 ~ x8 | z4 + z7 + z8,
                  data = df_final, minsize = 40000)
print(df_tree_disc)

## splitting the dataframe into nodes
result_1 <- split(df_final, predict(df_tree_cont, newdata = df_final, type = "node"))
result_2 <- split(df_final, predict(df_tree_disc, newdata = df_final, type = "node"))

## calculate how many rows are the same in one node
# CHANGE THE NODE NUMBER
x = nrow(inner_join(result_1$`3`, result_2$`3`))


