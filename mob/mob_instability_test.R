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
split(bh, predict(bh_tree, newdata = bh, type = "node"))
nodeapply(bh_tree, nodeids(bh_tree), function(x) info_node(x)$nobs)
data_party(bh_tree, id = 3)
