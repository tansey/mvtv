library(Matrix)
library(rpart)

args = commandArgs(trailingOnly = TRUE)
city = args[1]
trial = as.numeric(args[2]) # Trial ID

train_data = as.matrix(read.csv(file.path(sprintf("../../data/crime/%s/train/%d.csv", city, trial)), header=FALSE))
test_data = as.matrix(read.csv(file.path(sprintf("../../data/crime/%s/test/%d.csv", city, trial)), header=FALSE))
X_sweep = data.frame(as.matrix(read.csv(file.path(sprintf("../../data/crime/%s/sweep.csv", city)), header=FALSE)))
X = train_data[, c(1, 2)]
y = as.numeric(as.vector(train_data[, 3]))
X_test = test_data[, c(1, 2)]
y_test = test_data[, 3]

X = data.frame(X)
X_test = data.frame(X_test)
colnames(X) = c("x1", "x2")
colnames(X_test) = c("x1", "x2")
colnames(X_sweep) = c("x1", "x2")

fit = rpart(y ~ x1 + x2, data=X, method='anova')
y_hat = predict(fit, X_test)
write.csv(y_hat, file.path(sprintf("../../data/crime/%s/predictions/cart/%d.csv", city, trial)), row.names=FALSE)
rmse = sqrt(mean((y_test - y_hat)**2))
maxerr = max(abs(y_test - y_hat))

sprintf("CART --> RMSE: %f Max Error: %f", rmse, maxerr)
write.csv(c(rmse,maxerr), file.path(sprintf("../../data/crime/%s/results/cart/%d.csv", city, trial)), row.names=FALSE)

y_sweep = predict(fit, X_sweep)
write.csv(y_sweep, file.path(sprintf("../../data/crime/%s/sweeps/cart/%d.csv", city, trial)), row.names=FALSE)
