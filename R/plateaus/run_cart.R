library(Matrix)
library(rpart)

args = commandArgs(trailingOnly = TRUE)
trial = as.numeric(args[1]) # Trial ID
N = as.numeric(args[2]) # number of training samples

train_data = as.matrix(read.csv(file.path(sprintf("../../data/plateaus/train/%d/%d.csv", N, trial)), header=FALSE))
test_data = as.matrix(read.csv(file.path(sprintf("../../data/plateaus/truth/%d.csv", trial)), header=FALSE))
X = train_data[, c(1, 2)]
y = as.numeric(as.vector(train_data[, 3]))
X_test = test_data[, c(1, 2)]
y_test = test_data[, 3]

X = data.frame(X)
X_test = data.frame(X_test)
colnames(X) = c("x1", "x2")
colnames(X_test) = c("x1", "x2")

fit = rpart(y ~ x1 + x2, data=X, method='anova')
y_hat = predict(fit, X_test)
write.csv(y_hat, file.path(sprintf("../../data/plateaus/predictions/cart/%d/%d.csv", N, trial)), row.names=FALSE)
rmse = sqrt(mean((y_test - y_hat)**2))
maxerr = max(abs(y_test - y_hat))

sprintf("CART --> RMSE: %f Max Error: %f", rmse, maxerr)
write.csv(c(rmse,maxerr), file.path(sprintf("../../data/plateaus/results/cart/%d/%d.csv", N, trial)), row.names=FALSE)
