library(methods)
library(foreach)
library(Matrix)
library(caret)
library(crisp)

mse = function(y, y.hat) {
    sum((y - y.hat)^2)/length(y)
}

loadGapStatistic = function(city, trial) {
    as.integer(read.csv(sprintf('../../data/crime/%s/results/gfl/%d.csv', city, trial), header=FALSE)$V1[3])
}

args = commandArgs(trailingOnly = TRUE)
city = args[1]
trial = as.numeric(args[2]) # Trial ID
q = loadGapStatistic(city, trial) # Use the same Q as the GapTV method

sprintf("q: %d", q)

max.lam = 10.0
lambda.min.ratio = 0.001
n.lambda = 50
lambda.seq = exp(seq(log(max.lam), log(max.lam * lambda.min.ratio), len = n.lambda))

train_data = as.matrix(read.csv(file.path(sprintf("../../data/crime/%s/train/%d.csv", city, trial)), header=FALSE))
test_data = as.matrix(read.csv(file.path(sprintf("../../data/crime/%s/test/%d.csv", city, trial)), header=FALSE))
X_sweep = data.frame(as.matrix(read.csv(file.path(sprintf("../../data/crime/%s/sweep.csv", city)), header=FALSE)))

N = dim(train_data)[1]
X = train_data[, c(1, 2)]
y = as.numeric(as.vector(train_data[, 3]))
X_test = test_data[, c(1, 2)]
y_test = test_data[, 3]

X = data.frame(X)
X_test = data.frame(X_test)
colnames(X) = c("x1", "x2")
colnames(X_test) = c("x1", "x2")
colnames(X_sweep) = c("x1", "x2")

# Estimate the best lambda via k-fold cross validation
k = 5
folds <- createFolds(1:length(y), k=k)
prederr = rep(0, n.lambda)
for (fold in folds){
    print("Fold")
    foldmodel = crisp(y[-fold], X[-fold,], n.lambda=n.lambda, q=q, lambda.seq=lambda.seq)
    prederr = prederr + sapply(1:n.lambda, function(i) mse(y[fold], predict(foldmodel, X[fold,], i)))
}
best.lambda = which.min(prederr)
if (best.lambda == 1){
    lambda.seq = c(max.lam+1, max.lam)
    best.lambda = 2
}
model = crisp(y, X, n.lambda=best.lambda, q=q, lambda.seq=lambda.seq[1:best.lambda])
y_hat = predict(model, X_test, best.lambda)
write.csv(y_hat, file.path(sprintf("../../data/crime/%s/predictions/gapcrisp/%d.csv", city, trial)), row.names=FALSE)
rmse = sqrt(mean((y_test - y_hat)**2))
maxerr = max(abs(y_test - y_hat))


sprintf("CRISP --> q: %d lambda: %f RMSE: %f Max Error: %f", q, lambda.seq[best.lambda], rmse, maxerr)
write.csv(c(rmse,maxerr), file.path(sprintf("../../data/crime/%s/results/gapcrisp/%d.csv", city, trial)), row.names=FALSE)
print("Sweeping")
y_sweep = predict(model, X_sweep, best.lambda)
print("Saving sweeps")
write.csv(y_sweep, file.path(sprintf("../../data/crime/%s/sweeps/gapcrisp/%d.csv", city, trial)), row.names=FALSE)
print("Done!")

