library(methods)
library(foreach)
library(Matrix)
library(caret)
library(crisp)

args = commandArgs(trailingOnly = TRUE)
trial = as.integer(args[1]) # Trial ID
N = as.integer(args[2]) # number of training samples

mse = function(y, y.hat) {
    sum((y - y.hat)^2)/length(y)
}

calcDoF = function(y.hat, filename) {
    write.csv(y.hat, file.path(filename), row.names=FALSE)
    system(sprintf("python plateau_counter.py %s", filename))
    cnt = read.csv(filename, header=FALSE)$V1[1]
    system(sprintf("rm %s", filename))
    cnt
}

calcBIC = function(crisp.model, X.train, y.train, X.test, n.lambda) {
    loglikelihood = sapply(1:n.lambda, function(i) -0.5 * sum((y.train - predict(crisp.model, X.train, i))**2))
    DoF = sapply(1:n.lambda, function(i) calcDoF(predict(crisp.model, X.test, i), sprintf('%d_%d_vals.csv', trial, N)))
    print(DoF)
    -2 * loglikelihood + DoF * (log(dim(X.test)[1]) - log(2 * pi))
}

loadGapStatistic = function(N, trial) {
    as.integer(read.csv(sprintf('../../data/plateaus/results/gfl/%d/%d.csv', N, trial), header=FALSE)$V1[3])
}


q = loadGapStatistic(N, trial) # Use the same Q as the GapTV method
sprintf("q: %d", q)

max.lam = 10.0
lambda.min.ratio = 0.001
n.lambda = 50
lambda.seq = exp(seq(log(max.lam), log(max.lam * lambda.min.ratio), len = n.lambda))

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
sprintf("Found best lambda: %f", lambda.seq[best.lambda])
model = crisp(y, X, n.lambda=best.lambda, q=q, lambda.seq=lambda.seq[1:best.lambda])

# Estimate the best lambda via BIC
# model = crisp(y, X, n.lambda=n.lambda, q=q, lambda.seq=lambda.seq)
# BIC = calcBIC(model, X, y, X_test, n.lambda)
# best.lambda = which.min(BIC)

y_hat = predict(model, X_test, best.lambda)
write.csv(y_hat, file.path(sprintf("../../data/plateaus/predictions/gapcrisp/%d/%d.csv", N, trial)), row.names=FALSE)
rmse = sqrt(mean((y_test - y_hat)**2))
maxerr = max(abs(y_test - y_hat))

sprintf("CRISP --> q: %d lambda: %f RMSE: %f Max Error: %f", q, lambda.seq[best.lambda], rmse, maxerr)
write.csv(c(rmse, maxerr, q, lambda.seq[best.lambda]), file.path(sprintf("../../data/plateaus/results/gapcrisp/%d/%d.csv", N, trial)), row.names=FALSE)







