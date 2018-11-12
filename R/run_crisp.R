library(foreach)
library(Matrix)
library(caret)
library(crisp)

args = commandArgs(trailingOnly = TRUE)
dataset.name = args[1] # The name of the dataset
x1 = as.numeric(args[2]) # The first column of the data file
x2 = as.numeric(args[3]) # The second column of the data file
q = as.numeric(args[4]) # The number of splits per dimension

max.lam = 10.0
lambda.min.ratio = 0.001
n.lambda = 50
lambda.seq = exp(seq(log(max.lam), log(max.lam * lambda.min.ratio), len = n.lambda))

train_data = as.matrix(read.csv(file.path(sprintf("../data/%s_train.csv", dataset.name)), header=FALSE))
test_data = as.matrix(read.csv(file.path(sprintf("../data/%s_test.csv", dataset.name)), header=FALSE))
X = train_data[, c(x1, x2)]
y = as.numeric(as.vector(train_data[, ncol(train_data)]))
X_test = test_data[, c(x1, x2)]

y_mean = mean(y)
y_stdev = sd(y)
y = (y - y_mean) / y_stdev

# Don't make q larger than n
q = min(q, length(train_data))

# Estimate the best lambda via k-fold cross validation
k = 5
folds <- createFolds(1:length(y), k=k)
prederr = rep(0, n.lambda)
for (fold in folds){
    foldmodel = crisp(y[-fold], X[-fold,], n.lambda=n.lambda, q=q, lambda.seq=lambda.seq)
    prederr = prederr + sapply(1:n.lambda, function(i) mse(y[fold], predict.crisp(foldmodel, X[fold,], i)))
}
best.lambda = which.min(prederr)
if (best.lambda == 1){
    lambda.seq = c(max.lam+1, max.lam)
    best.lambda = 2
}
model = crisp(y, X, n.lambda=best.lambda, q=q, lambda.seq=lambda.seq[1:best.lambda])
predictions = predict.crisp(model, X_test, best.lambda)
predictions = (predictions * y_stdev) + y_mean
write.csv(predictions, file.path(sprintf("../data/crisp_predictions_%s.csv", dataset.name)), row.names=FALSE)
