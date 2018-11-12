library(Matrix)
library(genlasso)
library(caret)

source("CRISP_functions.R")

args = commandArgs(trailingOnly = TRUE)
dataset.name = args[1] # The name of the dataset
x1 = as.numeric(args[2]) # The first column of the data file
x2 = as.numeric(args[3]) # The second column of the data file
q = as.numeric(args[4]) # The number of splits per dimension
k = as.numeric(args[5]) # The trend filtering order

train_data = as.matrix(read.csv(file.path(sprintf("../data/%s_train.csv", dataset.name)), header=FALSE))
test_data = as.matrix(read.csv(file.path(sprintf("../data/%s_test.csv", dataset.name)), header=FALSE))
X = train_data[, c(x1, x2)]
y = as.numeric(as.vector(train_data[, ncol(train_data)]))
X_test = test_data[, c(x1, x2)]
n = length(y)

max.lam = 10.0
lambda.min.ratio = 0.001
n.lambda = 50
lambda.seq = exp(seq(log(max.lam), log(max.lam * lambda.min.ratio), len = n.lambda))

# Get the selector matrix for beta
q.seq = c(0, 1:(q-1)/q, 1)
block.X1 = findIntervalOverlaps(X[,1], quantile(X[,1], q.seq, type=8), all.inside=T)
block.X2 = findIntervalOverlaps(X[,2], quantile(X[,2], q.seq, type=8), all.inside=T)
blocks = block.X1 + q * (block.X2 - 1)
Q = as.matrix(sparseMatrix(i=1:nrow(X), j=blocks, dims=c(nrow(X),q^2)))

# Get the trend filtering penalty matrix
D = getD2dSparse(q, q)
Dx = D
for (i in 1:k){
    if (i %% 2 == 0){
        Dx = D %*% Dx
    }
    else{
        Dx = t(D) %*% Dx
    }
}

# Estimate the best lambda via k-fold cross validation
num_folds = 5
folds <- createFolds(1:length(y), k=num_folds)
prederr = rep(0, n.lambda)
for (fold in folds){
    foldmodel = genlasso(y[-fold], Q[-fold,], Dx)
    prederr = prederr + sapply(lambda.seq, function(lam) mse(y[fold], Q[fold,] %*% foldmodel$beta[,which(abs(foldmodel$lambda - lam)==min(abs(foldmodel$lambda - lam)))]))
}
best.lambda = which.min(prederr)

fit = genlasso(y, Q, Dx)
beta = fit$beta[,which(abs(fit$lambda - lambda.seq[best.lambda]) == min(abs(fit$lambda - lambda.seq[best.lambda])))]

# Predict the test data
x1.sort = sort(X[,1]); x2.sort = sort(X[,2])
new.block.X1 = findIntervalOverlaps(X_test[,1], quantile(x1.sort, q.seq, type=8), all.inside=T)
new.block.X2 = findIntervalOverlaps(X_test[,2], quantile(x2.sort, q.seq, type=8), all.inside=T)
closest = sapply(1:nrow(X_test), function(i) get.cell(block.X1[i], block.X2[i], q))
predictions = beta[closest]
write.csv(predictions, file.path(sprintf("../data/gtf%d_predictions_%s.csv", k, dataset.name)), row.names=FALSE)











