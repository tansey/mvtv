library(foreach)
library(Matrix)
library(caret)

source("CRISP_functions.R")

scenarios = matrix(c(c(1, 100), c(2, 100), c(3, 100), c(4, 100),
              c(1, 500), c(2, 500), c(3, 500), c(4, 500)), nrow=8, ncol=2, byrow=TRUE)

q = 100
max.lam = 10.0
lambda.min.ratio = 0.001
n.lambda = 50
lambda.seq = exp(seq(log(max.lam), log(max.lam * lambda.min.ratio), len = n.lambda))

out_grid = foreach(sidx = 1:8) %dopar% {
    s = scenarios[sidx,1]
    n = scenarios[sidx,2]
    print(s)
    print(n)
    X = as.matrix(read.csv(file.path(sprintf("../data/x_scenario%d_n%d.csv", s, n)), header=FALSE))
    y = as.numeric(as.matrix(read.csv(file.path(sprintf("../data/y_scenario%d_n%d.csv", s, n)), header=FALSE)))
    X_test = as.matrix(read.csv(file.path(sprintf("../data/test_x_scenario%d_n%d.csv", s, n)), header=FALSE))
    
    # Estimate the best lambda via k-fold cross validation
    k = 5
    folds <- createFolds(1:length(y), k=k)
    prederr = rep(0, n.lambda)
    for (fold in folds){
        foldmodel = crisp(y[-fold], X[-fold,], n.lambda=n.lambda, q=q, lambda.seq=lambda.seq)
        prederr = prederr + sapply(1:n.lambda, function(i) mse(y[fold], predict.crisp(foldmodel, X[fold,], i)))
    }
    best.lambda = which.min(prederr)
    model = crisp(y, X, n.lambda=n.lambda-best.lambda+1, q=q, lambda.seq=lambda.seq[1:best.lambda])
    beta = model$M.hat.list[which.min(model$obj.vec)]
    predictions = predict.crisp(model, X_test, best.lambda)
    write.csv(beta, file.path(sprintf("../data/crisp_beta_scenario%d_n%d.csv", s, n)), col.names=FALSE, row.names=FALSE)
    write.csv(predictions, file.path(sprintf("../data/crisp_predictions_scenario%d_n%d.csv", s, n)), col.names=FALSE, row.names=FALSE)

    model
}

