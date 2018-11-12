library(Matrix)
library(fields)

args = commandArgs(trailingOnly = TRUE)
dataset.name = args[1] # The name of the dataset
x1 = as.numeric(args[2]) # The first column of the data file
x2 = as.numeric(args[3]) # The second column of the data file

train_data = as.matrix(read.csv(file.path(sprintf("../data/%s_train.csv", dataset.name)), header=FALSE))
test_data = as.matrix(read.csv(file.path(sprintf("../data/%s_test.csv", dataset.name)), header=FALSE))
X = train_data[, c(x1, x2)]
y = as.numeric(as.vector(train_data[, ncol(train_data)]))
X_test = test_data[, c(x1, x2)]

fit = Tps(X, y)
predictions = predict(fit, X_test)
write.csv(predictions, file.path(sprintf("../data/tps_predictions_%s.csv", dataset.name)), row.names=FALSE)