setwd("D:\\DOCS\\Kaggle\\Smartphone")
library(caret)
library(foreach)
library(doParallel)
library(h2o)
training = read.csv("train.csv", header = T, na.string=c("","NA"));
testing = read.csv("test.csv", header = T, na.string=c("","NA"));

#Preprocessing
#Remove the first id column
training = training[,-1]
ID = testing[,1]
testing = testing[,-1]
nzv_features = nearZeroVar(training, uniqueCut = 50)
nzv_table = nearZeroVar(training, saveMetrics = T)

#Removing near zero variance features from the dataset
training = training[,-nzv_features]
testing = testing[,-nzv_features]

#Check correlated features
M = abs(cor(training[,-ncol(training)]))
diag(M) = 0
corrCols = unique(which(M>0.9, arr.ind = TRUE)[,2])
training = training[,-corrCols]
testing = testing[,-corrCols]

#scaling between 0-1
features = c(1:(ncol(training)-1))
training[,features] = apply(training[,features], 2, function(col) (col-mean(col))/(max(col)-min(col)) )

#unit variance
features = c(1:(ncol(training)-1))
training[,features] = apply(training[,features], 2, function(col) (col-mean(col))/sd(col))

#Partition data on training and cross-validation sets
inTrain = createDataPartition(y=training$activity, p=0.7, list=FALSE)
cv = training[-inTrain,]
training = training[inTrain,]


#add a constant and take sqrt
training[,features] = sqrt(training[,features] + (2))
cv[,features] = sqrt(cv[,features] + (2))

localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '6g',nthreads = 3)
training_h2o <- as.h2o(localH2O, training, key = 'train')
cv_h2o = as.h2o(localH2O, cv, key = 'cv')
testing_h2o <- as.h2o(localH2O, testing, key = 'test')
init.time <- Sys.time()
model <- h2o.deeplearning(x = 1:(ncol(training_h2o)-1),  # column numbers for predictors
                          y = "activity",   # column number for label
                          data = training_h2o, # data in H2O format
                          activation="RectifierWithDropout",
                          classification = T,
                          input_dropout_ratio = 0.25, # % of inputs dropout
                          hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
                          #balance_classes = TRUE, 
                          #variable_importance = T,
                          l1=1e-5,
                          l2=1e-5,
                          hidden = c(512,512,512), # three layers of 50 nodes
                          epochs = 50) # max. no. of epochs
h2o_prediction <- as.data.frame(h2o.predict(model, cv_h2o))
confusionMatrix(h2o_prediction$predict, cv$activity)
Sys.time() - init.time

h2o_prediction <- as.data.frame(h2o.predict(model, training_h2o))
confusionMatrix(h2o_prediction$predict, training$activity)
#random forest
init.time <- Sys.time()
model = h2o.randomForest(x = 1:(ncol(training_h2o)-1),  # column numbers for predictors
                         y = "activity",   # column number for label
                         data = training_h2o, # data in H2O format
                         classification = T,
                         ntree = c(200),
                         depth = c(100),
                         mtries = c(25))
h2o_prediction <- as.data.frame(h2o.predict(model, cv_h2o))
confusionMatrix(h2o_prediction$predict, cv$activity)
Sys.time() - init.time

load("rf_500_150_105")
#Apply the model on a test set
predictions_testing = as.data.frame(h2o.predict(model, testing_h2o))
submission = format(round(data.frame(ID,predictions_testing$predict),1),nsmall=1)

submission_old = read.csv(file = "submission2.csv",header = T)
sum(as.integer(submission[,2])!=as.integer(submission_old[,2])) #135
write("ID,activity",file = "submission.csv")
write.table(submission, file="submission.csv", sep=",",row.names = F,col.names=F,quote = F,append = T)
write(as.matrix(predictions_testing$predict), file="submission.csv")


h2o.shutdown(client = localH2O,prompt = F)
