setwd("D:\\DOCS\\Kaggle\\Smartphone")
library(caret)
library(foreach)
library(doParallel)
training = read.csv("train.csv", header = T, na.string=c("","NA"));
testing = read.csv("test.csv", header = T, na.string=c("","NA"));

#Preprocessing
#Remove the first id column
training = training[,-1]
training[,ncol(training)] = as.factor(training[,ncol(training)])
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
#inTrain = createDataPartition(y=training$activity, p=0.7, list=FALSE)
#cv = training[-inTrain,]
#training = training[inTrain,]

#add a constant and take sqrt
training[,features] = sqrt(training[,features] + (2))
cv[,features] = sqrt(cv[,features] + (2))

#Reducing the dimentionality to 2D and plotting the dataset
prComp = prcomp(training[,-ncol(training)], scale=TRUE)
plot(prComp$x[,1], prComp$x[,2], col=training$activity, type="p", cex=0.5)



#try C5.0
modelInfo = getModelInfo(model="C5.0")
modelInfo$C5.0$grid

cl <- makeCluster(3)
registerDoParallel(cl)
trainControl = trainControl(method = "repeatedcv", 
                            number = 10, 
                            repeats = 3, 
                            verboseIter = T, 
                            allowParallel = T)
grid = expand.grid(model="rule", trials=20, winnow=F)
modelFit = train(activity ~., 
                 data=training, 
                 method="C5.0",
                 #tuneGrid = grid, 
                 trControl = trainControl, 
                 metric="Accuracy")

predictions = predict(modelFit, newdata = cv)
confusionMatrix(predictions, cv$activity)

stopCluster(cl)

#try CART
modelInfo = getModelInfo(model="rpart")
modelInfo$rpart$grid

cl <- makeCluster(3)
registerDoParallel(cl)
trainControl = trainControl(method = "repeatedcv", 
                            number = 3, 
                            repeats = 1, 
                            verboseIter = T, 
#                            classProbs = T, 
                            allowParallel = T)
grid = expand.grid(cp=0.000005)
modelFit = train(activity ~., 
                 data=training, 
                 method="rpart",
#                 tuneGrid = grid, 
                 trControl = trainControl, 
                 metric="Accuracy")
predictions = predict(modelFit, newdata = cv)
confusionMatrix(predictions, cv$activity)
stopCluster(cl)

#random forest
cl <- makeCluster(3)
registerDoParallel(cl)
trainControl = trainControl(method = "oob", 
                            #number = 10, 
                            #repeats = 3, 
                            #summaryFunction = twoClassSummary,
                            verboseIter = T, 
                            allowParallel = T)
grid = expand.grid(mtry=c(22))
modelFit = train(activity ~., 
                 data=training, 
                 method="rf",
                 preProcess = c("center","scale"),
                 #PCAthresh = 0.99,
                 ntree = 2000,
                 tuneGrid = grid, 
                 trControl = trainControl, 
                 metric="Kappa")
predictions = predict(modelFit, newdata = training)
confusionMatrix(predictions, training$activity)
stopCluster(cl)

#svm
cl <- makeCluster(3)
registerDoParallel(cl)
trainControl = trainControl(method = "repeatedcv", 
                            number = 3, 
                            repeats = 2, 
                            verboseIter = T, 
                            allowParallel = T)
grid = expand.grid(cp=0.000005)
modelFit = train(activity ~., 
                 data=training, 
                 method="lssvmRadial",
                 preProcess = c("scale","center"),
                 #PCAthresh = 0.99,
                 #                 tuneGrid = grid, 
                 trControl = trainControl, 
                 metric="Accuracy")
predictions = predict(modelFit, newdata = cv)
confusionMatrix(predictions, cv$activity)
stopCluster(cl)


predictions_testing = predict(modelFit, testing)
submission = format(round(data.frame(as.numeric(ID),as.numeric(predictions_testing)),1),nsmall=1)

submission_old = read.csv(file = "submission2.csv",header = T)
sum(as.integer(submission[,2])!=as.integer(submission_old[,2])) #135
write("ID,activity",file = "submission.csv")
write.table(submission, file="submission.csv", sep=",",row.names = F,col.names=F,quote = F,append = T)

