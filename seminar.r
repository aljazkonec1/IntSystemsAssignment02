install.packages(c("ggplot2","rpart", "rpart.plot", "CORElearn", "e1071", "randomForest", "kernlab","caret","mlbench"))
library(ggplot2)

#
# Getting and editing data
#

train <- read.table("train.csv", header = T, sep=",")
test <- read.table("test.csv", header = T, sep=",")

train$Id<-NULL
test$Id<- NULL

train$Class = as.factor(train$Class)
test$Class = as.factor(test$Class)

train[train=='NA'] <- NA
train <- train[!rowSums(is.na(train)) > 0, ] 

observed <- test$Class # the target variable is the "Class" attribute

{# The majority classifier
table(observed)
sum(observed=='1')/length(observed) } # 0.645933

# The classification accuracy
CA <- function(observed, predicted) {
  t <- table(observed, predicted)
  
  sum(diag(t)) / sum(t)
}

#
# Correlation of features
#

library(mlbench)
library(caret)

{# calculate correlation matrix
correlationMatrix <- cor(train[,1:41])
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)}

# getting rid of highly correlated columns
# 5, 7,	10,	11,	13,	15,	17,	22,	27,	29,	34,	39 

train <- train[-c(5, 7,	10,	11,	13,	15,	17,	22,	27,	29,	34,	39)]

#
# Importance of features
#

{control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(Class ~ ., data=train, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# plot importance
plot(importance)}

# getting rid of unimportant columns
# 2, 4,	9, 17, 19, 20, 21, 24, 25, 26, 28, 29, 32, 40

train <- train[-c(2, 4,	9, 17, 19, 20, 21,	24,	25,	26,	28,	29,	32,	40)]

#
# KNN
#

library(CORElearn)
{# k = 1
cm.knn <- CoreModel(Class ~ ., data = train, model="knn", kInNN = 1)
predicted <- predict(cm.knn, test, type="class")
CA(observed, predicted) }

{# k = 5
cm.knn <- CoreModel(Class ~ ., data = train, model="knn", kInNN = 5)
predicted <- predict(cm.knn, test, type="class")
CA(observed, predicted) }

{# k = 9
cm.knn <- CoreModel(Class ~ ., data = train, model="knn", kInNN = 9)
predicted <- predict(cm.knn, test, type="class")
CA(observed, predicted) }

{# k = 13
cm.knn <- CoreModel(Class ~ ., data = train, model="knn", kInNN = 13)
predicted <- predict(cm.knn, test, type="class")
CA(observed, predicted) }

#
# RANDOM FOREST
#

{library(randomForest)
rf <- randomForest(Class ~ ., data = train)
predicted <- predict(rf, test, type="class")
CA(observed, predicted) }

#
# NAIVE BAYES CLASSIFIER
#

{library(CORElearn)
cm.nb <- CoreModel(Class ~ ., data = train, model="bayes")
predicted <- predict(cm.nb, test, type="class")
CA(observed, predicted) }

#
# 2.3 calculations
#

Sensitivity <- function(observed, predicted, pos.class){
  t <- table(observed, predicted)
  
  t[pos.class, pos.class] / sum(t[pos.class,])
}

Specificity <- function(observed, predicted, pos.class){
  t <- table(observed, predicted)
  
  # identify the negative class name
  neg.class <- which(row.names(t) != pos.class)

  t[neg.class, neg.class] / sum(t[neg.class,])
}

x <- Sensitivity(bin.observed, bin.predicted, "1")
y <- Specificity(bin.observed, bin.predicted, "1")

fMeasure <- function(x,y){
  (2 * x * y) / (x + y) 
}

# ROC curve

library(pROC)
library(ggplot2)
library(dplyr)
library(ggrepel) # For nicer ROC visualization

bin.predMat <- predict(dt2, bin.test, type = "prob")

rocobj <- roc(bin.observed, bin.predMat[,"1"])
plot(rocobj)

#
# cross-validation
#

library(CORElearn)

folds <- 5

# kNN
{
for (k in 1:20){ ## pregledamo za vse k med 1 in 20
  acckNN <- c()
  fMeaskNN <- c()
  preckNN <- c()
  reckNN <- c()
  auckNN <- c()
  evalCore<-list()
  print(k)
  ## we will make 10 calculations
  for (i in 1:10){
    foldIdx <- cvGen(nrow(train), k=folds)
    
    ## for each calculation, we perform a cross validation on the train set.
    for (j in 1:folds) {
          
      ## select data for train and test (within the train data set!!)
      dTrain <- train[foldIdx!=j,]
      dTest  <- train[foldIdx==j,]
      
      ## train the model 
      modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = k) 
      
      ## predict on the test set (within the train set)
      predCore <- predict(modelCore, dTest)
      
      ## compute the metrics
      evalCore[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                                predictedClass=predCore$class, predictedProb=predCore$prob )
      
      ## cleanup
      destroyModels(modelCore)
    }
    
    ## aggregate the results
    results <- gatherFromList(evalCore)
    
    ## get mean performance across all folds
    meanPerformanceskNN <- sapply(results, mean)
    
    acckNN <- c(acckNN, meanPerformanceskNN['accuracy'])
    fMeaskNN <- c(fMeaskNN, meanPerformanceskNN['Fmeasure'])
    preckNN <- c(preckNN, meanPerformanceskNN['precision'])
    reckNN <- c(reckNN, meanPerformanceskNN['recall'])
    auckNN <- c(auckNN, meanPerformanceskNN['AUC'])

  }
  print(fMeaskNN)
  print(preckNN)
  print(reckNN)
  print(auckNN)
  print(acckNN)
}
}

# Bayes
{
evalCore<-list()
accBayes <- c()
fMeasBayes <- c()
precBayes <- c()
recBayes <- c()
aucBayes <- c()
for (i in 1:10){
  print(i)
  foldIdx <- cvGen(nrow(train), k=folds)

  for (j in 1:folds) {
    dTrain <- train[foldIdx!=j,]
    dTest  <- train[foldIdx==j,]
    
    modelCore <- CoreModel(Class ~ ., data = dTrain, model="bayes")
    
    predCore <- predict(modelCore, dTest)
    
    evalCore[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                               predictedClass=predCore$class, predictedProb=predCore$prob )
    
    destroyModels(modelCore)
  }
  
  results <- gatherFromList(evalCore)
  
  meanPerformancesBayes <- sapply(results, mean)
  
  accBayes <- c(accBayes, meanPerformancesBayes['accuracy'])
  fMeasBayes <- c(fMeasBayes, meanPerformancesBayes['Fmeasure'])
  precBayes <- c(precBayes, meanPerformancesBayes['precision'])
  recBayes <- c(recBayes, meanPerformancesBayes['recall'])
  aucBayes <- c(aucBayes, meanPerformancesBayes['AUC'])

}
print(fMeasBayes)
print(precBayes)
print(recBayes)
print(aucBayes)
print(accBayes)
}

# random forest
{
evalCore<-list()
accRndForest <- c()
fMeasRndForest <- c()
precRndForest <- c()
recRndForest <- c()
aucRndForest <- c()
for (i in 1:10){
  print(i)
  foldIdx <- cvGen(nrow(train), k=folds)

  for (j in 1:folds) {
    dTrain <- train[foldIdx!=j,]
    dTest  <- train[foldIdx==j,]
    
    modelCore <- CoreModel(Class ~ ., data = dTrain, model="rf")
    
    predCore <- predict(modelCore, dTest)
    
    evalCore[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                               predictedClass=predCore$class, predictedProb=predCore$prob )
    
    destroyModels(modelCore)
  }
  
  results <- gatherFromList(evalCore)
  
  meanPerformancesRndForest <- sapply(results, mean)
  
  accRndForest <- c(accRndForest, meanPerformancesRndForest['accuracy'])
  fMeasRndForest <- c(fMeasRndForest, meanPerformancesRndForest['Fmeasure'])
  precRndForest <- c(precRndForest, meanPerformancesRndForest['precision'])
  recRndForest <- c(recRndForest, meanPerformancesRndForest['recall'])
  aucRndForest <- c(aucRndForest, meanPerformancesRndForest['AUC'])

}
print(fMeasRndForest)
print(precRndForest)
print(recRndForest)
print(aucRndForest)
print(accRndForest)
}
