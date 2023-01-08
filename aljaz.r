# sum(train$Class == 2)/ (sum(train$Class == 1) + sum(train$Class == 2))

# train[train=='NA'] <- NA
# new_train <- train[!rowSums(is.na(train)) > 0, ]
# new_train

# plot(new_train$V36, col=new_train$Class)
# na_train <- train[rowSums(is.na(train)) > 0, ]
# plot(na_train[, c("V1")], col=na_train$Class)

# sum(new_train$Class == 2)
# sum(new_train$Class == 1)



# for (i in colnames(train)) {
    
#     name <- paste(i, ".png", sep="")
#     print(name)
#     png(filename=name)
#     plot(train[, c(i)], col=train$Class)
#     dev.off()

# }
# train

# png(filename="V23.png")
# plot(train[train$V23 < 100, c("V23")], col=train$Class)
# dev.off()

# t <- train[train$V23 < 100, ]

# png(filename="boxplot_on_t.png")
# boxplot(t[t$Class ==1, ])
# dev.off()

# n_train <- new_train
# for (i in 1:41) {
#     n_train[, i] <- new_train[, i] / max(new_train[, i])
# }

# head(n_train)
# col <- c("red", "blue")
# col <- col[as.numeric(n_train$Class)]

library(ggplot2)
library(dplyr)
library(CORElearn)
library(mlbench)
library(caret)



train <- read.table(file="train.csv", sep=",", header= TRUE)
test <- read.table(file="test.csv", sep=",", header= TRUE)
# summary(train)

# odstanimo stolpce V2, 4, 9, 17, 19, 21, 24, 26, 28, 29, 32, 35, 40
train[train == 'NA'] <- NA

CA <- function(observed, predicted){
  t <- table(observed, predicted)
  print(t)
  sum(diag(t)) / sum(t)
}

observed <- test$Class
subset_procenti <- c()

for (i in 3:16){
    knn <- CoreModel(Class ~ ., data= train, model="knn", kInNN= i )
    predicted <- predict(knn, test, type="class" )
    procenti[i-3] <- CA(observed, predicted) 

    subset_knn <- CoreModel(Class ~ ., data= reduced_train, model="knn", kInNN= i )
    predicted <- predict(subset_knn, test, type="class" )
    subset_procenti[i-3] <- CA(observed, predicted)
}


png(filename="Knn classifier.png", width= 1920, height= 1080)
plot(3:15,procenti*100, type="b",xlab="k", ylab="Classification accuracy in %",col="blue", tck=1, ylim=c(75, 87))
lines(3:15,subset_procenti*100, type="b",col="red")
title(main="KNN classifier for k in range 3:15")
legend(3, 85, legend=c("KNN on all features", "KNN on subset of features"), col=c("blue", "red"), lty=1, cex=1.5)
dev.off()

naiveBayes <- CoreModel(Class ~ ., data= train, model="bayes")
predicted <- predict(naiveBayes, test, type="class")
predicted
CA(observed, predicted)








correlationMatrix <- cor(train[,1:41])
print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(highlyCorrelated)

# getting rid of highly correlated columns
# 5, 7,	10,	11,	13,	15,	17,	22,	27,	29,	34,	39 

train <- train[-c(5, 7,	10,	11,	13,	15,	17,	22,	27,	29,	34,	39)]

control <- trainControl(method="repeatedcv", number=10, repeats=3)

# train the model
model <- train(Class ~ ., data=train, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# plot importance
plot(importance)

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

#
# ROC curve
#

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

# kNN, k = 1
{
evalCore<-list()
acckNN1 <- c()
fMeaskNN1 <- c()
preckNN1 <- c()
reckNN1 <- c()
auckNN1 <- c()
## we will make 10 calculations
for (i in 1:10){
  print(i)
  foldIdx <- cvGen(nrow(train), k=folds)
  
  ## for each calculation, we perform a cross validation on the train (learn) set.
  for (j in 1:folds) {
        
    ## select data from train and test (within the train data set!!)
    dTrain <- train[foldIdx!=j,]
    dTest  <- train[foldIdx==j,]
    
    ## train the model 
    modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 1) 
    
    ## predict on the test set (within the learn set)
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
  meanPerformanceskNN1 <- sapply(results, mean)
  
  acckNN1 <- c(acckNN1, meanPerformanceskNN1['accuracy'])
  fMeaskNN1 <- c(fMeaskNN1, meanPerformanceskNN1['Fmeasure'])
  preckNN1 <- c(preckNN1, meanPerformanceskNN1['precision'])
  reckNN1 <- c(reckNN1, meanPerformanceskNN1['recall'])
  auckNN1 <- c(auckNN1, meanPerformanceskNN1['AUC'])

}
print(fMeaskNN1)
print(preckNN1)
print(reckNN1)
print(auckNN1)
print(acckNN1)
}

# kNN, k = 5
{
evalCore<-list()
acckNN5 <- c()
fMeaskNN5 <- c()
preckNN5 <- c()
reckNN5 <- c()
auckNN5 <- c()
for (i in 1:10){
  print(i)
  foldIdx <- cvGen(nrow(train), k=folds)

  for (j in 1:folds) {
    dTrain <- train[foldIdx!=j,]
    dTest  <- train[foldIdx==j,]
    
    modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 5) 
    
    predCore <- predict(modelCore, dTest)
    
    evalCore[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                               predictedClass=predCore$class, predictedProb=predCore$prob )
    
    destroyModels(modelCore)
  }
  
  results <- gatherFromList(evalCore)
  
  meanPerformanceskNN5 <- sapply(results, mean)
  
  acckNN5 <- c(acckNN1, meanPerformanceskNN5['accuracy'])
  fMeaskNN5 <- c(fMeaskNN1, meanPerformanceskNN5['Fmeasure'])
  preckNN5 <- c(preckNN1, meanPerformanceskNN5['precision'])
  reckNN5 <- c(reckNN1, meanPerformanceskNN5['recall'])
  auckNN5 <- c(auckNN1, meanPerformanceskNN5['AUC'])
}

print(fMeaskNN5)
print(preckNN5)
print(reckNN5)
print(auckNN5)
print(acckNN5)
}

# kNN, k = 9
{
evalCore<-list()
acckNN9 <- c()
fMeaskNN9 <- c()
preckNN9 <- c()
reckNN9 <- c()
auckNN9 <- c()
for (i in 1:10){
  print(i)
  foldIdx <- cvGen(nrow(train), k=folds)

  for (j in 1:folds) {
    dTrain <- train[foldIdx!=j,]
    dTest  <- train[foldIdx==j,]
    
    modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 9) 
    
    predCore <- predict(modelCore, dTest)
    
    evalCore[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                               predictedClass=predCore$class, predictedProb=predCore$prob )
    
    destroyModels(modelCore)
  }
  
  results <- gatherFromList(evalCore)
  
  meanPerformanceskNN9 <- sapply(results, mean)
  
  acckNN9 <- c(acckNN1, meanPerformanceskNN9['accuracy'])
  fMeaskNN9 <- c(fMeaskNN1, meanPerformanceskNN9['Fmeasure'])
  preckNN9 <- c(preckNN1, meanPerformanceskNN9['precision'])
  reckNN9 <- c(reckNN1, meanPerformanceskNN9['recall'])
  auckNN9 <- c(auckNN1, meanPerformanceskNN9['AUC'])
}
print(fMeaskNN9)
print(preckNN9)
print(reckNN9)
print(auckNN9)
print(acckNN9)
}

# kNN, k = 13
{
evalCore<-list()
acckNN1 <- c()
fMeaskNN1 <- c()
preckNN1 <- c()
reckNN1 <- c()
auckNN1 <- c()

for (i in 1:10){
  print(i)
  foldIdx <- cvGen(nrow(train), k=folds)

  for (j in 1:folds) {
    dTrain <- train[foldIdx!=j,]
    dTest  <- train[foldIdx==j,]
    
    modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 13) 
    
    predCore <- predict(modelCore, dTest)
    
    evalCore[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                               predictedClass=predCore$class, predictedProb=predCore$prob )
    
    destroyModels(modelCore)
  }
  
  results <- gatherFromList(evalCore)
  
  meanPerformanceskNN13 <- sapply(results, mean)
  
  acckNN13 <- c(acckNN13, meanPerformanceskNN13['accuracy'])
  fMeaskNN13 <- c(fMeaskNN13, meanPerformanceskNN13['Fmeasure'])
  preckNN13 <- c(preckNN13, meanPerformanceskNN13['precision'])
  reckNN13 <- c(reckNN13, meanPerformanceskNN13['recall'])
  auckNN13 <- c(auckNN13, meanPerformanceskNN13['AUC'])
}
print(fMeaskNN13)
print(preckNN13)
print(reckNN13)
print(auckNN13)
print(acckNN13)
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

