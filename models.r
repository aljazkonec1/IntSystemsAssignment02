install.packages(c("ggplot2","rpart", "rpart.plot", "CORElearn", "e1071", "randomForest", "kernlab","caret","mlbench"))
install.packages("ggrepel")
library(ggplot2)
library(mlbench)
library(caret)
library(CORElearn)
library(randomForest)
library(pROC)
library(dplyr)
library(ggrepel) # For nicer ROC visualization
#
# Getting and editing data
#
{
train <- read.table("train.csv", header = T, sep=",")
test <- read.table("test.csv", header = T, sep=",")

train$Id<-NULL
test$Id<- NULL

train$Class = as.factor(train$Class)
test$Class = as.factor(test$Class)

train[train=='NA'] <- NA
train <- train[!rowSums(is.na(train)) > 0, ] 
observed <- test$Class # the target variable is the "Class" attribute
}


# The classification accuracy
CA <- function(observed, predicted) {
  t <- table(observed, predicted)
  
  sum(diag(t)) / sum(t)
}


{# calculate correlation matrix
correlationMatrix <- cor(train[,1:41])

highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)

print(highlyCorrelated)}

# getting rid of highly correlated columns
# 5, 7,	10,	11,	13,	15,	17,	22,	27,	29,	34,	39 
trainWOCorelated <- train[-c(5, 7,	10,	11,	13,	15,	17,	22,	27,	29,	34,	39)]


control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(Class ~ ., data=trainWOCorelated, method="lvq", preProcess="scale", trControl=control)
importance <- varImp(model, scale=FALSE)
plot(importance)


# train all Features
# trainWOCorelated = train without Highly corelated features
# trainWOUnimportant = train without Unimportant features
# trainSheared = train without Higly Corelated features and also Unimportant features
# we will test KNN, RandomForrest, NaiveBayes on all 3 data sets, and compare the best performance


# getting rid of unimportant columns
# 2, 4,	9, 17, 19, 20, 21, 24, 25, 26, 28, 29, 32, 40
trainSheared <- trainWOCorelated[-c(2, 4,	9, 17, 19, 20, 21,	24,	25,	26,	28,	29,	32,	40)]
trainWOUnimportant <- train[-c(2, 4,	9, 17, 19, 20, 21,	24,	25,	26,	28,	29,	32,	40)]

#
# KNN
# KNN will be teseted on N = 3, ... , 15

train_procenti <- c()
trainWOCorelated_procenti <- c()
trainWOUnimportant_procenti <- c()
trainSheared_procenti <- c()

for (i in 3:16){
    knn <- CoreModel(Class ~ ., data= train, model="knn", kInNN= i )
    predicted <- predict(knn, test, type="class" )
    train_procenti[i-3] <- CA(observed, predicted) 
    destroyModels(knn)

    subset_knn <- CoreModel(Class ~ ., data= trainWOCorelated, model="knn", kInNN= i )
    predicted <- predict(subset_knn, test, type="class" )
    trainWOCorelated_procenti[i-3] <- CA(observed, predicted)
    destroyModels(knn)

    subset_knn <- CoreModel(Class ~ ., data= trainWOUnimportant, model="knn", kInNN= i )
    predicted <- predict(subset_knn, test, type="class" )
    trainWOUnimportant_procenti[i-3] <- CA(observed, predicted)
    destroyModels(knn)

    subset_knn <- CoreModel(Class ~ ., data= trainSheared, model="knn", kInNN= i )
    predicted <- predict(subset_knn, test, type="class" )
    trainSheared_procenti[i-3] <- CA(observed, predicted)
    destroyModels(knn)

}

# lets make a graph for KNN..
{
png(filename="Knn classifier.png", width= 1920, height= 1080)
plot(3:15,train_procenti*100, type="b",xlab="k", ylab="Classification accuracy (%)",col="#197ed6", tck=1, lwd= 2 ,ylim=c(78, 87))
lines(3:15,trainWOCorelated_procenti*100, type="b",col="#c16ec9")
lines(3:15,trainWOUnimportant_procenti*100, type="b",col="#ff6a89")
lines(3:15,trainSheared_procenti*100, type="b",col="#ff9742")

title(main="KNN classifier for k in range 3:15")
legend(3, 87, legend=c("KNN on all features", "KNN without highly correlated of features",
        "KNN without unimportant features", "KNN without highly correlated and unimportant features" ), col=c("#197ed6", "#c16ec9", "#ff6a89", "#ff9742"), lty=1, cex=1.5)
dev.off()

}




#
# RANDOM FOREST
#
percentages <- c()
{
rf <- randomForest(Class ~ ., data = train)
predicted <- predict(rf, test, type="class")
percentages[1] <- CA(observed, predicted) 
destroyModels(rf)

rf <- randomForest(Class ~ ., data = trainWOCorelated)
predicted <- predict(rf, test, type="class")
percentages[2] <- CA(observed, predicted) 
destroyModels(rf)

rf <- randomForest(Class ~ ., data = trainWOUnimportant)
predicted <- predict(rf, test, type="class")
percentages[3] <- CA(observed, predicted) 
destroyModels(rf)

rf <- randomForest(Class ~ ., data = trainSheared)
predicted <- predict(rf, test, type="class")
percentages[4] <- CA(observed, predicted) 
destroyModels(rf)

}

names <- c("Full Features","Without high correlation", "Only important features", "Without high correlation and unimportant features")
val <- percentages*100
d <- data.frame(names, val)
d$names <- factor(d$names, levels = d$names)

{
  p <- ggplot(d, aes(x=names, y=val)) + geom_bar(stat="identity",fill= c("#197ed6", "#c16ec9", "#ff6a89", "#ff9742"),  color= c("#197ed6", "#c16ec9", "#ff6a89", "#ff9742")) 
  p <- p + coord_cartesian(ylim=c(80, 85)) + xlab("Types of feature selections") + ylab("Classification accuracy (%)") + ggtitle("Random Forrest Classification Accuracy")
  ggsave("RandomForrest.png", plot=p, dpi= 1200)
}

#
# NAIVE BAYES CLASSIFIER
#
percentages <- c()
{
naiveBayes <- CoreModel(Class ~ ., data = train, model="bayes")
predicted <- predict(naiveBayes, test, type="class")
percentages[1] <- CA(observed, predicted) 
destroyModels(naiveBayes)

naiveBayes <- CoreModel(Class ~ ., data = trainWOCorelated, model="bayes")
predicted <- predict(naiveBayes, test, type="class")
percentages[2] <- CA(observed, predicted) 
destroyModels(naiveBayes)

naiveBayes <- CoreModel(Class ~ ., data = trainWOUnimportant, model="bayes")
predicted <- predict(naiveBayes, test, type="class")
percentages[3] <- CA(observed, predicted) 
destroyModels(naiveBayes)

naiveBayes <- CoreModel(Class ~ ., data = trainSheared, model="bayes")
predicted <- predict(naiveBayes, test, type="class")
percentages[4] <- CA(observed, predicted) 
destroyModels(naiveBayes)

}
# {png(filename="NaiveBayes.png", width= 1920, height= 1080)
# barplot(percentages*100, main = "Classification accuracy of Naive Bayes", ylab="Classification accuracy (%)", names.arg = c("Full Features","Without high correlation", "Only important features", "Without high correlation and unimportant features"),
#         col=c("#197ed6", "#c16ec9", "#ff6a89", "#ff9742"), ylim= c(75, 85))
# dev.off() }

val <- percentages*100
d <- data.frame(names, val)
d$names <- factor(d$names, levels = d$names)

{
  p <- ggplot(d, aes(x=names, y=val)) + geom_bar(stat="identity",fill= c("#197ed6", "#c16ec9", "#ff6a89", "#ff9742"),  color= c("#197ed6", "#c16ec9", "#ff6a89", "#ff9742")) 
  p <- p  + xlab("Types of feature selections") + coord_cartesian(ylim=c(75, 82)) + ylab("Classification accuracy (%)") + ggtitle("Naive Bayes Classification Accuracy")
  ggsave("NaiveBayes.png", plot=p, dpi= 1200)
}



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

fMeasure <- function(x,y){
  (2 * x * y) / (x + y) 
}


#
# ROC curve ??????????????????????????????????????
#


bin.predMat <- predict(dt2, bin.test, type = "prob")

rocobj <- roc(bin.observed, bin.predMat[,"1"])
plot(rocobj)

#
# cross-validation
#

folds <- 5
acc <- c(0,0,0,0)
fMes <- c(0,0,0,0)
precision <- c(0,0,0,0)
recall <- c(0,0,0,0)
auc <- c(0,0,0,0)

evalCore<-list()
evalCoreWOCorelated <- list()
evalCoreWOUnimportant <- list()
evalCoreSheared <- list()

acckNN1 <- c()
fMeaskNN1 <- c()
preckNN1 <- c()
reckNN1 <- c()
auckNN1 <- c()
## we will make 10 calculations
# for (j in 3:15) {
  #knn 9 

  for (i in 1:10){

    foldIdx <- cvGen(nrow(train), k=folds)
    for (j in 1:folds) {
      dTrain <- train[foldIdx!=j,]
      dTest  <- train[foldIdx==j,]
      
      modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 9) 
      predCore <- predict(modelCore, dTest)
      
      evalCore[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                                predictedClass=predCore$class, predictedProb=predCore$prob )

      ##
      dTrain <- trainWOCorelated[foldIdx!=j,]
      dTest  <- trainWOCorelated[foldIdx==j,]
      
      modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 9) 
      predCore <- predict(modelCore, dTest)
      
      evalCoreWOCorelated[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                                predictedClass=predCore$class, predictedProb=predCore$prob )


      ##
      dTrain <- trainWOUnimportant[foldIdx!=j,]
      dTest  <- trainWOUnimportant[foldIdx==j,]
      
      modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 9) 
      predCore <- predict(modelCore, dTest)
      
      evalCoreWOUnimportant[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                                predictedClass=predCore$class, predictedProb=predCore$prob )

      ##
      dTrain <- trainSheared[foldIdx!=j,]
      dTest  <- trainSheared[foldIdx==j,]
      
      modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 9) 
      predCore <- predict(modelCore, dTest)
      
      evalCoreSheared[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                                predictedClass=predCore$class, predictedProb=predCore$prob )
      
      destroyModels(modelCore)
    }
  
  
  results <- gatherFromList(evalCore)
  meanPerf <- sapply(results, mean)
  if (meanPerf['accuracy'] > acc[1] ) {
    acc[1] <- meanPerf['accuracy']
    fMes[1] <- meanPerf['Fmeasure']
    precision[1] <- meanPerf['precision']
    recall[1]<- meanPerf['recall']
    auc[1]<- meanPerf['AUC']
  }


  results <- gatherFromList(evalCoreWOUnimportant)
  meanPerf <- sapply(results, mean)
  if(meanPerf['accuracy'] > acc[2]) {
    acc[2] <- meanPerf['accuracy']
    fMes[2] <- meanPerf['Fmeasure']
    precision[2] <- meanPerf['precision']
    recall[2]<- meanPerf['recall']
    auc[2]<- meanPerf['AUC']
  }

  results <- gatherFromList(evalCoreWOCorelated)
  meanPerf <- sapply(results, mean)
  if(meanPerf['accuracy'] > acc[3]) {
    acc[3] <- meanPerf['accuracy']
    fMes[3] <- meanPerf['Fmeasure']
    precision[3] <- meanPerf['precision']
    recall[3]<- meanPerf['recall']
    auc[3]<- meanPerf['AUC']
  }
  
  results <- gatherFromList(evalCoreSheared)
  meanPerf <- sapply(results, mean)
  if( meanPerf['accuracy'] > acc[4] ) {
    acc[4] <- meanPerf['accuracy']
    fMes[4] <- meanPerf['Fmeasure']
    precision[4] <- meanPerf['precision']
    recall[4]<- meanPerf['recall']
    auc[4]<- meanPerf['AUC']
  }

  }

  print(acc)
  print(fMes)
  print(precision)
  print(recall)
  print(auc)
# }

# kNN, k = 5
acc <- c(0,0,0,0)
fMes <- c(0,0,0,0)
precision <- c(0,0,0,0)
recall <- c(0,0,0,0)
auc <- c(0,0,0,0)
evalCore<-list()
evalCoreWOCorelated <- list()
evalCoreWOUnimportant <- list()
evalCoreSheared <- list()
  for (i in 1:10){

    foldIdx <- cvGen(nrow(train), k=folds)
    for (j in 1:folds) {
      dTrain <- train[foldIdx!=j,]
      dTest  <- train[foldIdx==j,]
      
      modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 5) 
      predCore <- predict(modelCore, dTest)
      
      evalCore[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                                predictedClass=predCore$class, predictedProb=predCore$prob )

      ##
      dTrain <- trainWOCorelated[foldIdx!=j,]
      dTest  <- trainWOCorelated[foldIdx==j,]
      
      modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 5) 
      predCore <- predict(modelCore, dTest)
      
      evalCoreWOCorelated[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                                predictedClass=predCore$class, predictedProb=predCore$prob )


      ##
      dTrain <- trainWOUnimportant[foldIdx!=j,]
      dTest  <- trainWOUnimportant[foldIdx==j,]
      
      modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 5) 
      predCore <- predict(modelCore, dTest)
      
      evalCoreWOUnimportant[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                                predictedClass=predCore$class, predictedProb=predCore$prob )

      ##
      dTrain <- trainSheared[foldIdx!=j,]
      dTest  <- trainSheared[foldIdx==j,]
      
      modelCore <- CoreModel(Class~., dTrain, model="knn", kInNN = 5) 
      predCore <- predict(modelCore, dTest)
      
      evalCoreSheared[[j]] <- modelEval(modelCore, correctClass=dTest$Class,
                                predictedClass=predCore$class, predictedProb=predCore$prob )
      
      destroyModels(modelCore)
    }
  
  
  results <- gatherFromList(evalCore)
  meanPerf <- sapply(results, mean)
  if (meanPerf['accuracy'] > acc[1] ) {
    acc[1] <- meanPerf['accuracy']
    fMes[1] <- meanPerf['Fmeasure']
    precision[1] <- meanPerf['precision']
    recall[1]<- meanPerf['recall']
    auc[1]<- meanPerf['AUC']
  }


  results <- gatherFromList(evalCoreWOUnimportant)
  meanPerf <- sapply(results, mean)
  if(meanPerf['accuracy'] > acc[2]) {
    acc[2] <- meanPerf['accuracy']
    fMes[2] <- meanPerf['Fmeasure']
    precision[2] <- meanPerf['precision']
    recall[2]<- meanPerf['recall']
    auc[2]<- meanPerf['AUC']
  }

  results <- gatherFromList(evalCoreWOCorelated)
  meanPerf <- sapply(results, mean)
  if(meanPerf['accuracy'] > acc[3]) {
    acc[3] <- meanPerf['accuracy']
    fMes[3] <- meanPerf['Fmeasure']
    precision[3] <- meanPerf['precision']
    recall[3]<- meanPerf['recall']
    auc[3]<- meanPerf['AUC']
  }
  
  results <- gatherFromList(evalCoreSheared)
  meanPerf <- sapply(results, mean)
  if( meanPerf['accuracy'] > acc[4] ) {
    acc[4] <- meanPerf['accuracy']
    fMes[4] <- meanPerf['Fmeasure']
    precision[4] <- meanPerf['precision']
    recall[4]<- meanPerf['recall']
    auc[4]<- meanPerf['AUC']
  }

  }

  print(acc)
  print(fMes)
  print(precision)
  print(recall)
  print(auc)

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

# best model was Random Forrest on trainWOCorrelated training set

# lets see the actual performance of it on the test set

model <- CoreModel(Class ~ ., data = trainWOCorelated, model="rf")
predicted <- predict(model, test)
valCore <- modelEval(model, correctClass=test$Class,
                               predictedClass=predicted$class, predictedProb=predicted$prob )

results <- gatherFromList(evalCore)
meanPerfRF <- sapply(results, mean)

accRndForest <- meanPerfRF['accuracy']
fMeasRndForest <-  meanPerfRF['Fmeasure']
precRndForest <- meanPerfRF['precision']
recRndForest <-  meanPerfRF['recall']
aucRndForest <- meanPerfRF['AUC']

names <- c( "Accuracy", "fMeasure", "Precision", "Recall", "AUC")
data <- c( accRndForest, fMeasRndForest, precRndForest, recRndForest, aucRndForest) *100
df <- data.frame(names, data)
d$names <- factor(d$names, levels = d$names)

{
  p <- ggplot(df, aes(x=names, y=data)) + geom_bar(stat="identity",fill= c("#197ed6", "#ae71d6", "#ff5ea8", "#ff7261", "#ffa600"),  color= c("#197ed6", "#ae71d6", "#ff5ea8", "#ff7261", "#ffa600")) 
  p <- p  + xlab("Types of feature selections") + coord_cartesian(ylim=c(76, 100)) + ylab("%") + ggtitle("Random Forrest statistic on Train dataset")
  ggsave("RandomForrestFromCV.png", plot=p, dpi= 1200)
}

