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

# The majority classifier
table(observed)
sum(observed=='1')/length(observed) # 0.645933

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
