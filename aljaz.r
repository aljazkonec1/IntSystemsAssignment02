library(ggplot2)
library(dplyr)
library(CORElearn)


train <- read.table(file="train.csv", sep=",", header= TRUE)
test <- read.table(file="test.csv", sep=",", header= TRUE)
summary(train)
    
# odstanimo stolpce V2, 4, 9, 17, 19, 21, 24, 26, 28, 29, 32, 35, 40
reduced_train <- train[, c(1, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 25, 27, 30, 31, 33, 34, 36, 37, 38, 39, 41, 42)]


CA <- function(observed, predicted){
  t <- table(observed, predicted)
  
  sum(diag(t)) / sum(t)
}

observed <- test$Class
procenti <- 0
subset_procenti <- 0

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




