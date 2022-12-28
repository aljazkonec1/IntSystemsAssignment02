library(ggplot2)
library(dplyr)
install.packages("scatterplot3d") # Install
library("scatterplot3d") # load

train <- read.table(file="train.csv", sep=",", header= TRUE)

summary(train)

sum(train$Class == 2)/ (sum(train$Class == 1) + sum(train$Class == 2))

train[train=='NA'] <- NA
new_train <- train[!rowSums(is.na(train)) > 0, ]
new_train

plot(new_train$V36, col=new_train$Class)
na_train <- train[rowSums(is.na(train)) > 0, ]
plot(na_train[, c("V1")], col=na_train$Class)

sum(new_train$Class == 2)
sum(new_train$Class == 1)



for (i in colnames(train)) {
    
    name <- paste(i, ".png", sep="")
    print(name)
    png(filename=name)
    plot(train[, c(i)], col=train$Class)
    dev.off()

}
train

png(filename="V23.png")
plot(train[train$V23 < 100, c("V23")], col=train$Class)
dev.off()

t <- train[train$V23 < 100, ]

png(filename="boxplot_on_t.png")
boxplot(t[t$Class ==1, ])
dev.off()

n_train <- new_train
for (i in 1:41) {
    n_train[, i] <- new_train[, i] / max(new_train[, i])
}

head(n_train)
col <- c("red", "blue")
col <- col[as.numeric(n_train$Class)]


