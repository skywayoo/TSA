data.train <- read.csv("wafer_train.csv")
head(data.train)
data.train <- data.train[,-1]
data.train[data.train$label %in% 1,]$label <- c("normal")
data.train[data.train$label %in% 2,]$label <- c("abnormal")
data.train$label <- as.factor(data.train$label)
data.test <- read.csv("wafer_test.csv")
data.test <- data.test[,-1]
data.test[data.test$label %in% 1,]$label <- c("normal")
data.test[data.test$label %in% 2,]$label <- c("abnormal")
data.test$label <- as.factor(data.test$label)
library(rpart)
tree_model <- rpart(label~.,data.train,control = rpart.control(cp = 0))
summary(tree_model)

plot(tree_model,margin=0.2)
text(tree_model,use.n=T,cex=1)
# train data
tree_pred_train<- predict(tree_model,data.train,type="class")
tree_table.train <- table(data.train$label,tree_pred_train)
tree_table.train

tree_correct_train <- sum(diag(tree_table.train))/sum(tree_table.train)
cat("accuracy:",tree_correct_train*100,"%")

# test data
tree_pred<- predict(tree_model,data.test,type="class")
tree_table.test <- table(data.test$label,tree_pred)
tree_table.test

tree_correct <- sum(diag(tree_table.test))/sum(tree_table.test)
cat("accuracy:",tree_correct*100,"%")


library(randomForest)
rf_model <- randomForest(label~.,data=data.train,ntree=10000)
rf_model
#train data
rf_pred.train <- predict(rf_model,newdata=data.train)
rf_table.train <- table(data.train$label,rf_pred.train)
rf_table.train
rf_correct_train <- sum(diag(rf_table.train))/sum(rf_table.train)
cat("accuracy:",rf_correct_train*100,"%")

#test data
rf_pred <- predict(rf_model,newdata=data.test)
rf_table.test <- table(data.test$label,rf_pred)
rf_table.test
rf_correct <- sum(diag(rf_table.test))/sum(rf_table.test)
cat("accuracy:",rf_correct*100,"%")

library(e1071)
svm.model <- svm(label~ .,data=data.train,gamma=0.1,cost=10)
svm.pred <- predict(svm.model,data.test[,-7])
svm.pred.train <- predict(svm.model,data.train[,-7])
str(data.train)
#train
table.svm.train <- table(pred=svm.pred.train,true=data.train[,7])
correct.svm.train <- sum(diag(table.svm.train))/sum(table.svm.train)
cat("accuracy:",correct.svm.train*100,"%")

#test
table.svm.test <- table(pred=svm.pred,true=data.test[,7])
correct.svm <- sum(diag(table.svm.test))/sum(table.svm.test)
cat("accuracy:",correct.svm*100,"%")

#search beat gmma and cost
tuned <- tune.svm(label~.,data=data.train,gamma=10^(-3:-1),cost=10^(-1:1))
summary(tuned)


library(e1071)
nb_model <-naiveBayes(label ~ ., data = data.train)
nb.pred.train<-predict(nb_model,data.train,type="class")
nb.pred.test <- predict(nb_model,data.test,type="class")
#train
table.nb.train <- table(pred=nb.pred.train,true=data.train[,7])
correct.nb.train <- sum(diag(table.nb.train))/sum(table.nb.train)
cat("accuracy:",correct.nb.train*100,"%")
#test
table.nb.test <- table(pred=nb.pred.test,true=data.test[,7])
correct.nb <- sum(diag(table.nb.test))/sum(table.nb.test)
cat("accuracy:",correct.nb*100,"%")




library(nnet)
nn_model<-nnet(label ~., data = data.train,size = 100,maxit = 500)
#train
nn_pred.train <- predict(nn_model,data.train[,-7],type="class")
nntable.train <- table(nn_pred.train, data.train[,7])
correct.nn.train <- sum(diag(nntable.train))/sum(nntable.train)
cat("accuracy:",correct.nn.train*100,"%")
#test
nn_pred.test <- predict(nn_model,data.test[,-7],type="class")
nntable.test <- table(nn_pred.test, data.test[,7])
correct.nn.test <- sum(diag(nntable.test))/sum(nntable.test)
cat("accuracy:",correct.nn.test*100,"%")



cat(" decision tree","   train：",tree_correct_train*100,"   test：",tree_correct*100,"\n",
    "random forest","   train：",rf_correct_train*100,"        test：",rf_correct*100,"\n",
    "svm          ","   train：",correct.svm.train*100,"   test：",correct.svm*100,"\n",
    "naive bayes  ","   train：",correct.nb.train*100,"   test：",correct.nb*100,"\n",
    "neuralnetwork","   train：",correct.nn.train*100,"   test：",correct.nn.test*100)



#backpropagation
install.packages('neuralnet')
library(neuralnet)
#label要數字
str(data.train)

net.model <- neuralnet(label~dist_shapelet_1+dist_shapelet_2+dist_shapelet_3+dist_shapelet_4+dist_shapelet_5+dist_shapelet_6,data.train, hidden=10, threshold=0.01)

print(net.model)

plot(net.model)

results <- compute(net.model, data.train[,-7])

print(round(results$net.result))
table(data.train[,7], round(results$net.result))
accuracy <- sum(data.train[,7] == round(results$net.result))/length(data.train[,7])
sprintf("%.2f%%", accuracy * 100)
accuracy
