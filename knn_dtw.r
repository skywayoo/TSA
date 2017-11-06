library(dtw) 
library(dtwclust)

dat_train <- read.table("wafer_TRAIN",header = FALSE,sep = ',')
dat_test <- read.table("wafer_TEST",header = FALSE,sep=',')
#dtw with knn
train <- dat_train
test <- dat_test
test_label <- test[,1]

knn_dtw <- function(train,test,k,w){
  pred_label <- vector()
  dist_matrix <- dist(train[,-1],test[,-1],window.size=w,method = "LB_Keogh")
  if(k>length(train[,1])) return(cat("Warning:k can not be longer than n-1"))
  for(i in 1:length(test[,1])){
    nearest <- sort(dist_matrix[,i])[1:k]
    nearest_index <- which(dist_matrix[,i]%in%nearest)
    freq_table <- as.matrix(table(train[nearest_index,1]))
    label_index <- which(freq_table%in%max(freq_table))
    pred_label[i] <- row.names(freq_table)[label_index]
  }       
  return(pred_label)
}

#
result <- knn_dtw(train,test,1,10)
error <- length(which(result!=test_label))/length(test_label)
error

