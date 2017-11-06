
dat_train <- read.table("wafer_TRAIN",header = FALSE,sep = ',')
dat_test <- read.table("wafer_TEST",header = FALSE,sep=',')
#dtw with knn
train <- dat_train
test <- dat_test
test_label <- test[,1]

knn_euc <- function(train,test,k,w){
        pred_label <- c()
        cat("正在計算Euclidean的Distance Matrix","\n")
        dist_matrix <- dist(train[,-1],test[,-1],window.size = 0,method = "euclidean")
        if(k>length(train[,1])) return(cat("Warning:k can not be longer than n-1"))
        for(i in 1:length(test[,1])){
                nearest <- sort(dist_matrix[,i])[1:k]
                nearest_index <- which(dist_matrix[,i]%in%nearest)
                freq_table <- as.matrix(table(train[nearest_index,1]))
                label_index <- which(freq_table%in%max(freq_table))
                pred_label[i] <- row.names(freq_table)[label_index]
                cat("第",i,"個序列","分類為",pred_label[i],"原始為",test[i,1],"\n")
        }       
        return(pred_label)
}

result <- knn_euc(train,test,1,0)
error <- length(which(result!=test_label))/length(test_label)
error
#0.004542505
