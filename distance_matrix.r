setwd("F:/TSA/")

t <- c(sample(10))
length(t)
query <- c(1:5)
#sliding_window

sliding_window <- function(ts,w_size){
    window <- list()
    for(i in 1:c(length(ts)-w_size+1)){
        window_value <- ts[i:c(i-1+w_size)]
        window[[i]] <- window_value
    }
    return(window)
}

s_window <- sliding_window(t,length(query))
s_window
s_window[[1]]


#find min dist with dtw
library(dtw)
min_dist_dtw <- function(time_series_via_window,shapelet){
    alldist <- list()
    for(i in 1:length(time_series_via_window)){
        each_dtw_dist <- dtw(time_series_via_window[[i]],shapelet,dist.method = "euclidean")
        alldist[i] <- each_dtw_dist$distance
    }
    return(min(as.numeric(alldist)))
}
min_dist_dtw(s_window,query)

#normalize length euclidean
euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2)/length(x2))
min_dist_euc <- function(time_series_via_window,shapelet){
    alldist <- list()
    for(i in 1:length(s_window)){
        each_euc_dist <- euc.dist(s_window[[i]],shapelet)
        alldist[i] <- each_euc_dist
    }
    return(min(as.numeric(alldist)))    
}

min_dist_euc(s_window,query)
