
dat <- readLines("wafer-im8-n-train-75.txt")
dat
#with label
dat_list <- strsplit(dat,"\\s+")
for(i in 1:length(dat_list)){
dat_list[[i]] <- dat_list[[i]][-1]
dat_list[[i]] <- round(as.numeric(dat_list[[i]]),6)
}
#no label
dat_list_value <- strsplit(dat,"\\s+")
for(i in 1:length(dat_list_value)){
        dat_list_value[[i]] <- dat_list_value[[i]][c(-1,-2)]
        dat_list_value[[i]] <- round(as.numeric(dat_list_value[[i]]),6)
}
dat_list_value[[1]]


#find match
library(zoo)
shapelets[[1]]
query <- round(as.numeric(shapelets[[1]]),6)
#debug
query[38] = dat_list_value[[307]][38:127][38]
find_match <- sapply(dat_list_value, function(y){
        any(rollapply(y,length(query), function(v) all(v == query)))
})
match_index <- which(find_match)
match_index

#get match shapelet and sensor index
find_sensor_index <- function(data,sensor_size){
if(match_index %% sensor_size>=1)
sensor_index <- match_index %% sensor_size else sensor_index <- 6
sensor_index <- seq(sensor_index,length(data),by=sensor_size)
return(sensor_index)
}
find_sensor_index(dat_list_value,6)
sensor_index <- find_sensor_index(dat_list_value,6)

#sliding_window
sliding_window <- function(ts,w_size){
        window <- list()
        for(i in 1:c(length(ts)-w_size+1)){
                window_value <- ts[i:c(i-1+w_size)]
                window[[i]] <- window_value
        }
        return(window)
}
s_window <- sliding_window(dat_list_value[[1]],length(query))
s_window
length(s_window)
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

#find min distance with sliding window in one sensor
distance_sensor <- function(sensor_index,shapelet_value){
        dist <- vector()
        for(i in sensor_index){
                query <- query
                s_window <- sliding_window(dat_list_value[[i]],length(query))   
                dist[i] <- min_dist_euc(s_window,query)        
        }
        dist <- as.numeric(na.omit(dist))
        return(dist)
}
distance_sensor(sensor_index,query)

sensor1_dist <- distance_sensor(sensor_index,query)
sensor1_dist
