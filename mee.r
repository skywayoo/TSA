#get shapelet
getwd()
tree <- file('tree.txt')
tree <- readLines(tree)
shapelets <- strsplit(tree,"\\s+")
shapelets <- shapelets[c(22:25)]
for(i in 1 :length(shapelets)){
        shapelets[[i]] <- shapelets[[i]][c(-1,-2)]
        shapelets[[i]] <- as.numeric(shapelets[[i]])
}
shapelets[[3]]
str(shapelets)
plot(shapelets[[1]],type='l')
#read train data
dat <- readLines("FINAL.txt")
head(dat)
#with label
dat_list <- strsplit(dat,"\\s+")
head(dat_list)
for(i in 1:length(dat_list)){
        dat_list[[i]] <- as.numeric(sprintf("%9.6f",as.numeric(dat_list[[i]])),6)
}
dat_list[[1]]

#get label
label <- vector()
for(i in seq(1,length(dat_list),by=18)){
        label[i] <-   dat_list[[i]][1]
}
label <- as.numeric(na.omit(label))



#no label
dat_list_value <- strsplit(dat,"\\s+")
for(i in 1:length(dat_list_value)){
        dat_list_value[[i]] <- dat_list_value[[i]][c(-1)]
        dat_list_value[[i]] <- as.numeric(sprintf("%9.6f",as.numeric(dat_list_value[[i]])))
}

str(dat_list_value)


n=0.2*length(dat_list)
test.index =sample(1:length(dat_list),n)
train <- dat_list_value[-test.index]
test <- dat_list_value[test.index]
train_label <- label[-test.index]
test_label <- as.vector(na.omit(label[test.index]))
#find match
library(zoo)
#plot
length(shapelets[[3]])

plot(dat_list_value[[1500]], type = "l", col = "cornflowerblue",ylab="value", main = "Sensor_6",lwd=3)
lines(x=c(45:134),
      y=shapelets[[6]], col="red",lwd=3)
plot(dat_list_value[[4592]], type = "l", col = "cornflowerblue",ylab="value", main = "Sensor_2",lwd=3)
lines(x=c(45:124),
      y=shapelets[[3]], col="red",lwd=3)
dat_list_value[[1500]][c(45:134)]==shapelets[[6]]
#
match_index <- list()
for(i in 1:length(shapelets)){
        query <- shapelets[[i]]
        find_match <- sapply(dat_list_value, function(y){
                any(rollapply(y,length(query), function(v) all(v == query)))
        })
        match_index[[i]] <- which(find_match)
}
#get match shapelet and sensor index
find_sensor_index <- function(data,sensor_size){
        sensor_index <- list()
        for(i in 1 :length(match_index)){
                if(match_index[[i]] %% sensor_size>=1)
                        sensor_index[[i]] <- match_index[[i]] %% sensor_size else sensor_index[[i]] <- 6
                        sensor_index[[i]] <- seq(sensor_index[[i]],length(data),by=sensor_size) 
        }
        return(sensor_index)
}
find_sensor_index(dat_list_value,18)
sensor_index <- find_sensor_index(dat_list_value,18)

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
min_dist_dtw <- function(s_window,shapelet){
        alldist <- list()
        for(i in 1:length(s_window)){
                each_dtw_dist <- dtw(s_window[[i]],shapelet,dist.method = "euclidean")
                alldist[i] <- each_dtw_dist$distance
        }
        return(min(as.numeric(alldist)))
}
min_dist_dtw(s_window,query)
#normalize length euclidean
euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2)/length(x2))
min_dist_euc <- function(s_window,shapelet){
        alldist <- list()
        for(i in 1:length(s_window)){
                each_euc_dist <- euc.dist(s_window[[i]],shapelet)
                alldist[i] <- each_euc_dist
        }
        return(min(as.numeric(alldist)))    
}
min_dist_euc(s_window,query)

#find min distance with sliding window in one sensor
distance_final <- function(sensor_index,shapelet){
        dist <- vector()
        for(i in sensor_index){
                s_window <- sliding_window(dat_list_value[[i]],length(shapelet))   
                dist[i] <- min_dist_euc(s_window,shapelet)        
        }
        dist <- as.numeric(na.omit(dist))
        return(dist)
}



dist_shapelet_1 <- distance_final(sensor_index[[1]],shapelets[[1]])
dist_shapelet_2 <- distance_final(sensor_index[[2]],shapelets[[2]])
dist_shapelet_3 <- distance_final(sensor_index[[3]],shapelets[[3]])
dist_shapelet_4 <- distance_final(sensor_index[[4]],shapelets[[4]])
dist_shapelet_5 <- distance_final(sensor_index[[5]],shapelets[[5]])
dist_shapelet_6 <- distance_final(sensor_index[[6]],shapelets[[6]])

distance_martix_test <-data.frame(dist_shapelet_1,dist_shapelet_2,dist_shapelet_3,dist_shapelet_4,dist_shapelet_5,dist_shapelet_6,label)
head(distance_martix_test)
str(distance_martix_test)
