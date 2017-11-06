library(tm)
library(stringr)
#get path and normal filename
getwd()
setwd("D:/Users/skywayoo/Documents/wafer/normal/")
path <- DirSource(getwd())
s1_path <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/normal/[0-9]+_[0-9]+.11"))
s2_path <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/normal/[0-9]+_[0-9]+.12"))
s3_path <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/normal/[0-9]+_[0-9]+.15"))
s4_path <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/normal/[0-9]+_[0-9]+.6"))
s5_path <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/normal/[0-9]+_[0-9]+.7"))
s6_path <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/normal/[0-9]+_[0-9]+.8"))

#read sensor 1:6  normal data
#1
s1_dat <- list()
for(i in 1:length(s1_path)){
        value<- read.table(s1_path[i])
        s1_dat[i]<-value[2]
}
plot(unlist(s1_dat[1]),type='l')
plot(unlist(s1_dat[2]),type='l')
#2
s2_dat <- list()
for(i in 1:length(s2_path)){
        value<- read.table(s2_path[i])
        s2_dat[i]<-value[2]
}
plot(unlist(s2_dat[1]),type='l')
plot(unlist(s2_dat[2]),type='l')
#3
s3_dat <- list()
for(i in 1:length(s3_path)){
        value<- read.table(s3_path[i])
        s3_dat[i]<-value[2]
}
plot(unlist(s3_dat[1]),type='l')
plot(unlist(s3_dat[2]),type='l')
#4
s4_dat <- list()
for(i in 1:length(s4_path)){
        value<- read.table(s4_path[i])
        s4_dat[i]<-value[2]
}
plot(unlist(s4_dat[1]),type='l')
plot(unlist(s4_dat[2]),type='l')
#5
s5_dat <- list()
for(i in 1:length(s5_path)){
        value<- read.table(s5_path[i])
        s5_dat[i]<-value[2]
}
plot(unlist(s5_dat[1]),type='l')
plot(unlist(s5_dat[2]),type='l')
#6
s6_dat <- list()
for(i in 1:length(s6_path)){
        value<- read.table(s6_path[i])
        s6_dat[i]<-value[2]
}
plot(unlist(s6_dat[1]),type='l')
plot(unlist(s6_dat[2]),type='l')
#read abnormal
setwd("D:/Users/skywayoo/Documents/wafer/abnormal/")
path <- DirSource(getwd())
s1_path_f <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/abnormal/[0-9]+_[0-9]+.11"))
s2_path_f <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/abnormal/[0-9]+_[0-9]+.12"))
s3_path_f <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/abnormal/[0-9]+_[0-9]+.15"))
s4_path_f <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/abnormal/[0-9]+_[0-9]+.6"))
s5_path_f <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/abnormal/[0-9]+_[0-9]+.7"))
s6_path_f <- na.omit(str_extract(path$filelist,"D:/Users/skywayoo/Documents/wafer/abnormal/[0-9]+_[0-9]+.8"))

s1_dat_f <- list()
for(i in 1:length(s1_path_f)){
        value<- read.table(s1_path_f[i])
        s1_dat_f[i]<-value[2]
}
plot(unlist(s1_dat_f[1]),type='l')
plot(unlist(s1_dat_f[2]),type='l')
#2
s2_dat_f <- list()
for(i in 1:length(s2_path_f)){
        value<- read.table(s2_path_f[i])
        s2_dat_f[i]<-value[2]
}
plot(unlist(s2_dat_f[1]),type='l')
plot(unlist(s2_dat_f[2]),type='l')
#3
s3_dat_f <- list()
for(i in 1:length(s3_path_f)){
        value<- read.table(s3_path_f[i])
        s3_dat_f[i]<-value[2]
}
plot(unlist(s3_dat_f[1]),type='l')
plot(unlist(s3_dat_f[2]),type='l')
#4
s4_dat_f <- list()
for(i in 1:length(s4_path_f)){
        value<- read.table(s4_path_f[i])
        s4_dat_f[i]<-value[2]
}
plot(unlist(s4_dat_f[1]),type='l')
plot(unlist(s4_dat_f[2]),type='l')
#5
s5_dat_f <- list()
for(i in 1:length(s5_path_f)){
        value<- read.table(s5_path_f[i])
        s5_dat_f[i]<-value[2]
}
plot(unlist(s5_dat_f[1]),type='l')
plot(unlist(s5_dat_f[2]),type='l')
#6
s6_dat_f <- list()
for(i in 1:length(s6_path_f)){
        value<- read.table(s6_path_f[i])
        s6_dat_f[i]<-value[2]
}
plot(unlist(s6_dat_f[1]),type='l')
plot(unlist(s6_dat_f[2]),type='l')

