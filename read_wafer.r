rm(list=ls())

setwd("D:/Users/skywayoo/Documents/")
dat_train <- read.table("wafer-full-n-train-75.txt",fill = T,na.string="",col.names = paste("X", 1:199, sep = ""))
sum(dat_train$X1)
head(dat_train)


dat_test <- read.table("wafer-full-n-test-75.txt",fill = T,na.string="",col.names = paste("X", 1:153, sep = ""))
sum(dat_test$X1)
head(dat_test)
