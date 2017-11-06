#set your path
setwd("f:/TSA/")

#new method
tree <- readLines(tree)
shapelets <- strsplit(tree,"\\s+")
shapelets <- shapelets[c(22:25)]
for(i in 1 :length(shapelets)){
        shapelets[[i]] <- shapelets[[i]][c(-1,-2)]
}
shapelets
plot(shapelets[[1]])

######
tree <- file('Source Code/Executable/tree.txt')
tree <- readLines(tree)
tree
shapelets <- tree[26:31]
shapelets
x <- data.frame(shapelets,stringsAsFactors=FALSE)
s1 <- strsplit(x[1,1],split = " ",fixed = T)
s1 <- as.numeric(unlist(s1))
s1 <- na.omit(s1)
s1 <- s1[-1]
s2 <- strsplit(x[2,1],split = " ",fixed = T)
s2 <- as.numeric(unlist(s2))
s2 <- na.omit(s2)
s2 <- s2[-1]
s3 <- strsplit(x[3,1],split = " ",fixed = T)
s3 <- as.numeric(unlist(s3))
s3 <- na.omit(s3)
s3 <- s3[-1]
s4 <- strsplit(x[4,1],split = " ",fixed = T)
s4 <- as.numeric(unlist(s4))
s4 <- na.omit(s4)
s4 <- s4[-1]
s5 <- strsplit(x[5,1],split = " ",fixed = T)
s5 <- as.numeric(unlist(s5))
s5 <- na.omit(s5)
s5 <- s5[-1]
s6 <- strsplit(x[6,1],split = " ",fixed = T)
s6 <- as.numeric(unlist(s6))
s6 <- na.omit(s6)
s6 <- s6[-1]

shapelets <- list(s1,s2,s3,s4,s5,s6)
str(shapelets)

plot(shapelets[[2]],type='l')
