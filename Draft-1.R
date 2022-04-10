#### Pre-processing
# Read in files 
users <- read.csv("D:/users.csv")
likes <- read.csv("D:/likes.csv")
ul <- read.csv("D:/users-likes.csv")
# Match entries in ul with users and likes dictionaries
ul$user_row<-match(ul$userid,users$userid)
ul$like_row<-match(ul$likeid,likes$likeid)
# and inspect what happened: 
head(ul)

install.packages("Matrix")
require(Matrix)

# Construct the sparse User-Like Matrix M
M <- sparseMatrix(i = ul$user_row, j = ul$like_row, x = 1)
# Check the dimensions of M
dim(M)
# Save user IDs as row names in M
rownames(M) <- users$userid    
# Save Like names as column names in M
colnames(M) <- likes$name    
# Remove ul and likes objects (they won't be needed)
rm(ul, likes)

# Remove users/Likes occurring less than 50/150 times 
repeat {                                       
  i <- sum(dim(M))                             
  M <- M[rowSums(M) >= 50, colSums(M) >= 150]  
  if (sum(dim(M)) == i) break                  
  }
  # Check the new size of M
dim(M)
# Remove the users from users object that were removed from M
users <- users[match(rownames(M), users$userid), ]
# Check the new size of users
dim(users)

### Dimensionality Reduction
## SVD:

# Preset the random number generator in R for the comparability of the results
set.seed(seed = 68)
# Install irlba package (run only once)- used for both svd and pca
install.packages("irlba")
library(irlba)
Msvd <- irlba(M, nv = 5)
# User SVD scores are here:
u <- Msvd$u
# Like SVD scores are here:
v <- Msvd$v
# The scree plot of singular values:
plot(Msvd$d)

#  varimax-rotate the resulting SVD space:
# First obtain rotated V matrix: (unclass function has to be used to save it as an object of type matrix and not loadings)
v_rot <- unclass(varimax(Msvd$v)$loadings)
# The cross-product of M and v_rot gives u_rot:
u_rot <- as.matrix(M %*% v_rot)

## LDA:

# Install topicmodels package (run only once)
install.packages("topicmodels")
library(topicmodels)
# Conduct LDA analysis, see text for details on setting alpha and delta parameters. 
# WARNING: this may take quite some time!
Mlda <- LDA(M, control = list(alpha = 10, delta = .1, seed=68), k = 5, method = "Gibbs")
# Extract user LDA cluster memberships
gamma <- Mlda@gamma
# Extract Like LDA clusters memberships
# betas are stored as logarithms, 
# function exp() is used to convert logs to probabilities
beta <- exp(Mlda@beta)
# Log-likelihood of the model is stored here:
Mlda@loglikelihood
# and can be also accessed using logLik() function:
logLik(Mlda)
# Let us estimate the log-likelihood for 2,3,4, and 5 cluster solutions: 
lg <- list()
for (i in 2:5) {
Mlda <- LDA(M, k = i, control = list(alpha = 10, delta = .1, seed = 68), method = "Gibbs")
lg[[i]] <- logLik(Mlda) 
    }
plot(2:5, unlist(lg))

## PCA:

Mpca <- prcomp_irlba(M, n=5, retx = TRUE, center = TRUE, scale.=TRUE)
print(Mpca$x)

### Clustering:
## K-means:
install.packages("factoextra")
library(factoextra)

#SVD to K-means:
set.seed(123)
km.res_svd <- kmeans(Msvd$u, 5, iter.max = 10, nstart = 3)

print(km.res_svd)

#LDA to K-means
set.seed(123)
km.res_lda <- kmeans(Mlda@gamma, 5, iter.max = 10, nstart = 3)

print(km.res_lda)

#PCA to K-means:
set.seed(123)
km.res_pca <- kmeans(Mpca$x, 5, iter.max = 10, nstart = 3)

print(km.res_pca)

## DB Scan:
install.packages("fpc")
library(fpc)

#SVD to DB Scan:
set.seed(123)
db_svd <- fpc::dbscan(Msvd$u, eps = 15, MinPts = 5)

print(db_svd)

#LDA to DB Scan:
set.seed(123)
db_lda <- fpc::dbscan(Mlda@gamma, eps = 15, MinPts = 5)

print(db_lda)

#PCA to DB Scan:
set.seed(123)
db_pca <- fpc::dbscan(Mpca$x, eps = 15, MinPts = 5)

print(db_pca)

## Fuzzy C means:
install.packages("ppclust")
library(ppclust)

#SVD to FCM:
fcm_svd <- fcm(Msvd$u, centers = 5)
summary(fcm_svd)

#LDA to FCM:
fcm_lda <- fcm(Mlda@gamma, centers = 5)
summary(fcm_lda)

#PCA to FCM:
fcm_pca <- fcm(Mpca$x, centers = 5)
summary(fcm_pca)

## Interpreting clusters and dimensions:
# Correlate user traits and their SVD scores
# users[,-1] is used to exclude the column with IDs
cor(u_rot, users[,-1], use = "pairwise")
# LDA version
cor(gamma, users[,-1], use = "pairwise")

## plotting eat map
# You need to install ggplot2 and reshape2 packages first, run only once:
install.packages("ggplot2", "reshape2")
library(ggplot2)
library(reshape2)
    
# Get correlations
x<-round(cor(u_rot, users[,-1], use="p"),2)  
# Reshape it in an easy way using ggplot2
y<-melt(x)
colnames(y)<-c("SVD", "Trait", "r")    
# Produce the plot
qplot(x=SVD, y=Trait, data=y, fill=r, geom="tile") +
  scale_fill_gradient2(limits=range(x), breaks=c(min(x), 0, max(x)))+
  theme(axis.text=element_text(size=12), 
        axis.title=element_text(size=14,face="bold"),
        panel.background = element_rect(fill='white', colour='white'))+
  labs(x=expression('SVD'[rot]), y=NULL)

## print the Likes with the highest and lowest varimax-rotated SVD and LDA scores:

#SVD:
top <- list()
bottom <-list()
for (i in 1:5) {
f <- order(v_rot[ ,i])
temp <- tail(f, n = 10)
top[[i]]<-colnames(M)[temp]  
temp <- head(f, n = 10)
bottom[[i]]<-colnames(M)[temp]  
}

# LDA:
top <- list()
for (i in 1:5) {
f <- order(beta[i,])
temp <- tail(f, n = 10)
top[[i]]<-colnames(M)[temp]  
} 
