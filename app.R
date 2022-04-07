# Read in files (provide a full path, e.g., "~/Desktop/sample_dataset/users-likes.csv")
users <- read.csv("/Users/kunjpatel/Desktop/CSE573/project/sample_dataset/users.csv")
likes <- read.csv("/Users/kunjpatel/Desktop/CSE573/project/sample_dataset/likes.csv")
ul <- read.csv("/Users/kunjpatel/Desktop/CSE573/project/sample_dataset/users-likes.csv")    

# You can check what's inside each object using the following set of commands:
head(users)
head(likes)
head(ul)

tail(ul)
tail(users)
tail(likes)

dim(ul)
dim(users)
dim(likes)
    
# Match entries in ul with users and likes dictionaries
ul$user_row<-match(ul$userid,users$userid)
ul$like_row<-match(ul$likeid,likes$likeid)

# and inspect what happened: 
head(ul)

# Install Matrix library - run only once
install.packages("Matrix")
    
# Load Matrix library
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

 
repeat {                                       # Repeat whatever is in the brackets
  i <- sum(dim(M))                             # Check the size of M
  M <- M[rowSums(M) >= 50, colSums(M) >= 150]  # Retain only these rows/columns that meet the threshold
  if (sum(dim(M)) == i) break                  # If the size has not changed, break the loop
  }

# Check the new size of M
dim(M)

# Remove the users from users object that were removed
# from M
users <- users[match(rownames(M), users$userid), ]

# Check the new size of users
dim(users)

# Preset the random number generator in R 
# for the comparability of the results
set.seed(seed = 68)

# Install irlba package (run only once)
install.packages("irlba")

# Load irlba and extract 5 SVD dimensions
library(irlba)

p1 <- prcomp_irlba(M, n=5, retx = TRUE, center = TRUE, scale.=TRUE)
print(p1$x)
# summary(p1)

install.packages("factoextra")
library(factoextra)

set.seed(123)
km.res <- kmeans(p1$x, 5, iter.max = 10, nstart = 3)

print(km.res)
