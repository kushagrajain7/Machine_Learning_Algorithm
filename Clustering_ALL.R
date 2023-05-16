#################################### CLUSTERING ###########################################################

library(cluster)
library(ggplot2)
library (NbClust)
library(factoextra)
library(purrr)
set.seed(123)

load("~/Desktop/TAMU/Sem2/Stat639/cluster_data.RData")

y_scaled <- scale(y, center = TRUE, scale = TRUE)        #scale and mean center the data 

pca <- prcomp(y_scaled, center = TRUE, scale. = TRUE)
variance <- pca$sdev^2 / sum(pca$sdev^2)
cumulative_variance <- cumsum(variance)
num_dims <- which(cumulative_variance >= 0.95)[1]
y_pca <- data.frame(pca$x[, 1:num_dims])

# pca <- prcomp(y_scaled, center = TRUE, scale. = TRUE,rank.=2)
# y_pca <- as.data.frame(pca$x)


############find optimal k using the elbow method##########################################################
wcss <- numeric(length = 15)
for (i in 1:15) {
  km <- kmeans(y_pca, centers = i)
  wcss[i] <- sum(km$withinss)
}
df <- data.frame(k = 1:15, WCSS = wcss)
ggplot(df, aes(x = k, y = WCSS)) + geom_line()+ geom_point()   


############find optimal k using the nbclust function for elbow method#####################################
fviz_nbclust(y_pca, kmeans, method = "wss")            #to cross check


###########find optimal k using the silhoutte method#######################################################
library(fpc)
set.seed(13)
pam_k_best <- pamk(y_pca)
cat("number of clusters estimated by optimum average silhouette width:", pam_k_best$nc, "\n")
clusplot(pam(x=y_pca, k=pam_k_best$nc))


###########find optimal k using the nbclust function for silhoutte method##################################
fviz_nbclust(y_pca, kmeans, method = "silhouette")


##################CALINSKY CRITERION#######################################################################
library(vegan)
set.seed(123)
cal_fit2 <- cascadeKM(y_pca, 1, 10, iter = 1000)
plot(cal_fit2, sortg = TRUE, grpmts.plot = TRUE)
calinski.best2 <- as.numeric(which.max(cal_fit2$results[2,]))
cat("Calinski criterion optimal number of clusters:", calinski.best2, "\n")


##################BAYESIAN INFORMATION CRITERION FOR EXPECTATION - MAXIMIZATION##########################
library(mclust)
set.seed(123)
d_clust2 <- Mclust(as.matrix(y_pca), G=1:20)
m.best2 <- dim(d_clust2$z)[2]

cat("model-based optimal number of clusters:", m.best2, "\n")


#######################NBCLUST###################################################################
clusternum <- NbClust ((y_pca), distance="euclidean", method="complete")


##################Perform k-means clustering for evaluated value of k#############################################
set.seed(123)
final <- kmeans(y_pca,2 , nstart = 25)            #run final algo for k=5 in data with only 2 best PCA dimensions
print(final)

fviz_cluster(final, data = y_pca,geom = "point")


##################Perform PAM clustering for evaluated value of k##################################################
pam <- pam(y_pca, 2,  metric = "euclidean", stand = "FALSE")
fviz_cluster(pam, data = y_pca, geom="point" )


##################Perform Hierarchial clustering###########################################################################
dist_matrix <- dist(y_pca,method = "euclidean")
hclust_res <- hclust(dist_matrix, method = "complete")
plot(hclust_res, main = "Hierarchical Clustering Dendrogram", xlab = "", sub = "", cex = 0.6)
num_clusters <- 2
clusters <- cutree(hclust_res, k = num_clusters)
table(clusters)


##################Perform GMM clustering for evaluated value of k##################################################
library(mclust)
gmm_model <- Mclust(y_pca, G = 2)
gmm_model$class