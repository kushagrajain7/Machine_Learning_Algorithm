############################################# Naive Bayes ############################################################

# Load data
load("~/Downloads/class_data.RData")
df <- data.frame(x=x, y=y)

library(class)
library(e1071)
set.seed(41)

# Split the data into training and testing sets (70% for training, 30% for testing)
#train_index <- createDataPartition(df$y, p = 0.99, list = FALSE)
train <- df[,]
test <- data.frame(x=xnew)

# Define train control for cross-validation
folds <- seq(5,10, by = 5)
train_control_list <- list()
for (i in 1:length(folds)) {
  train_control_list[[i]] <- trainControl(method = "cv", number = folds[i], allowParallel = FALSE)
}

# Train the Naive Bayes model using cross-validation and hyperparameter tuning for each value of folds
nb.pred_list <- list()
best_sigma <- 0
best_fold <- 0
best_acc <- 0
for (i in 1:length(train_control_list)) {
  nb.pred_list[[i]] <- train(train[,-ncol(train)], as.factor(train$y), method = "nb", trControl = train_control_list[[i]])
  if (max(nb.pred_list[[i]]$results$Accuracy) > best_acc) {
    best_acc <- max(nb.pred_list[[i]]$results$Accuracy)
    best_sigma <- nb.pred_list[[i]]$bestTune$sigma
    best_fold <- i+4
  }
}

# Print the best fold and sigma combination based on the highest accuracy
cat("\nBest Fold-Sigma Combination (Naive Bayes):", best_fold, "folds and sigma =", best_sigma, "\n")

print(nb.pred_list[[best_fold-4]])
y_new <- predict(nb.pred_list[[best_fold-4]], newdata = test[,])
