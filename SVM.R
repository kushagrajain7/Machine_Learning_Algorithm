############################### SVM ########################################################################
# Load data
load("~/Downloads/class_data.RData")
df <- data.frame(x=x, y=y)

library(e1071)
set.seed(3)

# Split the data into training and testing sets (70% for training, 30% for testing)
#train_index <- createDataPartition(df$y, p = 0.7, list = FALSE)
train <- df[,]
test <- data.frame(x=xnew)

# Define hyperparameter grid
hyper_grid <- expand.grid(C = c(0.01, 0.1, 1, 10, 100), sigma = c(0.1, 1, 10))

# Define train control for cross-validation
folds <- seq(5, 10, by = 5)
train_control_list <- list()
for (i in 1:length(folds)) {
  train_control_list[[i]] <- trainControl(method = "cv", number = folds[i], allowParallel = FALSE)
}

# Train the SVM model using cross-validation and hyperparameter tuning for each value of folds
svm.pred_list <- list()
best_C <- 0
best_sigma <- 0
best_acc <- 0
best_fold <- 0
for (i in 1:length(train_control_list)) {
  svm.pred_list[[i]] <- train(train[,-ncol(train)], as.factor(train$y), method = "svmRadial", trControl = train_control_list[[i]], tuneGrid = hyper_grid, metric = "Accuracy")
  if (max(svm.pred_list[[i]]$results$Accuracy) > best_acc) {
    best_acc <- max(svm.pred_list[[i]]$results$Accuracy)
    best_C <- svm.pred_list[[i]]$bestTune$C
    best_sigma <- svm.pred_list[[i]]$bestTune$sigma
    best_fold <- i+4
  }
}

# Print the best fold and hyperparameters combination based on the highest accuracy
cat("\nBest Fold-Hyperparameters Combination (SVM):", best_fold, "folds, C =", best_C, "and sigma =", best_sigma, "\n")

print(svm.pred_list[[best_fold-4]])
y_new <- predict(svm.pred_list[[best_fold-4]], newdata = test[,])