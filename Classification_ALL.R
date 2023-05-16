################################################# CLASSIFIFCATION ##########################################################
######################################KNN##############################################################################

# Load data
load("~/Downloads/class_data.RData")
df <- data.frame(x=x, y=y)

library(class)
set.seed(35)

# Split the data into training and testing sets (70% for training, 30% for testing)
#train_index <- createDataPartition(df$y, p = 0.7, list = FALSE)
train <- df[,]
test <- data.frame(x=xnew)

# Define hyperparameter grid
hyper_grid <- expand.grid(k = 5:15)

# Define train control for cross-validation
folds <- seq(5,30, by = 1)
train_control_list <- list()
for (i in 1:length(folds)) {
  train_control_list[[i]] <- trainControl(method = "cv", number = folds[i], allowParallel = FALSE)
}

# Train the KNN model using cross-validation and hyperparameter tuning for each value of folds
knn.pred_list <- list()
best_k <- 0
best_acc <- 0
best_fold <- 0
for (i in 1:length(train_control_list)) {
  knn.pred_list[[i]] <- train(train[,-ncol(train)], as.factor(train$y), method = "knn", trControl = train_control_list[[i]], tuneGrid = hyper_grid)
  if (max(knn.pred_list[[i]]$results$Accuracy) > best_acc) {
    best_acc <- max(knn.pred_list[[i]]$results$Accuracy)
    best_k <- knn.pred_list[[i]]$bestTune$k
    best_fold <- i+4
  }
}

# Print the best fold and k combination based on the lowest misclassification error rate
cat("\nBest Fold-K Combination (kNN):", best_fold, "folds and k =", best_k, "\n")
print(knn.pred_list[[best_fold-4]])


y_new <- predict(knn.pred_list[[best_fold-4]], newdata = test[,],k=best_k)

test_error <- (1 - best_acc)*100
save(y_new,test_error,file="Group19.RData")


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

################################## Random Forest ######################################################################
library(xgboost)
library(caret)

load("~/Downloads/class_data.RData")
df <- data.frame(x=x, y=y)
set.seed(515)

# Convert y to a factor with two levels
df$y <- as.factor(df$y)
levels(df$y) <- c(0, 1)  # Specify levels as 0 and 1

# Create a vector of indices for 10-fold cross-validation
folds <- createFolds(y = y, k = 10, returnTrain = FALSE)

# Create an empty vector to store the cross-validated accuracy
cv_accuracy <- numeric(length = 10)
models = vector(mode='list', length=10)
feature_list = vector(mode='list', length=10)

# Perform 10-fold cross-validation
for (i in 1:10) {
  # Split the data into training and testing sets for the current fold
  train_indices <- unlist(folds[-i])
  test_indices <- folds[[i]]
  train_x <- x[train_indices, ]
  train_y <- y[train_indices]
  test_x <- x[test_indices, ]
  test_y <- y[test_indices]
  
  # Perform feature selection using rfe
  control <- rfeControl(functions = rfFuncs,
                        method = "cv",
                        number = 10)
  
  result_rfe <- rfe(x = train_x, 
                    y = as.factor(train_y), 
                    sizes = 193,
                    rfeControl = control,
                    metric = "Accuracy")
  
  # Get the selected features from rfe
  selected_features <- train_x[, result_rfe$optVariables]
  feature_list[[i]] <- result_rfe$optVariables
  
  train_control_list <- trainControl(method = "cv", number = 10, allowParallel = FALSE)
  hyper_grid <- expand.grid(mtry = 23)
  
  models[[i]] <- train(selected_features, as.factor(train_y), method = "rf", trControl = train_control_list, tuneGrid = hyper_grid)
  
  
  # Make predictions on the test set using the selected features
  pred_y <- predict(models[[i]], newdata = test_x[, result_rfe$optVariables])
  
  # Calculate accuracy for the current fold
  accuracy <- sum(pred_y == test_y) / length(test_y)
  
  # Store the accuracy in the vector
  cv_accuracy[i] <- accuracy
  
  # Print the accuracy for the current fold
  cat("Fold", i, "Accuracy:", round(accuracy, 4), "\n")
}

# Calculate and print the average cross-validated accuracy
avg_accuracy <- mean(cv_accuracy)
cat("Average Accuracy:", round(avg_accuracy, 4), "\n")

best_fold <- which.max(cv_accuracy)
y_new <- predict(models[[best_fold]], newdata = xnew[, feature_list[[best_fold]]])
########################################### Boosting ##################################################################
# Load required libraries
library(xgboost)
library(caret)

# Load data
load("~/Downloads/class_data.RData")
df <- data.frame(x=x, y=y)

# Convert y to a factor with two levels
df$y <- as.factor(df$y)
levels(df$y) <- c(0, 1)  # Specify levels as 0 and 1

# Define tuning parameter grid
# tune_grid <- expand.grid(
#   nrounds = c(50, 100, 150),
#   max_depth = c(3, 6, 9),
#   eta = c(0.1, 0.01, 0.001),
#   gamma = c(0, 0.1, 0.2, 0.5),
#   colsample_bytree = c(0.8, 1),
#   min_child_weight = c(1, 3, 5),
#   subsample = c(0.8, 1)
# )

tune_grid <- expand.grid(
  nrounds = c(100),
  max_depth = c(9),
  eta = c(0.001),
  gamma = c(0.5),
  colsample_bytree = c(0.8),
  min_child_weight = c(1),
  subsample = c(1)
)

# Train and tune the boosting model with cross-validation
set.seed(124)
boosting_model <- train(
  x = as.matrix(x),
  y = as.factor(y),
  method = "xgbTree",
  tuneGrid = tune_grid,
  trControl = trainControl(method = "cv", number = 10),
  verbose = FALSE
)

y_new_prob <- predict(boosting_model, newdata = as.matrix(x = xnew))

y_new_prob <- as.numeric(as.character(y_new_prob))
y_new_labels <- ifelse(y_new_prob > 0.5, 1, 0)

################################################ Ensemble ###################################################
# Load required libraries
library(xgboost)
library(caret)
library(randomForest)
library(glmnet)

# Load data
load("~/Downloads/class_data.RData")
df <- data.frame(x=x, y=y)
set.seed(336)

# Convert y to a factor with two levels
df$y <- as.factor(df$y)
levels(df$y) <- c(0, 1)  # Specify levels as 0 and 1

# Create a vector of indices for 10-fold cross-validation
folds <- createFolds(y = y, k = 10, returnTrain = FALSE)

# Create an empty vector to store the cross-validated accuracy
cv_accuracy <- numeric(length = 10)
models <- vector(mode='list', length=10)
feature_list <- vector(mode='list', length=10)

# Perform 10-fold cross-validation
for (i in 1:10) {
  # Split the data into training and testing sets for the current fold
  train_indices <- unlist(folds[-i])
  test_indices <- folds[[i]]
  train_x <- x[train_indices, ]
  train_y <- y[train_indices]
  test_x <- x[test_indices, ]
  test_y <- y[test_indices]
  
  # Perform feature selection using rfe
  control <- rfeControl(functions = rfFuncs,
                        method = "cv",
                        number = 10)
  
  result_rfe <- rfe(x = train_x,
                    y = as.factor(train_y),
                    sizes = 250:255,
                    rfeControl = control,
                    metric = "Accuracy")
  
  # # Get the selected features from rfe
  selected_features <- train_x[, result_rfe$optVariables]
  feature_list[[i]] <- result_rfe$optVariables
  
  model_knn <- train(as.data.frame(selected_features),
                     as.factor(train_y),
                     method = "knn",
                     trControl = trainControl(method = "cv", number = 10, allowParallel = FALSE),
                     tuneGrid = expand.grid(k = 8))
  
  # Train random forest model
  tune_grid <- expand.grid(mtry = 17:19)
  model_rf <- train(as.data.frame(selected_features),
                    as.factor(train_y),
                    method = "rf",
                    trControl = trainControl(method = "cv", number = 10, allowParallel = FALSE),
                    tuneGrid = tune_grid)
  
  # Train xgboost model
  train_control_list <- trainControl(method = "cv", number = 10, allowParallel = FALSE)
  hyper_grid <- expand.grid(
    nrounds = c(100),
    max_depth = c(6),
    eta = c(0.001),
    gamma = c(0.2),
    colsample_bytree = c(0.8),
    min_child_weight = c(1),
    subsample = c(1)
  )
  model_xgb <- train(as.data.frame(selected_features),#selected_features, 
                     as.factor(train_y), 
                     method = "xgbTree", 
                     trControl = train_control_list, 
                     tuneGrid = hyper_grid)
  
  # Retrain the random forest model
  rf_model <- randomForest(x=as.data.frame(selected_features), y=as.factor(train_y), 
                           mtry = model_rf$bestTune$mtry)
  
  # Make predictions on the test set using the selected features
  #pred_logreg <- as.numeric(as.character(predict(model_logreg, newdata = as.data.frame(test_x))))
  pred_knn <- as.numeric(as.character(predict(model_knn, newdata = test_x[, result_rfe$optVariables])))
  pred_rf <- as.numeric(as.character(predict(rf_model, newdata = test_x[, result_rfe$optVariables])))
  pred_xgb <- as.numeric(as.character(predict(model_xgb, newdata = test_x[, result_rfe$optVariables])))
  
  # Create a new data frame to store the predictions
  df_pred <- data.frame(knn = pred_knn, rf = pred_rf, xgb = pred_xgb)
  
  # Majority voting for ensemble prediction
  df_pred$ensemble <- ifelse(rowSums(df_pred == "1") > 1, 1, 0)
  
  # Calculate accuracy for current fold
  cv_accuracy[i] <- sum(df_pred$ensemble == test_y) / length(test_y)
  
  # Print the accuracy for the current fold
  cat("Fold", i, "Accuracy:", round(cv_accuracy[i], 4), "\n")
  
  # Store the models for each fold
  models[[i]] <- list(knn = model_knn, rf = rf_model, xgb = model_xgb)
}

# Calculate average cross-validated accuracy
mean_cv_accuracy <- mean(cv_accuracy)
# Print results
cat("Average Cross-Validated Accuracy:", mean_cv_accuracy, "\n")

# Function to make ensemble predictions on new data
make_ensemble_predictions <- function(xnew, model, f_list) {
  # Extract selected features from new data
  selected_features <- xnew[, f_list]
  
  # Make predictions using each model
  #pred_logreg <- as.numeric(as.character(predict(model$logreg, newdata = as.data.frame(xnew))))
  pred_knn <- as.numeric(as.character(predict(model$knn, newdata = as.data.frame(selected_features))))
  pred_rf <- as.numeric(as.character(predict(model$rf, newdata = as.data.frame(selected_features))))
  pred_xgb <- as.numeric(as.character(predict(model$xgb, newdata = as.data.frame(selected_features))))
  
  
  # Create a new data frame to store the predictions
  df_pred <- data.frame(knn = pred_knn, rf = pred_rf, xgb = pred_xgb)
  
  # Majority voting for ensemble prediction
  df_pred$ensemble <- ifelse(rowSums(df_pred == "1") > 1, 1, 0)
  
  # Return the ensemble predictions
  return(df_pred$ensemble)
}

# Example usage of the ensemble model on new data
best_fold <- which.max(cv_accuracy)
y_new <- make_ensemble_predictions(xnew, models[[best_fold]], feature_list[[best_fold]])
print(y_new)
####################################################### LDA ############################################################
# Load data
load("~/Downloads/class_data.RData")
df <- data.frame(x=x, y=y)

library(MASS)
set.seed(42)

# Split the data into training and testing sets (70% for training, 30% for testing)
#train_index <- createDataPartition(df$y, p = 0.7, list = FALSE)
train <- df[,]
test <- data.frame(x=xnew)

# Define train control for cross-validation
folds <- seq(5, 30, by = 1)
train_control_list <- lapply(folds, function(f) trainControl(method = "cv", number = f, allowParallel = FALSE))

# Train the LDA model using cross-validation for each value of folds
lda.pred_list <- lapply(train_control_list, function(control) train(train[,-ncol(train)], as.factor(train$y), method = "lda", trControl = control))

# Find the best model based on the highest accuracy
best_acc <- max(sapply(lda.pred_list, function(pred) max(pred$results$Accuracy)))
best_fold <- which.max(sapply(lda.pred_list, function(pred) max(pred$results$Accuracy)))
best_lda <- lda.pred_list[[best_fold]]

# Print the best fold and accuracy
cat("\nBest Fold-Accuracy Combination (LDA):", best_fold, "folds and accuracy =", best_acc, "\n")

# Print the best LDA model and its results
print(best_lda)
y_new <- predict(best_lda, newdata = test)

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