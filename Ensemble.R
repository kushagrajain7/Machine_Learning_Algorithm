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