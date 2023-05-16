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