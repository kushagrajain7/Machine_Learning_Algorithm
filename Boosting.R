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