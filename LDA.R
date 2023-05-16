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