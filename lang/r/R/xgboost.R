source("../import_lib.R")
import.pkgs('plyr', 'glmnet', 'caret', 'ggplot2', 'prophet', 'xgboost')

## load data
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test

## fit model
bst <- xgboost(data = train$data,
              label = train$label,
              max.depth = 2,
              eta = 1,
              nround = 2,
              nthread = 2,
              objective = "binary:logistic")

## predict
pred <- predict(bst, test$data)

importance <- xgb.importance(feature_names=colnames(train$data), model = bst)
