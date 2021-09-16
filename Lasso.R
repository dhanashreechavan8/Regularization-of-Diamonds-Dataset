setwd("~/Desktop/")
getwd()
install.packages("ggcorrplot")
install.packages("caret")
install.packages("glmnet")
install.packages("rsample")
library(ggcorrplot)
library(caret)
library(glmnet)
library(rsample)

#Importing dataset
Diamonds <- read.table(file="diamonds.csv", sep=",", header=TRUE)

#Summarize data
summary(Diamonds)

#Check null values
sapply(Diamonds, function(x) sum(is.na(x)))

head(Diamonds)

#Remove index column
Diamonds$X <- NULL

#Finding co-relation
cor(Diamonds[-c(2,3,4)])

ggcorrplot(cor(Diamonds[-c(2,3,4)]), hc.order = TRUE,method = "circle")

#Encoding Categorical Variables
#Diamonds$cut=factor(Diamonds$cut,
#                    levels = c('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'),
#                    labels = c(1, 2, 3, 4, 5))

#Diamonds$color=factor(Diamonds$color,
#                      levels = c('J', 'I', 'H', 'G', 'F', 'E', 'D'),
#                      labels = c(1, 2, 3, 4, 5, 6, 7))

#Diamonds$clarity=factor(Diamonds$clarity,
#                      levels = c('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'),
#                      labels = c(1, 2, 3, 4, 5, 6, 7, 8))


#X<-Diamonds%>%select(-c('price'))

# Create training (70%) and test (30%) sets for the Diamonds data.
# Use set.seed for reproducibility
set.seed(123)
diamond_split <- initial_split(Diamonds, prop = .7, strata = "price")
diamond_train <- training(diamond_split)
diamond_test  <- testing(diamond_split)

# Create training and testing feature model matrices and response vectors.
# we use model.matrix(...)[, -1] to discard the intercept
diamond_train_x <- model.matrix(price ~ ., diamond_train)[, -1]
diamond_train_y <- log(diamond_train$price)

diamond_test_x <- model.matrix(price ~ ., diamond_test)[, -1]
diamond_test_y <- log(diamond_test$price)

#the dimension of of your feature matrix
dim(diamond_train_x)

#To apply a ridge model we can use the glmnet::glmnet function. The alpha parameter tells glmnet to perform a ridge (alpha = 0), lasso (alpha = 1)
#predictor variables are standardized when performing regularized regression. glmnet performs this for you. If you standardize your predictors prior to glmnet you can turn this argument off with standardize = FALSE
#set.seed(123)
diamond_ridge <- glmnet(
  x = diamond_train_x,
  y = diamond_train_y,
  alpha = 0
)

plot(diamond_ridge, xvar = "lambda")

diamond_ridge$lambda

#, to identify the optimal λ value we need to perform cross-validation (CV). cv.glmnet provides a built-in option to perform k-fold CV, and by default, performs 10-fold CV.

#In penalized regression, you need to specify a constant lambda to adjust the amount of the coefficient shrinkage. The best lambda for your data, can be defined as the lambda that minimize the cross-validation prediction error rate. This can be determined automatically using the function cv.glmnet().

# Apply CV Ridge regression to diamond data
#set.seed(123)
diamond_ridge <- cv.glmnet(
  x = diamond_train_x,
  y = diamond_train_y,
  alpha = 0
)

# plot results
plot(diamond_ridge)
#we constrain our coefficients with log(λ)≥0 penalty, the MSE rises considerably.
#The numbers at the top of the plot (299) just refer to the number of variables in the model. Ridge regression does not force any variables to exactly zero so all features will remain in the model

#The first and second vertical dashed lines represent the λ value with the minimum MSE and the largest  λ value within one standard error of the minimum MSE.

min(diamond_ridge$cvm)       # minimum MSE
diamond_ridge$lambda.min     # lambda for this min MSE

diamond_ridge$cvm[diamond_ridge$lambda == diamond_ridge$lambda.1se]  # 1 st.error of min MSE

diamond_ridge$lambda.1se  # lambda for this MSE

#Using lambda.min as the best lambda, gives the following regression coefficients:
  
coef(diamond_ridge, diamond_ridge$lambda.min)

coef(diamond_ridge, diamond_ridge$lambda.1se)


#Here we plot the coefficients across the λ values and the dashed red line represents the largest λ that falls within one standard error of the minimum MSE. This shows you how much we can constrain the coefficients while still maximizing predictive accuracy.

#Compute the final model using lambda.1se
#set.seed(123)
diamond_ridge_min <- glmnet(
  x = diamond_train_x,
  y = diamond_train_y,
  alpha = 0,
  lambda = diamond_ridge$lambda.1se
)

predictedridge <- diamond_ridge_min %>% predict(newx = diamond_test_x)

#MSE
mean((diamond_test_y-predictedridge)^2)


#Lasso regression

#set.seed(123)
diamond_lasso <- glmnet(
  x = diamond_train_x,
  y = diamond_train_y,
  alpha = 1
)

plot(diamond_lasso, xvar = "lambda")

#set.seed(123)
diamond_lasso <- cv.glmnet(
  x = diamond_train_x,
  y = diamond_train_y,
  alpha = 1
)

plot(diamond_lasso)

#Compute the final model using lambda.1se
#set.seed(123)
diamond_lasso_min <- glmnet(
  x = diamond_train_x,
  y = diamond_train_y,
  alpha = 1,
  lambda = diamond_lasso$lambda.1se
)


coef(diamond_ridge_min, diamond_ridge_min$lambda.1se)

coef(diamond_lasso_min, diamond_lasso_min$lambda.1se)

predictedlasso <- diamond_lasso_min %>% predict(newx = diamond_test_x)

#predictedlasso <- predict(diamond_lasso_min, s=diamond_lasso$lambda.1se,newx = diamond_test_x)


#MSE
mean((diamond_test_y-predictedlasso)^2)



