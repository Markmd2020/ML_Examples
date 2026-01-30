#######ML Examples in R######

#Load libraries
library(gmodels)
library(class)
library(tm)
library(SnowballC)
library(naivebayes)
library(wordcloud)
library(C50)
library(rpart)
library(Cubist)
library(rpart.plot)
library(neuralnet)
library(kernlab)
library(caret)
library(irr)
library(vcd)
library(tidyverse)
library(gbm)
library(word2vec)
library(Rtsne)

#Set seed to ensure reproducibility
set.seed(135)

#KNN Example: Chapter 3
## Step 2: Exploring and preparing the data ---- 

# import the CSV file
wbcd <- read.csv("C:/Data/wisc_bc_data.csv")

# examine the structure of the wbcd data frame
str(wbcd)

# drop the id feature
wbcd <- wbcd[-1]

# table of diagnosis
table(wbcd$diagnosis)

# recode diagnosis as a factor
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"),
                         labels = c("Benign", "Malignant"))

# table or proportions with more informative labels
round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)

# summarize three numeric features
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# test normalization function - result should be identical
normalize(c(1, 2, 3, 4, 5))
normalize(c(10, 20, 30, 40, 50))

# normalize the wbcd data
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))

# confirm that normalization worked
summary(wbcd_n$area_mean)

# create training and test data
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]

# create labels for training and test data

wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

## Step 3: Training a model on the data ----
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test,
                      cl = wbcd_train_labels, k = 21)

## Step 4: Evaluating model performance ----

# Create the cross tabulation of predicted vs. actual
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred,
           prop.chisq = FALSE)

## Step 5: Improving model performance ----

# use the scale() function to z-score standardize a data frame
wbcd_z <- as.data.frame(scale(wbcd[-1]))

# confirm that the transformation was applied correctly
summary(wbcd_z$area_mean)

# create training and test datasets
wbcd_train <- wbcd_z[1:469, ]
wbcd_test <- wbcd_z[470:569, ]

# re-classify test cases
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test,
                      cl = wbcd_train_labels, k = 21)

# Create the cross tabulation of predicted vs. actual
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred,
           prop.chisq = FALSE)

# try several different values of k
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]

k_values <- c(1, 5, 11, 15, 21, 27)

# for loop to test all k values in k_values vector
for (k_val in k_values) {
  wbcd_test_pred <- knn(train = wbcd_train,
                        test = wbcd_test,
                        cl = wbcd_train_labels,
                        k = k_val)
  CrossTable(x = wbcd_test_labels,
             y = wbcd_test_pred,
             prop.chisq = FALSE)
} 

##### Chapter 4: Classification using Naive Bayes --------------------

## Example: Filtering spam SMS messages ----
## Step 2: Exploring and preparing the data ---- 

# read the sms data into the sms data frame
sms_raw <- read.csv("C:/Data/sms_spam.csv")

# examine the structure of the sms data
str(sms_raw)

# convert spam/ham to factor.
sms_raw$type <- factor(sms_raw$type)

# examine the type variable more carefully
str(sms_raw$type)
table(sms_raw$type)

# build a corpus using the text mining (tm) package
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# examine the sms corpus
print(sms_corpus)
inspect(sms_corpus[1:2])

as.character(sms_corpus[[1]])
lapply(sms_corpus[1:2], as.character)

# clean up the corpus using tm_map()
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

# show the difference between sms_corpus and corpus_clean
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) # remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords()) # remove stop words
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation) # remove punctuation

# tip: create a custom function to replace (rather than remove) punctuation
removePunctuation("hello...world")
replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
replacePunctuation("hello...world")

# illustration of word stemming
wordStem(c("learn", "learned", "learning", "learns"))

sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace) # eliminate unneeded whitespace

# examine the final clean corpus
lapply(sms_corpus[1:3], as.character)
lapply(sms_corpus_clean[1:3], as.character)

# create a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

# alternative solution: create a document-term sparse matrix directly from the SMS corpus
sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

# alternative solution: using custom stop words function ensures identical result
sms_dtm3 <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = function(x) { removeWords(x, stopwords()) },
  removePunctuation = TRUE,
  stemming = TRUE
))

# compare the result
sms_dtm
sms_dtm2
sms_dtm3

# creating training and test datasets
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

# also save the labels
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels  <- sms_raw[4170:5559, ]$type

# check that the proportion of spam is similar
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

# word cloud visualization
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

# subset the training data into spam and ham groups
spam <- subset(sms_raw, type == "spam")
ham  <- subset(sms_raw, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

# indicator features for frequent words
findFreqTerms(sms_dtm_train, 5)

# save frequently-appearing terms to a character vector
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)

# create DTMs with only the frequent terms
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() convert_counts() to columns of train/test data
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

## Step 3: Training a model on the data ----
sms_classifier <- naive_bayes(sms_train, sms_train_labels)

## Step 4: Evaluating model performance ----
sms_test_pred <- predict(sms_classifier, sms_test)


CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## Step 5: Improving model performance ----
sms_classifier2 <- naive_bayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
 
## Example: Identifying Risky Bank Loans ----
## Step 2: Exploring and preparing the data ----
credit <- read.csv("C:/Data/credit.csv", stringsAsFactors = TRUE)
str(credit)

# look at two characteristics of the applicant
table(credit$checking_balance)
table(credit$savings_balance)

# look at two characteristics of the loan
summary(credit$months_loan_duration)
summary(credit$amount)

# look at the class variable
table(credit$default)

# create a random sample for training and test data
# use set.seed to use the same random number sequence as the tutorial
train_sample <- sample(1000, 900)
str(train_sample)

# split the data frames
credit_train <- credit[train_sample, ]
credit_test  <- credit[-train_sample, ]

# check the proportion of class variable
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

## Step 3: Training a model on the data ----
# build the simplest decision tree
credit_model <- C5.0(default ~ ., data = credit_train)

# display simple facts about the tree
credit_model

# display detailed information about the tree
summary(credit_model)

## Step 4: Evaluating model performance ----
# create a factor vector of predictions on test data
credit_pred <- predict(credit_model, credit_test)

# cross tabulation of predicted versus actual classes
CrossTable(credit_test$default, credit_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

## Step 5: Improving model performance ----

## Boosting the accuracy of decision trees
# boosted decision tree with 10 trials
credit_boost10 <- C5.0(default ~ ., data = credit_train,
                       trials = 10)
credit_boost10
summary(credit_boost10)

credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default')) 

## Example: Estimating Wine Quality ----
## Step 2: Exploring and preparing the data ----
wine <- read.csv("C:/Data/whitewines.csv")

# examine the wine data
str(wine)

# the distribution of quality ratings
hist(wine$quality)

# summary statistics of the wine data
summary(wine)

wine_train <- wine[1:3750, ]
wine_test <- wine[3751:4898, ]

## Step 3: Training a model on the data ----
# regression tree using rpart
m.rpart <- rpart(quality ~ ., data = wine_train)

# get basic information about the tree
m.rpart

# get more detailed information about the tree
summary(m.rpart)

# a basic decision tree diagram
rpart.plot(m.rpart, digits = 3)

# a few adjustments to the diagram
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)

## Step 4: Evaluate model performance ----

# generate predictions for the testing dataset
p.rpart <- predict(m.rpart, wine_test)

# compare the distribution of predicted values vs. actual values
summary(p.rpart)
summary(wine_test$quality)

# compare the correlation
cor(p.rpart, wine_test$quality)

# function to calculate the mean absolute error
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))  
}

# mean absolute error between predicted and actual values
MAE(p.rpart, wine_test$quality)

# mean absolute error between actual values and mean value
mean(wine_train$quality) # result = 5.87
MAE(5.87, wine_test$quality)

## Step 5: Improving model performance ----
# train a Cubist Model Tree
m.cubist <- cubist(x = wine_train[-12], y = wine_train$quality)

# display basic information about the model tree
m.cubist

# display the tree itself
summary(m.cubist)

# generate predictions for the model
p.cubist <- predict(m.cubist, wine_test)

# summary statistics about the predictions
summary(p.cubist)

# correlation between the predicted and true values
cor(p.cubist, wine_test$quality)

# mean absolute error of predicted and true values
# (uses a custom function defined above)
MAE(wine_test$quality, p.cubist)

##### Chapter 7: Neural Networks and Support Vector Machines -------------------

##### Part 1: Neural Networks -------------------
## Example: Modeling the Strength of Concrete  ----

## Step 2: Exploring and preparing the data ----
# read in data and examine structure
concrete <- read.csv("C:/Data/concrete.csv")
str(concrete)

# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# apply normalization to entire data frame
concrete_norm <- as.data.frame(lapply(concrete, normalize))

# confirm that the range is now between zero and one
summary(concrete_norm$strength)

# compared to the original minimum and maximum
summary(concrete$strength)

# create training and test data
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

## Step 3: Training a model on the data ----
# train the neuralnet model

# simple ANN with only a single hidden neuron
concrete_model <- neuralnet(strength ~ cement + slag +
                              ash + water + superplastic + 
                              coarseagg + fineagg + age,
                            data = concrete_train)

# visualize the network topology
plot(concrete_model)

## Step 4: Evaluating model performance ----
# obtain model results
model_results <- compute(concrete_model, concrete_test[1:8])
# obtain predicted strength values
predicted_strength <- model_results$net.result
# examine the correlation between predicted and actual values
cor(predicted_strength, concrete_test$strength)

## Step 5: Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
set.seed(135) # to guarantee repeatable results
concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                             data = concrete_train, hidden = 5)

# plot the network
plot(concrete_model2)

# evaluate the results as we did before
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)

# an EVEN MORE complex neural network topology with two hidden layers and custom activation function

# create a custom softplus activation function
softplus <- function(x) { log(1 + exp(x)) }

concrete_model3 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                             data = concrete_train, hidden = c(5, 5), act.fct = softplus)

# plot the network
plot(concrete_model3)

# evaluate the results as we did before
model_results3 <- compute(concrete_model3, concrete_test[1:8])
predicted_strength3 <- model_results3$net.result
cor(predicted_strength3, concrete_test$strength)

# note that the predicted and actual values are on different scales
strengths <- data.frame(
  actual = concrete$strength[774:1030],
  pred = predicted_strength3
)

head(strengths, n = 3)

# correlation is unaffected by normalization...
# ...but measures like percent error would be affected by the change in scale!
cor(strengths$pred, strengths$actual)
cor(strengths$pred, concrete_test$strength)

# create an unnormalize function to reverse the normalization
unnormalize <- function(x) { 
  return(x * (max(concrete$strength) -
                min(concrete$strength)) + min(concrete$strength))
}

strengths$pred_new <- unnormalize(strengths$pred)
strengths$error_pct <- (strengths$pred_new - strengths$actual) / strengths$actual
head(strengths, n = 3)

# correlation stays the same despite reversing the normalization
cor(strengths$pred_new, strengths$actual)

##### Part 2: Support Vector Machines -------------------
## Example: Optical Character Recognition ----

## Step 2: Exploring and preparing the data ----
# read in data and examine structure
letters <- read.csv("C:/Data/letterdata.csv", stringsAsFactors = TRUE)
str(letters)

# divide into training and test data
letters_train <- letters[1:16000, ]
letters_test  <- letters[16001:20000, ]

## Step 3: Training a model on the data ----
# begin by training a simple linear SVM
letter_classifier <- ksvm(letter ~ ., data = letters_train,
                          kernel = "vanilladot")

# look at basic information about the model
letter_classifier

## Step 4: Evaluating model performance ----
# predictions on testing dataset
letter_predictions <- predict(letter_classifier, letters_test)

head(letter_predictions)

table(letter_predictions, letters_test$letter)

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement <- letter_predictions == letters_test$letter
table(agreement)
prop.table(table(agreement))

## Step 5: Improving model performance ----

# change to a RBF kernel
letter_classifier_rbf <- ksvm(letter ~ ., data = letters_train, kernel = "rbfdot")
letter_predictions_rbf <- predict(letter_classifier_rbf, letters_test)

agreement_rbf <- letter_predictions_rbf == letters_test$letter
table(agreement_rbf)
prop.table(table(agreement_rbf))

# test various values of the cost parameter
cost_values <- c(1, seq(from = 5, to = 40, by = 5))

accuracy_values <- sapply(cost_values, function(x) {
  m <- ksvm(letter ~ ., data = letters_train,
            kernel = "rbfdot", C = x)
  pred <- predict(m, letters_test)
  agree <- ifelse(pred == letters_test$letter, 1, 0)
  accuracy <- sum(agree) / nrow(letters_test)
  return (accuracy)
})

plot(cost_values, accuracy_values, type = "b")

## Automating 10-fold CV for a C5.0 Decision Tree using lapply() ----
credit <- read.csv("C:/Data/credit.csv", stringsAsFactors = TRUE)

folds <- createFolds(credit$default, k = 10)

cv_results <- lapply(folds, function(x) {
  credit_train <- credit[-x, ]
  credit_test <- credit[x, ]
  credit_model <- C5.0(default ~ ., data = credit_train)
  credit_pred <- predict(credit_model, credit_test)
  credit_actual <- credit_test$default
  kappa <- kappa2(data.frame(credit_actual, credit_pred))$value
  return(kappa)
})

# examine the results of the 10 trials
str(cv_results)

# compute the average kappa across the 10 trials
mean(unlist(cv_results))

# compute the standard deviation across the 10 trials
sd(unlist(cv_results)) 

#Categorical Data Feature Encoding  Example
titanic_train <- read_csv("C:/Data/titanic_train.csv") |>
  mutate(Title = str_extract(Name, ", [A-z]+\\.")) |>
  mutate(Title = str_replace_all(Title, "[, \\.]", ""))

# the Title feature has a large number of categories
table(titanic_train$Title, useNA = "ifany")

# group categories with similar real-world meaning
titanic_train <- titanic_train |>
  mutate(TitleGroup = fct_collapse(Title, 
                                   Mr = "Mr",
                                   Mrs = "Mrs",
                                   Master = "Master",
                                   Miss = c("Miss", "Mlle", "Mme", "Ms"),
                                   Noble = c("Don", "Sir", "Jonkheer", "Lady"),
                                   Military = c("Capt", "Col", "Major"),
                                   Doctor = "Dr",
                                   Clergy = "Rev",
                                   other_level = "Other")
  ) |>
  mutate(TitleGroup = fct_na_value_to_level(TitleGroup,
                                            level = "Unknown"))

# examine the recoding
table(titanic_train$TitleGroup)

# look at the counts and proportions of all levels, sorted largest to smallest
fct_count(titanic_train$Title, sort = TRUE, prop = TRUE)

# lump together everything outside of the top three levels
table(fct_lump_n(titanic_train$Title, n = 3))

# lump together everything with less than 1%
table(fct_lump_prop(titanic_train$Title, prop = 0.01))

# lump together everything with fewer than 5 observations
table(fct_lump_min(titanic_train$Title, min = 5))

#Hyper parameter tuning

# load the credit dataset
credit <- read.csv("C:/Data/credit.csv")

## Customizing the tuning process ----
# use trainControl() to alter resampling strategy
ctrl <- trainControl(method = "cv", number = 10,
                     selectionFunction = "oneSE")

# use expand.grid() to create grid of tuning parameters
grid <- expand.grid(model = "tree",
                    trials = c(1, 5, 10, 15, 20, 25, 30, 35),
                    winnow = FALSE)

# look at the result of expand.grid()
grid

# customize train() with the control list and grid of parameters 
m <- train(default ~ ., data = credit, method = "C5.0",
           metric = "Kappa",
           trControl = ctrl,
           tuneGrid = grid)

# see the results
m

# create a tuned gbm() model using caret
# start by creating the tuning grid
grid_gbm <- expand.grid(
  n.trees = c(100, 150, 200),
  interaction.depth = c(1, 2, 3),
  shrinkage = c(0.01, 0.1, 0.3),
  n.minobsinnode = 10
)

ctrl <- trainControl(method = "cv", number = 10,
                     selectionFunction = "best")

m_gbm_c <- train(default ~ ., data = credit, method = "gbm",
                 trControl = ctrl, tuneGrid = grid_gbm,
                 metric = "Kappa",
                 verbose = FALSE)

# see the results
m_gbm_c

#Vector Embeddings
# load the Google-trained 300-dimension word2vec embedding
m_w2v <- read.word2vec(file = "GoogleNews-vectors-negative300.bin",
                       normalize = TRUE)

# examine the structure of the model
str(m_w2v)

# obtain the vector for a few terms
foods <- predict(m_w2v, c("cereal", "bacon", "eggs", "sandwich", "salad", "steak", "spaghetti"), type = "embedding")
meals <- predict(m_w2v, c("breakfast", "lunch", "dinner"), type = "embedding")

# examine a single word vector
head(foods["cereal", ])

# examine the first few columns
foods[, 1:5]

# compute the similarity between the foods and meals
word2vec_similarity(foods, meals)

# can also use cosine similarity (not shown in book)
word2vec_similarity(foods, meals, type = "cosine")

#Dimensionality Reduction using t-SNE
sns_sample <- read_csv("C:/Data/snsdata.csv") |>
  slice_sample(n = 1000)

sns_tsne <- sns_sample |>
  select(basketball:drugs) |>
  Rtsne(check_duplicates = FALSE)

# visualize the t-SNE result
data.frame(sns_tsne$Y) |>
  ggplot(aes(X1, X2)) + geom_point(size = 2, shape = 1) 

# create a categorical feature for the number of terms used
sns_sample_tsne <- sns_sample |>
  bind_cols(data.frame(sns_tsne$Y)) |> # add the t-SNE data
  rowwise() |> # work across rows rather than columns
  mutate(n_terms = sum(c_across(basketball:drugs))) |>
  ungroup() |> # remove rowwise behavior
  mutate(`Terms Used` = if_else(n_terms > 0, "1+", "0"))

# visualize the t-SNE result by number of terms used
sns_sample_tsne |>
  ggplot(aes(X1, X2, shape = `Terms Used`, color = `Terms Used`)) +
  geom_point(size = 2) +
  scale_shape(solid = FALSE)
