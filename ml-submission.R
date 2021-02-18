#### General project info ####
# Project code for the movielense projecect #
# Part of Data Science, Capstone, HarvardX #
# submitted by Marcus Schmidt, run on R version 4.0.3 #

#### Part 1 - Package installation and loading ####

# installing packages if necessary
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# loading packages
library(tidyverse)
library(caret)
library(data.table)

#### Part 2 - Creating the data set #### 
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# if using R 3.6 or earlier, instead use:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

#### Interlude - saving and loading movielens dataset as .Rdata file ####
# this is so I can load immediately from .Rdata file, not going through the process 
# of downloading every time. This part can be ignored when grading:

# save(movielens, file = "P:\\r\\projects\\ex-data\\movielens.Rdata") # save at work
# save(movielens, file = "~/r-projects/ex-data/movielens.Rdata") # load at home

# load("P:\\r\\projects\\ex-data\\movielens.Rdata") # load at work
# load("~/r-projects/ex-data/movielens.Rdata") # load at home

#### Part 3 - Short exploration + adding 'week' variable ####

dim(movielens) # dimension of dataset
names(movielens) # variable names
as_tibble(head((movielens))) # see variable types

# package 'lubridate' for rounding date to week
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
library(lubridate)

# add rounded week to dataset, also factorize movieId and userId
movielens <- movielens %>% mutate(movieId = as.factor(movieId), userId = as.factor(userId),
                                  week = round_date(as_datetime(timestamp), "week"))
# check if adding a week variable worked
str(movielens$week)

#### Part 4 - Creating a 10% Validation set ####

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId and genre and week in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId") %>% 
  semi_join(edx, by = "week") %>% 
  semi_join(edx, by = "genres")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
dim(removed)
edx <- rbind(edx, removed)

#### Part 5 - Creating a 10% Test / 90% Train set from edx ####
# note: reducing the test set to 0.1 helped to improve the result
# interpretation: higher amount of rows for training improved accuracy
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId and week in test set are also in train set
edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")%>% 
  semi_join(edx_train, by = "week") %>% 
  semi_join(edx_train, by = "genres")

dim(edx_test)

# Add rows removed from test set back into train set
removed <- anti_join(temp, edx_test)
dim(removed)

edx_train <- rbind(edx_train, removed)

dim(edx_train)
dim(edx_test)


#### Part 6 - Setting up RMSE function to evaluate models ####
RMSE <- function(ratings, predictions){
  sqrt(mean((ratings - predictions)^2))
}


#### Part 7 - Setting up a reference model - guessing the outcome ####
# calculating the overall mean rating in train set
mu <- mean(edx_train$rating) 
mu # 3.51

# calculating RMSE for reference model
rmse_guessing <- RMSE(edx_test$rating, mu)
rmse_guessing

# plotting 100 predictions for reference model
qplot(edx_test$rating[1:100], mu) 

# building a RMSE table to see model performances
rmse_table <- data.frame(model = "guessing", rmse = rmse_guessing)
rmse_table


#### Part 8 - Creating linear models ####
#### Part 8.1. - Movie effect ####

# create effect-variable = the deviation fromt he overall mean for this effect
movie_means <- edx_train %>% group_by(movieId) %>% summarize(movie_effect = mean(rating - mu))
head(movie_means)

hist(movie_means$movie_effect) # visualize distribution of the effect

pred1 <- mu + # adding the effect to the mean
  edx_test %>% left_join(movie_means, by = "movieId") %>% .$movie_effect # .$ equals pull()

# calculating RMSE
rmse_1 <- RMSE(edx_test$rating, pred1)
rmse_1

# adding effect to the table
rmse_table <- rbind(rmse_table,
                    data.frame(model = "lm movie effect", rmse = rmse_1))
rmse_table


#### Part 8.2. - Movie & user effect ####

# create effect variable, note that this is on top of the movie effect
user_means <- edx_train %>% left_join(movie_means, by = "movieId") %>% 
  group_by(userId) %>% summarize(user_effect = mean(rating - mu - movie_effect))
head(user_means)

pred2 <- mu + # adding the effects to the mean
  edx_test %>% left_join(movie_means, by = "movieId") %>% .$movie_effect +
  edx_test %>% left_join(user_means, by = "userId") %>% .$user_effect

# calculating RMSE
rmse_2 <- RMSE(edx_test$rating, pred2)
rmse_2

# adding effect to the table
rmse_table <- rbind(rmse_table,
                    data.frame(model = "lm movie + user effect", rmse = rmse_2))
rmse_table

# plotting 1000 predictions
qplot(edx_test$rating[1:1000], pred2[1:1000]) 



#### Part 8.3. - Movie & user & time effect ####

# create effect variable, note that this is on top of the movie & user effects
time_means <- edx_train %>% 
  left_join(movie_means, by = "movieId") %>% 
  left_join(user_means, by = "userId") %>% 
  group_by(week) %>% summarize(time_effect = mean(rating - mu - movie_effect - user_effect))
head(time_means)

pred3 <- mu + # adding the effects to the mean
  edx_test %>% left_join(movie_means, by = "movieId") %>% .$movie_effect +
  edx_test %>% left_join(user_means, by = "userId") %>% .$user_effect +
  edx_test %>% left_join(time_means, by = "week") %>% .$time_effect

# calculating RMSE
rmse_3 <- RMSE(edx_test$rating, pred3)
rmse_3

# adding effect to the table
rmse_table <- rbind(rmse_table,
                    data.frame(model = "lm movie + user + time effect", rmse = rmse_3))
rmse_table

# plotting 1000 predictions
qplot(edx_test$rating[1:1000], pred3[1:1000]) 



#### Part 8.4. - Movie & user & time & genre effect ####

# create effect variable, note that this is on top of the movie & user & time effects
genre_means <- edx_train %>% 
  left_join(movie_means, by = "movieId") %>% 
  left_join(user_means, by = "userId") %>% 
  left_join(time_means, by = "week") %>% 
  group_by(genres) %>% summarize(genre_effect = mean(rating - mu - movie_effect - user_effect - time_effect))
head(genre_means)

pred4 <- mu + # adding the effects to the mean
  edx_test %>% left_join(movie_means, by = "movieId") %>% .$movie_effect +
  edx_test %>% left_join(user_means, by = "userId") %>% .$user_effect +
  edx_test %>% left_join(time_means, by = "week") %>% .$time_effect +
  edx_test %>% left_join(genre_means, by = "genres") %>% .$genre_effect

# calculating RMSE
rmse_4 <- RMSE(edx_test$rating, pred4)
rmse_4

# adding the effect to the table
rmse_table <- rbind(rmse_table,
                    data.frame(model = "lm movie + user + time + genre effect", rmse = rmse_4))
rmse_table

# plotting 1000 predictions
qplot(edx_test$rating[1:1000], pred4[1:1000]) 


#### Interlude - a few conclusions ####
# (a) Adding movie and user effects strongly improves the model
# (b) Adding time and genre effects slightly improves the model



#### Part 9 - Creating linear models with regularization ####
# note: regularization reduces the effect of categories that appear less often
# through dividing through a higher number when taking the mean effect of that category


#### Part 9.1. Movie effect with regularization ####

# choosing lambda, the penalty number 
lambdas <- seq(0,5, by = 0.5)

# running trough a range of lambdas 
result <- sapply(lambdas, function(lambda){
  movie_reg_means <- edx_train %>% 
    group_by(movieId) %>% 
    summarize(movie_effect = sum(rating - mu)/(n()+lambda), n_movies = n()) # here, lambda is added
  
  pred_1r <- mu +
    edx_test %>% left_join(movie_reg_means, by = "movieId") %>% .$movie_effect
  
  rmse_1r <- RMSE(edx_test$rating, pred_1r)
  rmse_1r
})

# plotting lambdas and resulting RMSE
qplot(lambdas, result)

# choosing best lambda
best_lambda <- lambdas[which.min(result)]
best_lambda

# from here, the chosen lambda is taken to calculate the regularized effect
lambda = best_lambda

movie_reg_means <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(movie_effect = sum(rating - mu)/(n()+lambda), n_movies = n()) # here, lamda is added

head(movie_reg_means)

pred_1r <- mu +
  edx_test %>% left_join(movie_reg_means, by = "movieId") %>% .$movie_effect

rmse_1r <- RMSE(edx_test$rating, pred_1r)
rmse_1r

rmse_table <- rbind(rmse_table,
                    data.frame(model = "lm movie effect (with regularization)", rmse = rmse_1r))
rmse_table

# plotting 1000 predictions
qplot(edx_test$rating[1:1000], pred_1r[1:1000]) 


#### Part 9.2. - Movie + user effect with regulatization ####

# choosing lambda, the penalty number
lambdas <- seq(0,7, by = 1)

result <- sapply(lambdas, function(lambda){
  user_reg_means <- edx_train %>% 
    left_join(movie_reg_means, by = "movieId") %>% 
    group_by(userId) %>% 
    summarize(user_effect = sum(rating - mu - movie_effect)/(n()+lambda), n_user = n())

  pred_2r <- mu +
    edx_test %>% left_join(movie_reg_means, by = "movieId") %>% .$movie_effect +
    edx_test %>% left_join(user_reg_means, by = "userId") %>% .$user_effect
  
  rmse_2r <- RMSE(edx_test$rating, pred_2r)
  rmse_2r
})

# plotting lambdas and resulting RMSE
qplot(lambdas, result)

# choosing best lambda
best_lambda <- lambdas[which.min(result)]
best_lambda


#from here, the chosen lambda is taken to calculate the regularized effect
lambda = best_lambda

user_reg_means <- edx_train %>% 
  left_join(movie_reg_means, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(user_effect = sum(rating - mu - movie_effect)/(n()+lambda), n_user = n())
head(user_means)

pred_2r <- mu +
  edx_test %>% left_join(movie_reg_means, by = "movieId") %>% .$movie_effect +
  edx_test %>% left_join(user_reg_means, by = "userId") %>% .$user_effect

rmse_2r <- RMSE(edx_test$rating, pred_2r)
rmse_2r

rmse_table <- rbind(rmse_table,
                    data.frame(model = "lm movie + user effect (with regulatization)", rmse = rmse_2r))
rmse_table

# plotting 1000 predictions
qplot(edx_test$rating[1:1000], pred_2r[1:1000]) 



#### Part 9.3. - Movie + user + time effect with regularization ####

#choosing lambda, the penalty number
lambdas <- c(0,3,6,7,8,9,10,15,20)

result <- sapply(lambdas, function(lambda){
  time_reg_means <- edx_train %>% 
    left_join(movie_reg_means, by = "movieId") %>% 
    left_join(user_reg_means, by = "userId") %>% 
    group_by(week) %>% summarize(time_effect = sum(rating - mu - movie_effect - user_effect)/(n()+lambda), n_time = n())
  head(time_reg_means)
  
  pred_3r <- mu +
    edx_test %>% left_join(movie_reg_means, by = "movieId") %>% .$movie_effect +
    edx_test %>% left_join(user_reg_means, by = "userId") %>% .$user_effect +
    edx_test %>% left_join(time_reg_means, by = "week") %>% .$time_effect
  
  rmse_movie_user_time_reg_effect <- RMSE(edx_test$rating, pred_3r)
  rmse_movie_user_time_reg_effect
})

# plotting lambdas and resulting RMSE
qplot(lambdas, result)

# choosing best lambda
best_lambda <- lambdas[which.min(result)]
best_lambda



#from here, the chosen lambda is taken to calculate the regularized effect
lambda = best_lambda

time_reg_means <- edx_train %>% 
  left_join(movie_reg_means, by = "movieId") %>% 
  left_join(user_reg_means, by = "userId") %>% 
  group_by(week) %>% summarize(time_effect = sum(rating - mu - movie_effect - user_effect)/(n()+lambda), n_time = n())
head(time_reg_means)

pred_3r <- mu +
  edx_test %>% left_join(movie_reg_means, by = "movieId") %>% .$movie_effect +
  edx_test %>% left_join(user_reg_means, by = "userId") %>% .$user_effect +
  edx_test %>% left_join(time_reg_means, by = "week") %>% .$time_effect

rmse_3r <- RMSE(edx_test$rating, pred_3r)
rmse_3r

rmse_table <- rbind(rmse_table,
                    data.frame(model = "lm movie + user + time effect (with regulatization)", rmse = rmse_3r))
rmse_table

# plotting 1000 predictions
qplot(edx_test$rating[1:1000], pred_3r[1:1000])



#### Part 9.4. - Movie + user + time + genre effect with regulatization ####

# choose lambda, the penalty number
lambdas <- seq(0, 5, by = 1)

result <- sapply(lambdas, function(lambda){
  genre_reg_means <- edx_train %>% 
  left_join(movie_reg_means, by = "movieId") %>% 
  left_join(user_reg_means, by = "userId") %>% 
  left_join(time_reg_means, by = "week") %>% 
  group_by(genres) %>% summarize(genre_effect = sum(rating - mu - movie_effect - user_effect - time_effect)/(n()+lambda), n_genre = n())

pred_4r <- mu +
  edx_test %>% left_join(movie_reg_means, by = "movieId") %>% .$movie_effect +
  edx_test %>% left_join(user_reg_means, by = "userId") %>% .$user_effect +
  edx_test %>% left_join(time_reg_means, by = "week") %>% .$time_effect +
  edx_test %>% left_join(genre_reg_means, by = "genres") %>% .$genre_effect

rmse_movie_user_time_genre_reg_effect <- RMSE(edx_test$rating, pred_4r)
rmse_movie_user_time_genre_reg_effect
})

# plotting lambdas and resulting RMSE
qplot(lambdas, result)

# choosing best lambda
best_lambda <- lambdas[which.min(result)]
best_lambda

# best lambda is 0 here so it does not improve the result


# from here, the chosen lambda is taken to calculate the regularized effect
lambda = best_lambda

genre_reg_means <- edx_train %>% 
  left_join(movie_reg_means, by = "movieId") %>% 
  left_join(user_reg_means, by = "userId") %>% 
  left_join(time_reg_means, by = "week") %>% 
  group_by(genres) %>% summarize(genre_effect = sum(rating - mu - movie_effect - user_effect - time_effect)/(n()+lambda), n_genre = n())
head(genre_reg_means)

pred_4r <- mu +
  edx_test %>% left_join(movie_reg_means, by = "movieId") %>% .$movie_effect +
  edx_test %>% left_join(user_reg_means, by = "userId") %>% .$user_effect +
  edx_test %>% left_join(time_reg_means, by = "week") %>% .$time_effect +
  edx_test %>% left_join(genre_reg_means, by = "genres") %>% .$genre_effect

rmse_4r <- RMSE(edx_test$rating, pred_4r)
rmse_4r

rmse_table <- rbind(rmse_table,
                    data.frame(model = "lm movie + user + time + genre effect (with regularization)", rmse = rmse_4r))
rmse_table

# plotting 1000 predictions
qplot(edx_test$rating[1:1000], pred_4r[1:1000])

#### PART 10 - adding a limit cut ####

# check whether there are predictions which are below 1 or higher than 5 which they cannot be
range(pred_4r)

# letting a function run over the predictions and setting 
# those that are lower than 1 to 1 and
# those that are higher than 5 to 5
pred_4r_limcut <- 
  sapply(pred_4r, function(x){
    if(x < 1){x <- 1}
    if(x > 5){x <- 5}
    x
  })

# calculating RMSE
rmse_4r_limcut <- RMSE(edx_test$rating, pred_4r_limcut)
rmse_4r_limcut

rmse_table <- rbind(rmse_table,
                    data.frame(model = "lm 4 regularized effects + limit cut", rmse = rmse_4r_limcut))
rmse_table



  
#### PART 11 - FINAL EVALUATION OF BEST MODEL ####

# this is the model with four regulatized effect (movie, user, time & genre),
# with a cut of the predicted values in case they are above 5 or below 0

pred_final <- (
  mu +
  validation %>% left_join(movie_reg_means, by = "movieId") %>% .$movie_effect +
  validation %>% left_join(user_reg_means, by = "userId") %>% .$user_effect +
  validation %>% left_join(time_reg_means, by = "week") %>% .$time_effect +
  validation %>% left_join(genre_reg_means, by = "genres") %>% .$genre_effect
  ) %>% 
  sapply(function(x){
    if(x < 1){x <- 1}
    if(x > 5){x <- 5}
    x
  })
  
# check a few values to see if they make sense:
head(pred_final)

# calculating the final RMSE
rmse_final <- RMSE(validation$rating, pred_final)
rmse_final

# plotting 1000 predictions for validation set
qplot(validation$rating[1:5000], pred_final[1:5000])  

# final check whether all 999999 rows were predicted
length(pred_final)

# final check whether there are no NAs in predictions
table(!is.na(pred_final))






#### Ccnclusions drawn ####
# (a) Take a small subsample as a test set. When the data set is large, 
# 10 % is enough and improves the RMSE compared to 50%
# (b) Add several effects, often regularization also improves RMSE
# (c) KNN, random forest and recommenderlab exceed calculation capacity
# (at least on a regular computer with this large dataset)
# (d) a cut of predicted values below and above values that make sense
# may also improve the RMSE


