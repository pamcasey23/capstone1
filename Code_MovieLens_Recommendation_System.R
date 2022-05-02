# MovieLens Recommendation System
# 1 Install packages -----------------------------------------
if(!require(tidyverse)) install.packages(
  "tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages(
  "caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages(
  "data.table", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
# 2 Format edx & validation sets from MovieLens data ---------------------
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% 
# mutate(movieId = as.numeric(levels(movieId))[movieId],
# title = as.character(title), genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(movieId),
  title= as.character(title), genres= as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- 
  createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

#Ensure userId & movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, 
   movielens, removed)
# End of given code #

# 3 Create train and test sets from edx dataset ------
# Create a train and test set from the edx dataset to
# train and test models. Test set is 10% of edx data.
set.seed(1, sample.kind="Rounding")
model_test_index <- 
  createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
mod_train <- edx[-model_test_index,]
tempm <- edx[model_test_index,]

# Ensure userId & movieId in validation set are also in edx set
mod_test <- tempm %>% 
  semi_join(mod_train, by = "movieId") %>%
  semi_join(mod_train, by = "userId")
# Add rows removed from validation set back into edx set
removed <- anti_join(tempm, mod_test)
mod_train <- rbind(mod_train, removed)

# 4 Root Mean Square Error Function ----------------
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# 5 Naive Model ---------------------------------------
#Test RMSE based on using average (mu) for all unknown ratings
mu <- mean(mod_train$rating)
naive_rmse <- RMSE(mod_test$rating, mu)
naive_rmse

# 6 Movie Bias Model ---------------------------------
#Calculate the movie bias for each movie (bias_m)
# bias_m = (sum(actual movie ratings - mu))/ # ratings 
movie_bias <- mod_train %>% 
  group_by(movieId) %>% 
  summarize(bias_m = mean(rating - mu))

# Histogram: Movie Bias Distribution
qplot(bias_m, data = movie_bias, bins = 20, 
      color = I("black"), xlab = "bias = rating - mu", 
      ylab="count", main="Movie Bias Distribution")

# Add movie-specific bias to mu to predict rating
pred_ratings_bm <- 
  left_join(mod_test, movie_bias, by='movieId') %>% 
  mutate(adj_rating_bm = mu + bias_m)
#RMSE based on predicted ratings adjusted for movie bias
rmse_with_bias_m <- 
  RMSE(mod_test$rating, pred_ratings_bm$adj_rating_bm)
rmse_with_bias_m

# 7 User Bias Model ----------------------------------
# Calculate individual user bias based on the ratings
# that were adjusted for movie bias
user_movie_adj <- mod_train %>%
  left_join(movie_bias, by='movieId') %>%
  group_by(userId) %>% 
  mutate(bias_u = mean(rating - mu - bias_m)) %>% 
  distinct(userId,bias_u)

# Histogram of user bias
qplot(bias_u, data = user_movie_adj, bins = 30, 
      color = I("black"), 
      xlab="rating - adjusted rating",
      ylab="count", xlim = c(-3,2),
      main="User Bias Distribution")

# Calculate individual stats for reference
mod_train %>% filter(grepl('Gump', title)) %>% 
  summarize(mean(rating))
mod_train %>% filter(grepl('Titanic', title)) %>% 
  summarize(mean(rating))

# Adjust the predictions with the addition of the user bias
test_pred_both <- mod_test %>%
  left_join(movie_bias, by = "movieId") %>% 
  left_join(user_movie_adj, by = "userId") %>% 
  mutate(pred_rat_both = mu + bias_m + bias_u)

# Calculate new RMSE
rmse_with_bias_both <- 
  RMSE(mod_test$rating, test_pred_both$pred_rat_both)
rmse_with_bias_both

# user_avgs_givencode <- mod_train %>% 
#   left_join(movie_bias, by='movieId') %>%
#   group_by(userId) %>%
#   summarize(b_u = mean(rating - mu - bias_m))

# RMSE results on test including movie and user bias ---------
rmse_test_results <- tibble(Prediction_Method = c(
      "Average of Ratings", "Adjusted for Movie Bias",
        "Adjusted for Movie and User Bias"), 
        RMSE = c(naive_rmse, rmse_with_bias_m,
                       rmse_with_bias_both))
rmse_test_results

# 8 Regularization prelim analysis ----
# Visualize movie ratings with greatest +/- bias
large_bias <- mod_train %>%
  left_join(movie_bias, by = "movieId") %>% 
  left_join(user_movie_adj, by = "userId") %>% 
  mutate(bias = bias_m + bias_u) %>% 
  group_by(movieId) %>% 
  summarize(mean_bias = mean(bias), n = n()) %>% 
  filter(abs(mean_bias) > 2.5)
nrow(large_bias)
# There are 32 movies with bias_m < -2.5
# There are no movies with bias_m > 2.5

# Plot of |bias_m|>2.5 vs # of Ratings
qplot(mean_bias, n, data = large_bias, 
      color = I("black"), xlab="Movie and User Bias", 
      ylab = "# of ratings", 
      main="Number of Ratings vs Large Bias")

# 9 Regularize the data ----------------------------------
# Determine rmse's with different tuning parameters (lambdas)

lambdas <- seq(0.1, 7, 0.2)
rmses_on_train <- sapply(lambdas, function(lambda){
movie_bias_reg <- mod_train %>% 
  group_by(movieId) %>% 
  summarize(bias_m_reg = sum(rating - mu)/(n()+lambda))

user_bias_reg <- mod_train %>%
  left_join(movie_bias_reg, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(bias_u_reg = sum(rating - mu - bias_m_reg)/(n()+lambda))

ratings_reg <-
  mod_test %>% 
  left_join(movie_bias_reg, by = "movieId") %>% 
  left_join(user_bias_reg, by = "userId") %>% 
  mutate(pred_ratings_reg = mu + bias_m_reg + bias_u_reg) %>% 
  pull(pred_ratings_reg)

return(RMSE(mod_test$rating, ratings_reg))
})

qplot(lambdas, rmses_on_train, ylab="RMSE",
      main="Tuning Parameter on Test Set")
lambda <- lambdas[which.min(rmses_on_train)]
lambda
rmse_with_reg <- min(rmses_on_train)
rmse_with_reg

#Compare new regularized movie biases with 
#lambda = 4.9 with unregularized movie biases 
lambda_reg = 4.9
ratings_per_movie <- mod_train %>% 
  group_by(movieId) %>% 
  summarize(ratings_per_movie = n())
regset_test <- mod_train %>% 
    group_by(movieId) %>% 
    summarize(bias_m_reg = sum(rating - mu)/(n()+lambda_reg))
reg_biasm_comp <- movie_bias %>% 
  left_join(regset_test, by="movieId") %>% 
  left_join(ratings_per_movie, by="movieId") %>% 
  mutate(bias_diff=bias_m-bias_m_reg) %>% distinct()
plot(sqrt(reg_biasm_comp$ratings_per_movie), reg_biasm_comp$bias_diff,
     xlab="sqrt of Ratings per Movie",
     ylab="Diff between Reg Bias and Movie Bias",
     main="Regularization Effect on Movie Bias")

# RMSE results on test dataset with Regularization -----
rmse_test_results_with_reg <- tibble(Prediction_Method = c(
  "Average of Ratings", "Adjusted for Movie Bias",
  "Adjusted for Movie and User Bias", "Regularized"), 
  RMSE = c(naive_rmse, rmse_with_bias_m,
           rmse_with_bias_both, rmse_with_reg))
rmse_test_results_with_reg

# Apply the optimal lambda of 4.9 into the code for 
# Regularization which will be used as the regularization 
# bias in the final model
lambda <- 4.9
movie_bias_regl <- mod_train %>% 
  group_by(movieId) %>% 
  summarize(bias_m_regl = sum(rating - mu)/(n()+lambda))

user_bias_regl <- mod_train %>%
  left_join(movie_bias_regl, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(bias_u_regl = sum(rating - mu - bias_m_regl)/(n()+lambda))

ratings_regl <-mod_test %>% 
  left_join(movie_bias_regl, by = "movieId") %>% 
  left_join(user_bias_regl, by = "userId") %>% 
  mutate(pred_ratings_regl =mu+bias_m_regl+bias_u_regl) %>% 
  pull(pred_ratings_regl)

# 10 Set all adjusted ratings over 5 back to a rating of 5 -----
# and all adjusted ratings under 0.5 to a rating of 0.5
high_low <- mod_test %>% 
  left_join(movie_bias_regl, by = "movieId") %>% 
  left_join(user_bias_regl, by = "userId") %>% 
  mutate(pred_ratings_regl =mu+bias_m_regl+bias_u_regl) %>%
  mutate(no_over=ifelse(pred_ratings_regl>5, 5, 
                        pred_ratings_regl)) %>% 
  mutate(no_over_under=ifelse(pred_ratings_regl<0.5,0.5,
                              no_over))

rmse_no_over_under <- RMSE(mod_test$rating, high_low$no_over_under)
rmse_no_over_under

#Finally to run the model using "edx" and "validation" ------------
#Movie Bias Model
mu_x <- mean(edx$rating)
naive_rmse_v <- RMSE(validation$rating, mu)

movie_bias_x <- edx %>% 
  group_by(movieId) %>% 
  summarize(bias_m = mean(rating - mu_x))

pred_ratings_bm_v <- 
  left_join(validation, movie_bias, by='movieId') %>% 
  mutate(adj_rating_bm = mu_x + bias_m)
rmse_with_bias_m_v <- 
  RMSE(validation$rating, pred_ratings_bm_v$adj_rating_bm)

# Move and User Bias Model
user_movie_adj_x <- edx %>%
  left_join(movie_bias_x, by='movieId') %>%
  group_by(userId) %>% 
  mutate(bias_u = mean(rating - mu_x - bias_m)) %>% 
  distinct(userId,bias_u)
test_pred_both_v <- validation %>%
  left_join(movie_bias_x, by = "movieId") %>% 
  left_join(user_movie_adj_x, by = "userId") %>% 
  mutate(pred_rat_both = mu_x + bias_m + bias_u)

rmse_with_bias_both_v <- 
  RMSE(validation$rating, test_pred_both_v$pred_rat_both)

# Adjust for Regularization with lambda = 4.9
lambda <- 4.9
movie_bias_regl_x <- edx %>% 
  group_by(movieId) %>% 
  summarize(bias_m_regl_x = sum(rating - mu_x)/(n()+lambda))

user_bias_regl_x <- edx %>%
  left_join(movie_bias_regl_x, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(bias_u_regl_x = sum(rating - mu_x - bias_m_regl_x)/(n()+lambda))

ratings_regl_v <-
  validation %>% 
  left_join(movie_bias_regl_x, by = "movieId") %>% 
  left_join(user_bias_regl_x, by = "userId") %>% 
  mutate(pred_ratings_regl_v =mu_x+bias_m_regl_x+bias_u_regl_x) %>% 
  pull(pred_ratings_regl_v)
rmse_regl_v <- RMSE(validation$rating, ratings_regl_v)

high_low_v <- validation %>% 
  left_join(movie_bias_regl_x, by = "movieId") %>% 
  left_join(user_bias_regl_x, by = "userId") %>% 
  mutate(pred_ratings_regl_hilo =mu_x+bias_m_regl_x+bias_u_regl_x) %>%
  mutate(no_over=ifelse(pred_ratings_regl_hilo>5, 5, pred_ratings_regl_hilo)) %>% 
  mutate(no_over_under=ifelse(pred_ratings_regl_hilo<0.5,0.5, no_over))

rmse_no_over_under_v <- RMSE(validation$rating, high_low_v$no_over_under)
final_rmse <- rmse_no_over_under_v
final_rmse

# 11 Investigate Timestamp on Pulp and Gump ----------------
pulp_fict <- mod_train %>%
  filter(grepl('Pulp', title))
pulp_fict_t <- pulp_fict %>% select(rating, timestamp)
qplot(pulp_fict_t$rating, pulp_fict_t$timestamp)
pulp_fict_4.5 <- pulp_fict_t %>% 
  filter(rating == '4.5')
which.min(pulp_fict_4.5$timestamp)
pulp_fict_4.5[60]
pulp_early <- pulp_fict %>% filter(timestamp < 1045623054)
pulp_early_ratings <- pulp_early %>% 
  group_by(rating) %>% count()
pulp_early_ratings
head(pulp_early)
mean(pulp_early$rating)
mean(pulp_fict_t$rating)

pulp_fict_ratings <- pulp_fict_t %>% 
  group_by(rating) %>% count()
pulp_fict_ratings

gump <- mod_train %>%
  filter(grepl('Gump', title))
gump_t <- gump %>% select(rating, timestamp)
qplot(gump_t$rating, gump_t$timestamp)
gump_4.5 <- gump_t %>% 
  filter(rating == '4.5')
which.min(gump_4.5$timestamp)
gump_4.5[1085]
gump_ratings <- gump_t %>% 
  group_by(rating) %>% count()/27987*100
gump_ratings
percent_ratings <- gump_ratings/27987*100
qplot(percent_ratings$rating*279.87, percent_ratings$n)


                            
