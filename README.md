# An Investigation into Nutrition vs Yummy: How Does Nutritional Content Impact the Average Rating of a Recipe?

**Name(s)**: Mia Jerphagnon, Alyssa Tallada

**Website Link**: [Nutritional Analysis](https://miajerph.github.io/recipe-project/)


## Introduction

If you have ever been a college student, you know just how much unhealthy food we consume on a daily basis. Without the help from our parents, we struggle with eating healthy food. We crave quick and tasty meals in order to get through our busy schedules, sleepless nights, and hours of homework. But, does it have to be this way? What if we could eat healthy _and_ yummy food? 

Well, as part of UC San Diego's DSC 80 curriculum, this project explores __how the nutritional contect of a recipe (calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates) affects its average rating__. In Step 4, we specifically examine the relationship between protein and average rating. In Steps 5-8, we develop a multivariate predictive model to predict average rating based on nutritional content. 

We analyze two datasets from food.com. These datasets include recipes and ratings posted until and including 2008.

The first dataset is called `recipes`. It has 83,782 rows and 12 columns. 

| Column | Description |
|--------|-------------|
| 'name' | Recipe name |
| 'id' | Recipe ID |
| 'minutes' | Minutes to prepare recipe |
| 'contributor_id'| User ID who submitted this recipe |
| 'submitted'    | Date recipe was submitted |
| 'tags' | Food.com tags for recipe |
| 'nutrition' | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| 'n_steps' | Number of steps in recipe |
| 'steps' | Text for recipe steps, in order |
| 'description' | Description of the recipe |

The second dataset is called `interactions`. It has 73,1927 rows and 5 columns. 

| Column | Description |
|--------|-------------|
| 'user_id' | User ID |
| 'recipe_id' | Recipe ID |
| 'date' | Date of interaction |
| 'ratingt'| Rating given |
| 'review' | Review text |

## Data Cleaning and Exploratory Data Analysis

#### Data Cleaning

In order to clean the data, we implement the following steps:
1. Read the data sets
2. Left merge datasets on `id`
3. Fill ratings of 0 with NaN
4. Calculate average rating per recipe
5. Merge average ratings back to recipes dataset
6. Rename the column containing the average rating
7. Create new columns from the `nutrition` column
8. Drop the `nutrition` column
9. Create a new column called `fulfills_protein_DV` with Boolean values. If the value is True, then the protein PDV $\geq100$, and if the value is False, the protein PDV $<100$. This column aims to differentiate recipes that meet or do not meet the reccomended daily value of protein.
10. Reorder the columns so more relevant features are to the left

Ratings for recipes can only be between 1 and 5, so intuitively, ratings of 0 imply a user did not properly rate the recipe. 

Because the multiple features are stored as a string of a list in the `nutrition` column, we create new columns with these features. These columns, in addition to recipe `name`, `avg_rating`, and `fulfills_protein_DV` are integral and relevant to our analysis.

The full list of these columns are:

| Relevant Column | Description |
|--------|-------------|
| 'name' | Recipe name |
| 'avg_rating' | Average rating for recipe | 
| 'fulfills_protein_DV' | True/false whether protein content fulfills reccomended daily value intake |
| 'calories (#)' | Number of kilocalories | 
| 'total fat (PDV)' | Percent daily value of total fat |
| 'sugar (PDV)' | Percent daily value of sugar |
| 'sodium (PDV)' | Percent daily value of sodium | 
| 'protein (PDV)' | Percent daily value of protein | 
| 'saturated fat (PDV)' | Percent daily value of saturated fat |
| 'carbohydrates (PDV)' | Percent daily value of carbohydrates | 

Below is an excerpt of the first 5 rows of the cleaned `recipes_avg_ratings` dataset, except without showing `tags`, `description`, `steps`, and `ingredients` in order to preserve website readability.

|    | name                                 |   avg_rating | fulfills_protein_DV   |   calories (#) |   total fat (PDV) |   sugar (PDV) |   sodium (PDV) |   protein (PDV) |   saturated fat (PDV) |   carbohydrates (PDV) |     id |   minutes |   contributor_id | submitted   |   n_steps |   n_ingredients |
|---:|:-------------------------------------|-------------:|:----------------------|---------------:|------------------:|--------------:|---------------:|----------------:|----------------------:|----------------------:|-------:|----------:|-----------------:|:------------|----------:|----------------:|
|  0 | 1 brownies in the world    best ever |            4 | False                 |          138.4 |                10 |            50 |              3 |               3 |                    19 |                     6 | 333281 |        40 |           985201 | 2008-10-27  |        10 |               9 |
|  1 | 1 in canada chocolate chip cookies   |            5 | False                 |          595.1 |                46 |           211 |             22 |              13 |                    51 |                    26 | 453467 |        45 |          1848091 | 2011-04-11  |        12 |              11 |
|  2 | 412 broccoli casserole               |            5 | False                 |          194.8 |                20 |             6 |             32 |              22 |                    36 |                     3 | 306168 |        40 |            50969 | 2008-05-30  |         6 |               9 |
|  3 | millionaire pound cake               |            5 | False                 |          878.3 |                63 |           326 |             13 |              20 |                   123 |                    39 | 286009 |       120 |           461724 | 2008-02-12  |         7 |               7 |
|  4 | 2000 meatloaf                        |            5 | False                 |          267   |                30 |            12 |             12 |              29 |                    48 |                     2 | 475785 |        90 |          2202916 | 2012-03-06  |        17 |              13 |

#### Univariate Analysis

The distribution of protein in the dataset is skewed to the right. Most recipes are 200% of daily value or less. As the PDV of protein increases, the number of recipes decreases. Below is a plot of the distribution of sugar after removing outliers with sugar above 1,000 PDV. 

<iframe src="assets/fig_1.html" width="800" height="600" frameborder="0"></iframe>

The distribution of calories in the dataset is also skewed to the right. Most recipes have 1,500 calories or less. As the number of calories increases, so does the number of recipes. Below is a plot of the distribution of sugar after removing outliers with more than 5,000 calories. 

<iframe src="assets/fig_2.html" width="800" height="600" frameborder="0"></iframe>

#### Bivariate Analysis

The relationship between calories and average rating is quite weak. For instance, the correlation between the two variables is approximately 0. Recipes with higher calorie content are spread across all average ratings, and the same is true for recipes with lower calorie content. Looking at the scatter plot, knowing a recipe's calorie content does not tell you much about what its average rating would be.

<iframe src="assets/fig_3.html" width="800" height="600" frameborder="0"></iframe>

In our next bivariate analysis, we looked at the distribution of the average rating of the recipe conditioned on whether the protein content meets or does not meet the reccomended daily value. Based on the bar chart below, the proportion of non-protein-fulfilling recipes with an average rating of 5 is higher than the proportion of protein-fulfilling recipes with an average rating of 5. The same goes for recipes with an average rating between [1,2) and between [2,3). For average rating groups [3,4) and [4,5), the proportion of non-protein-fulfilling recipes is smaller than then the proportion of protein of protein-fulfilling recipes. 

However, the differences in proportions are quite small. As the chart displays, the proportions are __almost the same__ for each range of average rating. Thus, we cannot find any pattern that indicates how whether a recipe reaches a 100 protein PDV impacts its average rating. We need to do further analysis to see if we can perhaps find some pattern, or if truly there is no relationship between protein and rating. 

<iframe src="assets/fig_4.html" width="800" height="600" frameborder="0"></iframe>

## Interesting Aggregates

For this section, we wanted to get a broad sense of the distributions and averages of some relevant features. We created a small dataframe called summary to store the mean, standard deviation, min, max, and count of avg_rating, calories (#), total fat (PDV), sugar (PDV, sodium (PDV), and saturated fat (PDV).

We found that the standard deviation for calories and sugar had standard deviations greater than 200, so we must be careful later on and understand that these features might have high variance. Also, for all of the features, there was a non-neglible difference between the mean and median. This means that we must be wary of outliers when doing later analyses and when conducting the predictive model.

|        |   avg_rating |   calories (#) |   total fat (PDV) |   sugar (PDV) |   sodium (PDV) |   saturated fat (PDV) |
|:-------|-------------:|---------------:|------------------:|--------------:|---------------:|----------------------:|
| mean   |     4.62536  |        430.06  |           32.6254 |       68.6644 |        28.9417 |               40.2443 |
| median |     5        |        305.5   |           20      |       23      |        14      |               21      |
| std    |     0.640763 |        636.682 |           60.1488 |      247.239  |       144.975  |               80.9128 |
| min    |     1        |          0.1   |            0      |        0      |         0      |                0      |
| max    |     5        |      45609     |         3464      |    30260      |     29338      |             6875      |
| count  | 81173        |      83756     |        83782      |    83782      |     83782      |            83782      |

## Assessment of Missingness

Upon analyzing the data, we see that many of the columns describe characteristics of the recipes. For example, for each recipe, we can see the breakdown of the nutritonal content, the number of steps, ingredients, and a brief description of what exactly the recipe is. Based on the columns of the dataset, we see that the columns with many missing values is 'avg_rating.' Of all columns, 'avg_rating' is the only column that describes ordinal data; the missingness of average rating cannot be inferred based on the rest of the data because of the fundamentally different nature of what ratings represent. Objectively, ratings are more of an extrinsic measure, reflecting user preferences, user experience, and more, which are all factors that are not reflected within the dataset. Because of this, we can infer that patterns of missingness within 'avg_rating' are not missing at random (NMAR) are likely related to factors which exist outside the scope of our data. Some additional data that might help to explain the missingness of 'avg_rating' and points its missingness mechanism towards missing at random (MAR) include user feedback and trends, such as the number of users who reviewed the recipe or how popular the recipe is, as recipes reviewed by less people can easily have the average skewed in one direction.

### Missingness Dependency

In this section we will analyze the relationships between missing values within columns of our dataset and the missingness mechanisms that are found when analyzing column-to-column distributions.

#### Description and Number of Ingredients

#### Hypothesis
 
Null hypothesis: The missingness of description does not depend on the number of ingredients.

Alternative hypothesis: The missingness of description depends on the number of ingredients.

#### Test Statistic

Absolute difference of means

#### Significance Level

0.05

Here, we analyze the dependency between the missingness of `description` and the column `n_ingredients` using a permutation test to see if missing values of description are related to the number of ingredients. The test statistic used in this permutation test is the difference in group means of `n_ingredient` for recipes with descriptions against recipes without descriptions.

<iframe src="assets/diff_1.html" width="800" height="600" frameborder="0"></iframe>

With a p-value of 0.001, we reject the null hypothesis and infer that from performing a permutation
test and comparing the absolute difference of means for 'n_ingredients' for recipes with null descriptions against
recipes with valid descriptions, there exists a dependence of 'description' on 'n_ingredients', making
the missingness mechanism missing at random (MAR)

#### Description and Calories (#)

#### Hypothesis

Null hypothesis: The missingness of description does not depend on the number of calories

Alternative hypothesis: The missingness of description depends on the number of calories.

#### Test Statistic

Difference in means

#### Significance Level

0.05 

Here, we analyze the dependency between the missingness of `description` and the column `calories (#)` using a permutation test to see if missing values of description are related to a recipe's calories. The test statistic used in this permutation test is the difference in group means of `calories (#)` for recipes with descriptions against recipes without descriptions.

<iframe src="assets/diff_2.html" width="800" height="600" frameborder="0"></iframe>


Upon analyzing the relationship between the missingness of `description` and `calories (#)'`
and performing a permutation test by shuffling the description column, we see that the p-value, the probability of seeing diffs in group means of calories for recipes with descriptions vs. without descriptions is not statistically significant at 0.229. This suggests that the missingness of description based on calories is likely due to random chance, and points towards the missingness mechanism for description to be missing completely at random (MCAR).

## Hypothesis Testing

#### Research Question

Is there a difference between the ratings of protein-fulfilling and non-protein-fulfilling recipes? 

#### Hypothesis

There is no difference between the population mean rating of protein-fulfilling recipes (`protein (PDV)` $\geq 100$) and the population mean rating of non-protein-fulfilling recipes. 

There is a difference between the population mean rating of protein-fulfilling recipes (`protein (PDV)` $\geq 100$) and the population mean rating of non-protein-fulfilling recipes. 

#### Test Statistic

Absolute value of the difference between the mean rating for protein-fulfilling and non-protein fulfilling recipes

`{protein PDV >= 100} - \mu_{protein PDV < 100}`

#### Significance Level

`alpha=0.05`

To test our hypothesis, we run a permutation test to see if under the null (which is simulated through shuffling the `fulfills_protein_DV` column), whether the observed absolute mean difference is unlikely to occur under the null, ergo, is there statistically significant evidence in favor of the alternate hypothesis. The observed absolute mean difference between the two groups in the dataset is approximately 0.0031266. 

To run the test, we split the dataset into two groups, one where `fulfills_protein_DV` is true, and the other where it is false. Then, we shuffle the average ratings $n=1000$ times to find the mean differences of the two groups for the thousand simulations. 


Because our p-value of 0.739 is greater than the signifiance level, we fail to reject the null hypothesis. There is no statistially significant evidence to suggest that the absolute mean difference in average ratings between protein-fulfilling and non-protein-fulfilling recipes is difference in the population. Based on this permutation test, and previous bivariate analysis, it does not seem that people rate protein-heavy foods higher or lower than non-protein-heavy foods. 

## Framing a Prediction Problem

For our predictive model, we plan on predicting the average rating of a recipe using multivariate regression. Rather than predict the ordinal values of an individual recipe rating (1-5), we decided to predict a recipe's average rating, a continuous response variable that we believe is a better representation of the overall reception and popularity of a recipe. 

We hope to use variables related to nutritional content (e.g. sugar, calories) as predictors of average rating. We initially hoped to use protein as a predictor, but our previous analyses show that there might not be such a strong correlation between protein and average rating. Thus, we will include protein in the model, but might not weight it heavily compared to other variables in the regression. 

We will evaluate our model using the R2 score and root mean squared error. The R2 score shows the variance in average rating that is predictable from the predictor variables, and thus helps us understand how well our model's predictions match the data. The root mean squared error measures the average magnitude of the errors in our model's predictions and will help us understand the accuracy of our model. We will not use other scores such as F1 because they work better for classification.

At the time of prediction, we should have access to the nutritional content and all the other features in the rating dataset as described in the Introduction section. These features are related to the recipe and do not include data on users' opinions on the recipe.

## Baseline Model

For the baseline model, we want to look at `calories (#)`, `sugar (PDV)`, `carbohydrates (PDV)`, `protein (PDV)`.  These feautures are all continuous and quantitative. `fulfills_protein_DV` is a categorical variable that will be one-hot encoded. 

It would be interesting to try a logarithmic regression on the predictor variables. However, because there are many rows with `carbohydrates (PDV)` and `sugar (PDV)`,  equal to 0, logarithmic regression does not work well. So, those variables will be pre-processed as is (passthrough) for linear regression. 

Our model is thus a combination of linear regression model on carbohydrates and sugar, a logarithm regression on calories, and a one-hot encoding of fulfills protein. 

Based on the high root mean squared error (0.6403) and a $R^2$ score of almost 0 (0.0003), our baseline model performed extremely poorly. The low $R^2$ indicates that our predictors have low explanatory power. The high RMSE indicates a high error and inaccuracy. We must change the model completely going forward.

## Final Model

For the final model, we used `calories (#)`, `sugar (PDV)`, `carbohydrates (PDV)`, `sodium (PDV)`, `minutes` as the features. We decided not to use `fulfills_protein_DV` anymore because our EDA showed it had little effect, and it did not help in our baseline model.  We also changed the model from Linear Regression to a Decision Tree Regressor because the baseline model performed so poorly. We need a model with more complexity. 

`calories (#)`

This feature represents the number of calories in a recipe. To transform the calories feature, we decided to use a  RobustScaler in order to deal with outliers. 

`sugar (PDV)` and `carbohydrates (PDV)`

We decided to keep sugar and carbohydrates in the model, but because our baseline model performed so poorly, we thought logarithmic regression might not be a good fit. This time, we are transforming the two features using the Standard Scaler which might help when there are recipes that have an extreme amount of sugar or carbs. 

`sodium (PDV)`

We decided to add sodium because we hypothesize that recipes that are more salty might be more flavorful and appealing, leading to a higher average rating. Like with sugar, we are transforming this feature using Standard Scaler. 

`minutes`

This feature represents the cooking time of a recipe in minutes. We hypothesize that recipes that take longer to make have a lower average rating. It might be more frustrating and an overall worse experience to make a more complicated recipe. To transform this feature, we use a StandardScaler because some recipes take much longer than others. 

We used DecisionTreeRegressor for our modeling algorithm and used GridSearchCV to tune the hyperparameters of max_depth and max_features of the DecisionTreeRegressor. Decision trees are prone to high variance, and the hyperparameter max_depth helps control the variance and avoid overfitting. The best hyperparameters we found was 10 for the max_depth and 'sqrt' for max features.

The RMSE of the final model is 0.6319, which is a slight decrease from the RMSE of the baseline model. Additionally, the R2 score increased as well by approximately 0.02605. Our model still needs a lot of work, but it at least improved. 

## Fairness Analysis

For our fairness analysis, we want to split the recipes by sugar (PDV). Recipes with sugar (PDV) >= 23 (the median) will be designated as high in sugar, while recipes with sugar (PDV) < 23 will be designated as low in sugar. We used the median as the threshold instead of the mean because the median is less sensitive to outliers. 

We want to use the R2 score to analyze fairness to understand how the explanatory power of the sugar variable might change between the two groups. 

Null Hypothesis: Our model is fair. Its R2 for recipes with higher sugar PDV and lower sugar PDV are roughly the same, and any differences are due to random chance.

Alternative Hypothesis: Our model is unfair. Its R2 for recipes with higher sugar PDV is greater than its R2 for recipes with lower sugar PDV.

Test Statistic: Difference in R2 (high sugar PDV - low sugar PDV)

Significance Level: 0.05

Before running the test, the observed test statistic was approx. 0.0047. After running the 1000 permutation simulations, we got a p-value of 0.86. Because this is greater than the signifance level of 0.05, we fail to reject the null hypothesis that the model is fair. There is no statistically significant evidence of unfairness between high and low sugar groups.


