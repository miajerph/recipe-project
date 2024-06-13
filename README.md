# An Investigation into Nutrition vs Yummy: How Does Nutritional Content Impact the Average Rating of a Recipe?

**Name(s)**: Mia Jerphagnon, Alyssa

**Website Link**: [Nutritional Analysis](https://miajerph.github.io/recipe-project/)


## Step 1: Introduction

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

## Step 2: Data Cleaning and Exploratory Data Analysis

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


```py
recipes_avg_ratings.head()
```
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



## Assessment of Missingness

Upon analyzing the data, we see that many of the columns describe characteristics of the recipes. For example, for each recipe, we can see the breakdown of the nutritonal content, the number of steps, ingredients, and a brief description of what exactly the recipe is. Based on the columns of the dataset, we see that the columns with many missing values is 'avg_rating.' Of all columns, 'avg_rating' is the only column that describes ordinal data; the missingness of average rating cannot be inferred based on the rest of the data because of the fundamentally different nature of what ratings represent. Objectively, ratings are more of an extrinsic measure, reflecting user preferences, user experience, and more, which are all factors that are not reflected within the dataset. Because of this, we can infer that patterns of missingness within 'avg_rating' are not missing at random (NMAR) are likely related to factors which exist outside the scope of our data. Some additional data that might help to explain the missingness of 'avg_rating' and points its missingness mechanism towards missing at random (MAR) include user feedback and trends, such as the number of users who reviewed the recipe or how popular the recipe is, as recipes reviewed by less people can easily have the average skewed in one direction.

### Missingness Dependency

In this section we will analyze the relationships between missing values within columns of our dataset and the missingness mechanisms that are found when analyzing column-to-column distributions.

#### Description and Number of Ingredients
 
Null hypothesis: The missingness of description does not depend on the number of ingredients.

Alternative hypothesis: The missingness of description depends on the number of ingredients.

Here, we analyze the dependency between the missingness of `description` and the column `n_ingredients` using a permutation test to see if missing values of description are related to the number of ingredients. The test statistic used in this permutation test is the difference in group means of `n_ingredient` for recipes with descriptions against recipes without descriptions.

<iframe src="assets/diff_1.html" width="800" height="600" frameborder="0"></iframe>

With a p-value of 0.001, we reject the null hypothesis and infer that from performing a permutation
test and comparing the abs diff of means for 'n_ingredients' for recipes with null descriptions against
recipes with valid descriptions, there exists a dependence of 'description' on 'n_ingredients', making
the missingness mechanism missing at random (MAR)

### Description and Calories (#)

Null hypothesis: The missingness of description does not depend on the number of calories

Alternative hypothesis: The missingness of description depends on the number of calories.

Here, we analyze the dependency between the missingness of `description` and the column `calories (#)` using a permutation test to see if missing values of description are related to a recipe's calories. The test statistic used in this permutation test is the difference in group means of `calories (#)` for recipes with descriptions against recipes without descriptions.

<iframe src="assets/diff_2.html" width="800" height="600" frameborder="0"></iframe>


Upon analyzing the relationship between the missingness of `description` and `calories (#)'`
and performing a permutation test by shuffling the description column, we see that the p-value, the probability of seeing diffs in group means of calories for recipes with descriptions vs. without descriptions is not statistically significant at 0.229. This suggests that the missingness of description based on calories is likely due to random chance, and points towards the missingness mechanism for description to be missing completely at random (MCAR).

## Hypothesis Testing

#### Research Question

Is there a difference between the ratings of protein-fulfilling and non-protein-fulfilling recipes? 
#### Hypothesis

$H_0$: There is no difference between the population mean rating of protein-fulfilling recipes (`protein (PDV)` $\geq 100$) and the population mean rating of non-protein-fulfilling recipes. 

$H_a$: There is a difference between the population mean rating of protein-fulfilling recipes (`protein (PDV)` $\geq 100$) and the population mean rating of non-protein-fulfilling recipes. 

#### Test Statistic

Absolute value of the difference between the mean rating for protein-fulfilling and non-protein fulfilling recipes

$|\mu_{protein PDV >= 100} - \mu_{protein PDV < 100}|$

#### Significance Level

$\alpha=0.05$ 

To test our hypothesis, we run a permutation test to see if under the null (which is simulated through shuffling the `fulfills_protein_DV` column), whether the observed absolute mean difference is unlikely to occur under the null, ergo, is there statistically significant evidence in favor of the alternate hypothesis. The observed absolute mean difference between the two groups in the dataset is approximately 0.0031266. 

To run the test, we split the dataset into two groups, one where `fulfills_protein_DV` is true, and the other where it is false. Then, we shuffle the average ratings $n=1000$ times to find the mean differences of the two groups for the thousand simulations. 


Because our p-value of 0.739 is greater than the signifiance level, we fail to reject the null hypothesis. There is no statistially significant evidence to suggest that the absolute mean difference in average ratings between protein-fulfilling and non-protein-fulfilling recipes is difference in the population. Based on this permutation test, and previous bivariate analysis, it does not seem that people rate protein-heavy foods higher or lower than non-protein-heavy foods. 

## Framing a Prediction Problem

For our predictive model, we plan on predicting the average rating of a recipe using multivariate regression. Rather than predict the ordinal values of an individual recipe rating (1-5), we decided to predict a recipe's average rating, a continuous response variable that we believe is a better representation of the overall reception and popularity of a recipe. 

We hope to use variables related to nutritional content (e.g. sugar, calories) as predictors of average rating. We initially hoped to use protein as a predictor, but our previous analyses show that there might not be such a strong correlation between protein and average rating. Thus, we will include protein in the model, but might not weight it heavily compared to other variables in the regression. 

We will evaluate our model using the $R^2$ score and root mean squared error. The $R^2$ score shows the variance in average rating that is predictable from the predictor variables, and thus helps us understand how well our model's predictions match the data. The root mean squared error measures the average magnitude of the errors in our model's predictions and will help us understand the accuracy of our model. We will not use other scores such as F1 because they work better for classification.

At the time of prediction, we should have access to the nutritional content and all the other features in the rating dataset as described in the Introduction section. These features are related to the recipe and do not include data on users' opinions on the recipe.

## Baseline Model

For the baseline model, we want to look at `calories (#)`, `sugar (PDV)`,  and `carbohydrates (PDV)`.  These feautures are all continuous and quantitative. 

<iframe src="assets/fig_5.html" width="800" height="600" frameborder="0"></iframe>

<iframe src="assets/fig_6.html" width="800" height="600" frameborder="0"></iframe>
