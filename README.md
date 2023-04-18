The main problem statement of our project is how can we estimate a future movie’s gross revenue, profit and budget? We aim to solve this problem using linear regression.

Our code is divided into 3 parts. In the first part, we store the original movies data in a pandas dataframe data, then we appended a new column “profit” to data, obtained by taking “gross” - “budget”. Then we cleaned data by removing NaN values and removing unnecessary columns to obtain cleanoridata. To further clean it, for each categorical variable, we grouped “gross” by each level of the categorical variable and calculated the mean “gross” for each level, then we assigned this to become the numerical value of the level, thereby obtaining final cleandata where all variables are numerical.

For exploratory data analysis, we use cleandata to calculate the correlation matrix and visualise it as a heatmap. We also drew boxplots for categorical variables (limited to a few levels due to space constraints) to observe the general relationship between them and our dependent variables “gross” and “profit”.

The second part focuses on the linear regression. We split cleandata into 20% testing data set and 80% training data set. lm is a list of 9 linear model objects all fitted to our training data sets, and lmdescript is a list of strings whose ith element is the description for lm[i]. For Models 0 to 2, the dependent variable is profit. For Model 0, we use all reasonable independent variables, while for Model 1 we use only numerical ones and for Model 2 we use only categorical ones. Models 3, 4, 5 are exactly the same as Models 0, 1, 2 respectively except that their dependent variable is gross revenue instead of profit. Models 6,7,8 are similar to Models 0,1,2 respectively, except that their dependent variable is budget, and profit was included while budget was excluded from their independent variables. Then for each linear model, we print its description, its summary of results, its p-values, its coefficients, its R-squared value and its mean squared error on the testing data set.

In the last part, for each of our linear models, for both training and testing data sets, we used scatter plots to visualise a graph of predicted response against the true response, together with the control line y = x to help us better visually estimate each model’s general prediction accuracy.

Contributions:

Jace - conversion of categorical variables into numerical ones, fitting of linear models, displaying results of linear models, visualising prediction accuracy of linear models, results analysis, derivation of data-driven insights and recommendations

Ye Wint - Edited and made the videos and slides. Did the heat map correlation matrix and find data sets. Come up with the idea for logistic regression for things we implement beyond the course syllabus. Did the practical motivations and conclusion of outcome and interesting findings.

Joel - removing of null values in dataset, data exploration and visualisation of dataset, implementation of logistic regression

References:

Walker, S. H., & Duncan, D. R. (1967). Estimation of the Probability of an Event as a Function of Several Independent Variables. Biometrika, 54(1/2), 167. https://doi.org/10.2307/2333860
