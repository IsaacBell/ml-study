# Linear Regressions

Predicts a scalar value given a feature set by computing weights along with a bias value.

Its output is a linear combination of features (it looks like a line when graphed/plotted).

In smaller datasets, it can be directed computed with linear algebra. In large datasets, techniques like batching, mini-batching, and stochastic gradient descent (SGD) come into play.

## Use Cases

- financial forecasting
- risk assessment
- real estate (rental/home price predictions)

## Summary

- Supervised
- Input: a set of numerical features
- Output: a scalar representing the prediction

## Concepts

- Jr: R-squared, MAE, MSE, correlation of features
- Mid-Level: Loss function (L1, L2), Regularization (Lasso, Ridge)
- Sr: Variance inflation, BIC/AIC (model selection), contours of loss function, stability

### Definitions

- Regularization: adding a penalty to the loss function, to prevent overfitting and encourage simpler models
- L1 Regularization: uses the abs value of weight as the penalty term (loss gets bigger, weights become smaller)
- L2: drives the model params towards zero less dramatically than L1
- Coefficient magnitudes: reflect importance per feature (only in cases where features are all scaled)
- t-statistic: reflects the importance of a feature

## Loss Functions

see `regression-loss-functions.md`

- MAE: abs value of error
- Squared loss: square of the loss. punishes large errors & outliers
- Huber Loss: MAE + Squared Loss
- Ridge Loss: squared loss + regularization (penalty term)
- Lasso Loss: squared loss + penalty term that reduces less important params to be 0 (ignores them)

## Evaluation

### AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)

Used to adjust for param count and # of observations.

Useful when R-squared values are similar between 2 models. A lower AIC/BIC score means a better model.

## Limitations

- Doesn't inherently factor in causual relationships between features. May require feature engineering to capture features that represent those relationships (example - combining neighborhood location data with square footage).
- Limited usefulness in very complex data scenarios. Example - stock prices are affected by many non-linear factors like market sentiment, volatility, interplay with other stocks, and time-related factors like time of day and seasonality.

## Algo

```python
import numpy as np
from sklearn.linear_model import Ridge
import statsmodels.api as sm

X = ...
y = ...

# Create and fit Ridge regression model
model = Ridge(alpha=0.1)
model.fit(X, y)

# OR use StatsModels
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

## Exploratory Data Analysis (EDA)

- Look at the distribution of the target (e.g. home price values). Does it follow a normal distribution?
- Histogram and density plots of the target variable (price)
- Plot residual vs. fitted values, or residual vs. feature
    - Varying spread between the residuals across varying values of a feature will negatively impact the estimated coefficients, creating bias
    - Fixes: ridge loss, lasso loss, L1/L2 regularization
- BoxCox tests: @TODO
- Correlation plots to surface colinear feature vars (ex - Correlation Heatmap)

![Correlation Heatmap Example](https://i.imgur.com/04t9DYC.png)

We can compare R-squared scores while removing one correlated feature at a time.

-

