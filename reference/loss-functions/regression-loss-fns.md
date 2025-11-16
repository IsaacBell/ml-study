# Measuring Error

## Mean Absolute Error

Straightforward eval of prediction accuracy. 

Tells you how far off your predictions are, w/o tracking directional distance (too high? too low?).

### Use Cases

- How closely do forecasts match actual sales?
- Healthcare: How close are predicted recovery times to real outcomes?

### Naive Code Example (w/ Numpy)

```python
import numpy as np

actual = np.array([100, 150, 200, 250])
predicted = np.array([110, 140, 210, 240])

mae = np.mean(np.abs(actual - predicted))
print("MAE:", mae) # => 10.0
```

### Code Example (SciKit Learn)

```python
from sklearn.metrics import mean_absolute_error

# actual values
y_true = [3, -0.5, 2, 7]

# predicted values
y_pred = [2.5, 0.0, 2, 8]

# Calculate MAE using scikit-learn
mae_value = mean_absolute_error(y_true, y_pred)
print(mae_value)
#0.5
```

## Mean Squared Error (MSE)

Like MAE, but squares each value before averaging. This makes it more sensitive to large errors.

Use it when big mistakes matter a lot.

Commonly used as the loss function for gradient descent-based training.

MSE is often preferred over RMSE for model training.

### Code Example (w/ SciKit Learn)
```python
from sklearn.metrics import mean_squared_error

# actual values
y_true = [1, 2, 3, 4, 5]

# predicted values
y_pred = [1.1, 2.2, 2.9, 4.1, 4.9]

# Calculate MSE using scikit-learn
mse_value = mean_squared_error(y_true, y_pred)
print(mse_value)
# 0.016
```

## Root Mean Squared Error (RMSE)

Basically the square root of the MSE.

Measures average magnitude of errors in a regression model. Measures prediction error in the same units as the response variable.

Penalizes large errors more heavily than MAE.

Note that the square root introduces non-linearity.

Often used in post-training predictions.

Common in regression analysis.


### Challenges

- Sensitive to outliers
- Not normalized: can't be used to compare across diff datasets and target scales by default

### Use Cases

- Comparing models on the same dataset
- Minimizing large errors
- When we need to interpret errors in natural units

### Code Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset = ...

X = dataset[['Temperature']]
y = dataset['Revenue']

# Model 1
model1 = LinearRegression()
model1.fit(X, y)
pred1 = model1.predict(X)
rmse1 = np.sqrt(mean_squared_error(y, pred1))
print(f"Model 1 RMSE: {rmse1:.3f}")

# Model 2 with an irrelevant predictor
# This is an example of a faulty model we're comparing results against
np.random.seed(0)
dataset['Noise'] = np.random.normal(0, 1, size=len(dataset))
X2 = dataset[['Temperature', 'Noise']]

model2 = LinearRegression()
model2.fit(X2, y)
pred2 = model2.predict(X2)
rmse2 = np.sqrt(mean_squared_error(y, pred2))
print(f"Model 2 RMSE: {rmse2:.3f}")
```

# R-Squared

### Use Cases

- Understanding how well a training model fits its dataset
- Simple model comparisons

### Challenges

- Overfitting
- Misleading when comparing models w/ diff predictors

# Adjusted R-Squared

### Use Cases

- Multiple regressions
- Preventing overfitting

### Challenges

- Doesn't indicate predictive performance on new data
- Low value doesn't neccessarily mean model performance is poor

# Huber Loss (a.k.a. Smooth MAE)

Scale-invariant, doesn't over-emphasize outlier values.




