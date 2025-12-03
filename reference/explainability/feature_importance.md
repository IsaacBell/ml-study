# Feature Importance

What features bear the largest weight in predictions?

Feature importance is the measure of this.

## How To Measure It

There are many ways.

- Permutation Importance
-
-

## Permutation Importance

- Fast
- Straightforward
- Commonly used

PI is calculated after the model is fitted. It tells us WHAT features affect predictions. 

It asks the question: how would the model perform after randomly shuffling a single column of validation data?

For a feature that factors heavily into predictions, randomly re-ordering the column should reduce prediction accuracy.

PI is the technique of reshuffling each column, one by one, and measuring how much the loss function suffers per each column shuffling.

The scale of a feature doesn't affect PI (or at least shouldn't). The exception is if rescaling helps/hurts our model's handling the feature. In a tree-based model, like Random Forest, there won't be an effect. But there could be with e.g. Ridge Regression. 

A higher PI for a feature vs. another could indicate many things:

- Values in the dataset for that column are higher
- Values in the dataset for that column have more variance
- Practical considerations

If a feature has medium permutation importance, that could mean it has

- a large effect for a few predictions, but no effect in general, or
- a medium effect for all predictions.

### Code Example

```python
import eli5
from eli5.sklearn import PermutationImportance

feature_names = ["latitude", "longitude", ...]
model = ...

perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

Results are ordered from most to least important. To account for randomness, the same shuffling may be repeated multiple times.

## Partial Dependence Plots (aka Partial Plots)

PD plots tell us HOW a feature impacts predictions. They are calculated after model fitting.

They answer questions like:

- All things equal, how would similarly sized houses be priced in different areas? (different latitudes/longitudes)
- To what degree does diet difference impact patient health outcomes, ceterus paribus?

*Technique:* repeatedly alter the value of one variable's value, track the change in prediction outcome, and chart the results.

```python
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Note: you will only plot one feature
feature_names = ["Points Scored"]

disp1 = PartialDependenceDisplay.from_estimator(model, val_X, feature_names)
plt.show()
```

Plot multiple features:

```python
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

feature_names = [...]

for feat_name in feature_names:
    PartialDependenceDisplay.from_estimator(first_model, val_X, [feat_name])
    plt.show()
```

### 2D Partial Dependence Plots

Similar to previous PDP plot except we use tuple of features instead of single feature

```python
fig, ax = plt.subplots(figsize=(8, 6))
f_names = [('Goal Scored', 'Distance Covered (Kms)')]
disp4 = PartialDependenceDisplay.from_estimator(tree_model, val_X, f_names, ax=ax)
plt.show()
```


## SHAP Values (aka Shapley Values)

Break down how a model works in individual predictions.

Example Use Cases:

- FinTech: Explaining the basis behind every loan approval/rejection
- Healthcare: Identifying factors behind a patient's predicted risk rating for disease(s)  

### How It Works

We compare the impact of a specific value for a feature compared to a baseline value.

*Formula:* `sum(SHAP values for all features) = pred_for_team - pred_for_baseline_values`

![Example SHAP Values Graph](https://storage.googleapis.com/kaggle-media/learn/images/JVD2U7k.png)

Feature values causing increases in prediction value are colored pink, and sized by magnitude. Those decreasing the prediction value are colored blue. 

### Code Examples

```python
# ======================================================
# Make a prediction
# ======================================================

row_to_show = 1 # Show the first row of the dataset 
data_for_prediction = val_X.iloc[row_to_show]  # Using 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

model.predict_proba(data_for_prediction_array)

# ======================================================
# Get SHAP values for the prediction 
# ======================================================

import shap 

# Create object that can calculate shap values
# This example uses TreeExplainer
explainer = shap.TreeExplainer(model)

# Calculate Shap values - this gives back two arrays:
# 1. values  for the negative outcome (e.g., "didn't win the game")
# 2. values for the positive outcome (e.g., "did win the game")
shap_values = explainer.shap_values(data_for_prediction)

# Plot the visualization for the positive outcome prediction
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

```

### Other Explainers

- *shap.TreeExplainer*
- *shap.DeepExplainer* - deep learning models
- *shap.KernelExplainer* - works with all models (gives an approximated result)

*Kernel Explainer Example*

```python
k_explainer = shap.KernelExplainer(model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
```

## Summary Plots

Use them to get a birds-eye view of numeric feature importance and trends driving model predictions.

- Vertical location shows what feature it is depicting
- Color shows whether that feature was high or low for that row of the dataset
- Horizontal location shows whether the effect of that value caused a higher or lower prediction

The width of the range of dots for a feature doesn't directly indicate the importance of a feature, HOWEVER, if the dots are spread widely apart for that feature, it's a pretty good indication. Best to check permutation importance to verify.

Calculating the SHAP values can be slow with large datasets. The exception is with `xgboost` which is optimized for it.

![Summary Plot](https://i.imgur.com/NnOfiUW.png)

In this image, we see that:

- The model ignored the Red and Yellow & Red features.
- Usually Yellow Card doesn't affect the prediction, but there is an extreme case where a high value caused a much lower prediction.
- High values of Goal scored caused higher predictions, and low values caused low predictions

### Code Example

This example uses FIFA statistics to predict the probability of a player being declared "Man of the Match" (or "Player of the Game" for American readers).

*Model Setup*

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')

y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary

feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]

X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
```

*Summary Plot*

```python
import shap

explainer = shap.TreeExplainer(my_model)

# Calculate shap_values for all of val_X 
shap_values = explainer.shap_values(val_X)

shap.summary_plot(shap_values[1], val_X) # Index of [1] for positive outcome prediction
```

## SHAP Dependence Contribution Plots (SDC Plots)

Use in conjunction (or to replace) PDP plots, to show how a feature impacts predictions.

Each dot in an SDC Plots is a row of data. The X axis is the feature value, and Y is how it impacted prediction. 

The shape of the plot is important. If the plot shows a positive slope, then higher values for the feature lead to higher prediction values. 

From the following example, we might conclude that other features are interacting with Ball Possession % to influence the prediction value.

![Similar value with different impacts](https://i.imgur.com/DmM6DD6.png)

Outlier points also tell a story. In the following example, we could come to this conclusion: 

> In general, having the ball increases a team's chance of having their player win the award. But if they only score one goal, that trend reverses and the award judges may penalize them for having the ball so much if they score that little. 

![SDC Plot Outlier Example](https://i.imgur.com/x4SGC9m.png)

### Code Example

```python
import shap

feature_name = 'Ball Possession %' # the feature of interest
interaction_index = 'Goal Scored'  # the feature to compare against

explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(X)

# focus on one feature - the Y axis is chosen by the shap algo
shap.dependence_plot(feature_name, shap_values[1], X) 

# compare against another feature - we choose our own Y axis (the interaction index)
shap.dependence_plot(feature_name, shap_values[1], X, interaction_index=interaction_index)
```

Note: If you don't supply an argument for interaction_index, Shapley uses some logic to pick one that may be interesting.


## Other Code

### Compare PDP Plot for a feature to prediction values for each value of that feature

*Real-World Example* 

It seems like `time_in_hospital` doesn't matter as a data point. The difference between the lowest value on the partial dependence plot and the highest value is about 5%.

```python
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(my_model, val_X, ["time_in_hospital"])
plt.show()
```

Could  the data be wrong, or is the model doing something more complex?

Let's check the raw readmission rates (the prediction value, `y`) for each value of `time_in_hospital`.

```python
import pandas as pd
all = pd.concat([train_X, train_y], axis=1)
all.groupby("time_in_hospital").mean().readmitted.plot()
plt.show()
```

*Bonus*

The model overview looks reasonable. Let's make a function that visualises which features increase a patient's risk of readmission, which ones decrease it, and by how much.

```python
import shap

def patient_risk_factors(model, patient_data):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_vals[1], patient_data)

patient_record = val_X.iloc[0,:]
patient_risk_factors(my_model, patient_record)
```

### Tree Graph Visualization

@TODO - move to other reference file

```python
from sklearn import tree
import graphviz

feature_names = ["latitude", "longitude", ...]
model = ...

tree_graph = tree.export_graphviz(model, out_file=None, feature_names=feature_names)
graphviz.Source(tree_graph)
```

### Further Reading

- https://towardsdatascience.com/the-shapley-value-for-ml-models-f1100bff78d1/




























