# Linear SVMs (support vector machines)

Classifies data by creating a maximal margin hyperplane to separate the classes. This is a line in 2D or a plane in 3D.

It assumes that the classes are separable in 2D/3D space. The closest data points to the separator/boundary plane are called support vectors, and the distance between them are called the margin. The larger the margin, the more robust the linear SVM.

Linear SVMs are fast and useful for high-dimensional data classification.

@TODO: add image examples

## Use Cases

- text classification (after pre-processing e.g. BOW, CBOW, or embedding)
- OCR and handwriting recognition
- Document classification

## Summary

- Supervised
- Input: Numerical features (which can also be continuous or represent categorization)
- Output: a class label

## Concepts

- Jr: The concept itself, pros and cons
- Mid-Level: Soft-margin and hard-margin classification, handling data that isn't linearly separable
- Sr: Kernels, comparison vs. logistic regressions, primal & dual forms in objective function

### Definitions

- C parameter: balances between maximizing the margin and minimizing the classification errors.
  - Larger C values lead to stricter classifications and narrower margins
- Primal formulation optimizes in the space of weight vectors and biases
  - Better for large datasets
- Dual formulation: optimizes Lagrange multipliers
  - Better when we have non-linearity and need to use kernel methods
- Concept drift: when the data distribution changes over time.
  - Typically requires regular retraining or online learning.

## SVM vs. logistic regression

- SVMs are less susceptible to outliers - support vectors tend to ignore data points far from the margin. Logistic regressions consider all data points in their optimization.
- SVMs are more sample-efficient after training (when support vectors are established). New outlier data samples have little effect on the margin.
- SVMs handle [multicolinearity](https://en.wikipedia.org/wiki/Multicollinearity) better.
- Logistic regressions output predictions as probabilities, unlike SVMs.
- Logistic regression decisions are more interpretable.
- Both models can suffer from [concept drift](#definitions)

## Evaluation

### Offline Eval

- precision
- recall
- ROC curves
- F1 scores.

### Online Eval

- Online SGD is possible, but concept drift is a constant issue to address
  - Choose a learning rate for weight adjustment, and balance between stability and adaptability
- Extensive A/B testing is often needed between deployed versions

## Limitations

- [See comparison w/ logistic regression](#svm-vs-logistic-regression)
- Online eval isn't necessary straightforward

## Algo

@TODO: steps

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Create a small 2D dataset
X = np.array([
   [1, 2], [2, 3], [3, 3], [2, 1], [3, 2],   # Class 1
   [6, 6], [7, 8], [8, 7], [7, 6], [8, 6],   # Class 2
   [6, 5], [4, 4],     # Points within the margin or wrong side
])
y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1])

# Train a Linear SVM with soft margin
svc = SVC(kernel='linear', C=0.5)
svc.fit(X, y)

# Extract weight vector (w) and bias (b)
w = svc.coef_[0]
b = svc.intercept_[0]

# Calculate decision function values for all points
y_decision = svc.decision_function(X)

# Identify margin violations (points strictly within the margin)
margin_violations = np.abs(y_decision) < 1 - 1e-10

# Create new test points
X_test = np.array([[5, 4.5], [5, 5.5], [4.5, 5], [5.5, 5]])
y_test_pred = svc.predict(X_test)
y_test_decision = svc.decision_function(X_test)

# Print test point classifications
for i, (x, y_pred, y_dec) in enumerate(zip(X_test, y_test_pred, y_test_decision)):
   print(f"Test point {i+1} at {x}: Predicted class {y_pred}, Decision value {y_dec:.4f}")
```

## Exploratory Data Analysis (EDA)

@TODO
