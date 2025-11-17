# Logistic Regressions

Categorizes data points based on a linear combination of inputs.

## Use Cases

Binary and multiclass classifications.

- credit scoring
- diagnoses
- spam detection

## Summary

- Supervised
- Input: a set of numerical features
- Outpout: a scalar representing the prediction

## Concepts

- Jr: Gradient descent, accuracy, precision, recall
- Mid-Level: ROC curves, hyperparameters (mini-batch size, learning rate)
- Sr: Class imbalance, L1/L2 regularization, @TODO

### Definitions

- Regularization: adding a penalty to the loss function, to prevent overfitting and encourage simpler models
- L1 Regularization: uses the abs value of weight as the penalty term (loss gets bigger, weights become smaller)
- L2: drives the model params towards zero less dramatically than L1
- Coefficient magnitudes: reflect importance per feature (only in cases where features are all scaled)
- t-statistic: reflects the importance of a feature

## Loss Functions

See `classification-loss-functions.md`

- Cross-Entropy

## Evaluation

### Metrics

Accuracy is often not useful because there tends to be an uneven amount of one class/category.

Instead, precision, recall, and the F1-score tell how well the model classifies sparse positives.

i.e., in a spam detection system they measure how well a system classifies spam (which is relatively rare and often occupies less than 1% of emails in an inbox). A model could label every email as "not spam" and achieve 99% accuracy.

## Limitations

- Prone to overfitting, which is common when using cross-entropy. L1/L2 regularization helps (see [Linear Regressions](linear-regression.md)).
- Sensitive to training data imbalances. Techniques that help:
  - SMOTE
  - Boosting/bagging (also see: Decision Trees, Random Forests)
- Assume linearity between inputs and (logs odds of) predictions, with predictable data relationships and no multicolinearity. Gives biased/faulty predicts when this doesn't hold up.
  - Decision Trees, Random Forests, or ensemble algos may work better in these cases.

## Algo

1. Initialize weights and bias term
2. Minimize cross-entropy loss on the training set (using multiple runs of gradient descent via backwards passes)
3. Stop training on a set number of epochs, convergence of loss, etc.
4. Add new data points to the forward pass and use the activation function (sigmoid=binary, softmax=multiclass)
5. Model is ready to predict new info

Numpy Example:

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def calculate_gradients(X, y, W, b):
    N = len(X)
    y_pred = predict(X, W, b)
    weights_grad = X.T.dot(y_pred - y) / N
    bias_grad = np.mean(y_pred - y)
    return weights_grad, bias_grad

def binary_cross_entropy(y_pred, y):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def predict(X, W, b):
    z = X.dot(W) + b
    return sigmoid(z)

def train(X, y, W, b, learning_rate, batch_size, n_epochs):
    # too long for this example
    # # Iterate over the specified number of epochs
    # # Iterate over the training data in batches

    # # In each iteration:
    # # # Extract the input features & ground truth labels for the current batch
    # # # Calculate the gradients of the loss function
    # # # Update the weights and bias using gradient descent

    # # Once all batches in the current epoch are processed:
    # # # Make predictions on the entire dataset
    # # # Calculate the loss function (binary cross-entropy)

    pass
```

### SciKit Learn Example

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Binary classification (0 or 1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Evaluations
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
```

# Mini-batch stochastic gradient descent (minibatch SGD)

- Smaller batches fit in GPU/CPU memory better
- Small batches help with regularization
- Models can converge faster

## Exploratory Data Analysis (EDA)

@TODO
