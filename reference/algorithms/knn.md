# K-Nearest Neighbors

Used for regression and classification. KNN predicts new data points by finding points in the existing training set. It's non-parametric and doesn't rely on a fixed form or set parameters.

There are different ways to calculate distance, but Euclidean distance is the most common.

Choosing a value for K is critical. For binary classification, choose an odd K to avoid tie votes.

Note that KNN uses neighbors to assigns values/labels, while [NNS](#nearest-neighbor-search-nns) is used to return the top K data points in the training set. NNS is used for search and reccommendation systems, for instance the "Movies You Might Like" feature on Netflix.

## Summary

- Supervised
- Input: Numerical data
- Output: A category or numerical value

Easy to use, with no standalone training phase. However, it is expensive and performs poorly when given poorly scaled features, noisy training data, or large datasets.

## Use Cases

- Classifying text,tabular, or image data
- NNS used for similarity search (finding similar data points)
  - movie reccommendations based on watch history
    - Shopping: similar products you might like
    - Similar locations

## Concepts

- Jr: can implement with a third-party library
- Mid-Level: tradeoffs, distance metrics, choosing optimal K
- Sr: approximate nearest neighbor search, computational complexity

## Algo

Need to choose a K first.

- Get distance between new data point and all other points in the training data
- Sort distances (ascending) and take the first K points
- For regressions, average the top K. For classification, take the majority vote.

### Raw Code Example

```python
import numpy as np

def euclidean_distance(x, y):
    return np.sqrt((x - y).T.dot(x - y))

def knn(X_train, y_train, X, k):
    """Binary classification"""
    distances = []
    for X_i in X_train:
        distance = euclidean_distance(X_i, X)
        distances.append(distance)

    top_k_indices = np.argsort(distances)[::-1][:k]
    top_k_labels = [y[idx].item() for idx in top_k_indices]
    if sum(top_k_labels) > k / 2:
        return 1
    else:
        return 0

def run_knn():
    N = 100
    D = 4
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, (N, 1))

    to_predict = np.random.randn(
        4,
    )
    predicted_label = knn(X, y, to_predict, k=3)
    print(predicted_label)
```

### SciKit Learn Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

K = 3

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling is very important for KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Interesting part
knn = KNeighborsClassifier(n_neighbors=K)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
```

## Choosing An Optimal K Value

Options:

- Treat K as a hyperparameter, and see which value performs best on training data.
- Use [cross-validation](https://www.kaggle.com/code/alexisbcook/cross-validation). Divide the training data into smaller subsets and run different experiments on each "fold".

Consider the [bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff). A too-small K value leads to overfitting. Larger K lowers model variance, but can also cause underfitting.

## Distance Metrics

- Euclidean Distance: for continuous variables of similar scale
- Manhattan Distance: for grid patterns, city blocks, and sometimes high-dimensional spaces
- Cosine Similarity: for text and certain high-dimensional data, where orientation matters instead of magnitude
- Hamming Distance: for binary data, categorical data, or discrete data of equal length
  - Can be used to count mismatches between data points (of equal length), e.g. error detection, comparing strings, or binary data

## Nearest Neighbor Search (NNS)

Search and reccommendation systems typically use vector embeddings and approximate NNS, to return data points (e.g. an Amazon product).

## Limitations

- Expensive and slow when given high dimensionality (large # features)
- Slow for large datasets
- Expensive in general (high memory consumption)
- Very sensitive to noise and outliers
- Requires good feature scaling
