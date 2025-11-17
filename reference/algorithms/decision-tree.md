# Decision Trees (a.k.a DTrees)

Similar conceptually to the classic idea of binary trees, with each leaf node representing a label or an action to take.

Used in both classification and regression problems.

Fast, interpretable, explainable.

## Use Cases

- credit scoring
- customer segmentation
- fraud detection

## Summary

- Supervised
- Input: @TODO
- Output: EITHER, class label and probability of classification, OR scalar value prediction

## Concepts

- Jr: The concept itself, pros and cons
- Mid-Level: Ensembles (concept), score functions like MSE & gini impunity
- Sr: gini impunity vs. entropy, distributed/online learning, pseudocode, when to use

### Definitions

@TODO

-

## Loss Functions

- Cross-Entropy
-

## Evaluation

### Metrics

Accuracy is often not useful because there tends to be an uneven amount of one class/category.

Instead, precision, recall, and the F1-score tell how well the model classifies sparse positives.

i.e., in a spam detection system they measure how well a system classifies spam (which is relatively rare and often occupies less than 1% of emails in an inbox). A model could label every email as "not spam" and achieve 99% accuracy.

## Limitations

- Prone to overfitting and sensitive to noise. Techniques that help:
  - Tree Pruning (a term covering several concepts)
    - limiting the max tree depth
    - setting a min # of samples per leaf
- Brittle
  - Gradient-Boosted DTrees, Random Forests, or ensemble algos may work better in these cases.
- Computationally Expensive
  - Algos like Random Forests help w/ this

## Algo

@TODO: steps
-

### Regression Example

```python
from sklearn.tree import DecisionTreeRegressor

# model
clf = tree.DecisionTreeRegressor()

# dataset
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]

# learn
clf = clf.fit(X, y)

# predict
y_pred = clf.predict([[1, 1]])

# plot eval / test metrics, MSE etc.
```

### Classification Example

```python
from sklearn.tree import DecisionTreeClassifier

# create a new classifier
clf = DecisionTreeClassifier()

# labeled dataset
X = [[0, 0], [1, 1]]
y = [0, 1]

# learn
clf = clf.fit(X, y)

# predict for new data
y_pred = clf.predict([[-1,0]])

# plot metrics, AUC-ROC curve etc.
```

## Training

Requires the entire dataset to be loaded in memory. Can't train in batches.

Incremental updates or online learning aren't possible w/ a single DTree.

In practice, ensemble methods are often used to train several DTrees on subsets of training data. Each DTree in the ensemble is weighted based on the # of training examples it has seen.

During prediction, the output from different trees are averaged out.

For online learning and incremental updates, we can minibatch new data to a new DTree, then add it to our ensemble.

## Exploratory Data Analysis (EDA)

@TODO
