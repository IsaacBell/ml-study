import numpy as np

def euclidean_distance(x, y):
  return np.sqrt((x - y).T.dot(x - y))

def knn(X_train, y_train, X_new, k):
  distances = []

  for X_i in X_train:
    distance = euclidean_distance(X_i, X_new)
    distances.append(distance)

  top_k_indices = np.argsort(distances)[:k]

  top_k_labels = [y_train[idx].item() for idx in top_k_indices]

  # Ensure floating-point division by explicitly converting `k` to a float
  if sum(top_k_labels) > (k / 2.0):
    return 1
  else:
    return 0

