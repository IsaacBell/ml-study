# Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

DBScan is similar to [K-Means Clustering](k-means-clustering.md).

Unlike K-Means Clutering, with DBSCAN the number of clusters doesn't need to be known in advance. DBSCAN is robust against noise and outliers, stays consistent across runs, and it handles clusters of different (non-circular) shapes.

DBSCAN uses two key parameters:

- Epsilon (ϵ): max radius of each cluster. The larger the ϵ value, the more likely nearby clusters are to be merged into one.
- MinPts: how many points are required to consider them a cluster?

## Visual Example

[Visualizing DBSCAN Clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

## Summary

- Unsupervised
- Input: Numerical data
- Output: Cluster labels for each new data point

Easy to use, with no standalone training phase. However, it is expensive and performs poorly when given poorly scaled features, noisy training data, or large datasets.

## Use Cases

(Similar/same as K-means clustering)

- Geospatial data
- Anomaly Detection [see paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025522005953))
- Image Compression & Processing (ex - [MRI image analysis](https://www.semanticscholar.org/paper/Segmentation-of-Brain-Tumour-from-MRI-image-%E2%80%93-of-Bandyopadhyay/a082abca6c53cc8d4f5fc80c7ad0fa83464cca48))
- Document Clustering
- Customer Segmentation
- Reccommendation Systems

## Algo

### SciKit Learn Example (Short)

```python
from sklearn.cluster import DBSCAN

X = ...  # training data

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

labels = dbscan.labels_
```

### SciKit Learn Example (Long)

See [DataCamp article](https://www.datacamp.com/tutorial/dbscan-clustering-algorithm)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Make a synthetic dataset
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Example of using the elbow method by plotting a k-distance graph
def plot_k_distance_graph(X, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances[:, k-1])
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('K-distance Graph')
    plt.show()

plot_k_distance_graph(X, k=5)

# Perform DBSCAN clustering
epsilon = 0.15  # Chosen based on k-distance graph
min_samples = 5  # 2 * num_features (2D data)
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(X)

# Visualize the results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

## Evaluation

Without labeled datasets, we can't use traditional metrics like accuracy and recall.

However, we can analyze data w/ similar techniques as in K-means clustering.

### Silhouette Analysis

(See [Wikipedia page](https://en.wikipedia.org/wiki/Silhouette_(clustering)))

Compares how similar each point is to its own cluster (cohesion) compared to other clusters (separation). Silhouette scores range from -1 to 1, where higher scores mean better-defined clusters (data points match their cluster well and don't match other clusters).

### The Elbow Method

(See [Wikipedia page](https://en.wikipedia.org/wiki/Elbow_method_(clustering)))

(The elbow of a curve is conceptually similar to the idea of [compressor knees](https://theproaudiofiles.com/compressor-knee/), found in audio engineering.)

## Distance Metrics

- Euclidean Distance: for continuous variables of similar scale
- Manhattan Distance: for grid patterns, city blocks, and sometimes high-dimensional spaces
- Cosine Similarity: for text and certain high-dimensional data, where orientation matters instead of magnitude
- Hamming Distance: for binary data, categorical data, or discrete data of equal length
  - Can be used to count mismatches between data points (of equal length), e.g. error detection, comparing strings, or binary data

## Limitations

- Less efficient for large datasets, especially with high-dimensional data
- Requires the right distance metric
