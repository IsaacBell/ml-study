# K-Means Clustering

Group data points into clusters based on their similarity. The cluster centers (centroids) are adjusted until they stabilize into distinct cluster regions.

Has many uses, like organizing customer data for targeted marketing, or simplifying images to speed up processing.

Also see: [Centroids](https://en.wikipedia.org/wiki/Centroid).

## Summary

- Unsupervised
- Input: Numerical data
- Output: Cluster labels for each new data point

Easy to use. However, it is expensive and performs poorly when given poorly scaled features, noisy training data, or large datasets.

## Use Cases

- Anomaly Detection
- Image Compression
- Document Clustering
- Customer Segmentation

## Algo

Need to choose a K first.

- Get distance between new data point and all other points in the training data
- Sort distances (ascending) and take the first K points
- For regressions, average the top K. For classification, take the majority vote.

### SciPy Example

```python
from scipy.cluster.vq import kmeans, vq

centroids, labels = kmeans(data, K)

# Gives cluster assignments for each data point
labels, _ = vq(data, centroids)
```

### SciKit Learn Example

```python
from sklearn.cluster import KMeans
import numpy as np

k = 3
kmeans = KMeans(n_clusters=k)

kmeans.fit(data)

# Get cluster assignments for each data point
labels = kmeans.labels_

# Get cluster centroids
centroids = kmeans.cluster_centers_
```

## Choosing An Optimal K Value

Without labeled datasets, we can't use traditional metrics like accuracy and recall.

### The Elbow Method

(See [Wikipedia page](https://en.wikipedia.org/wiki/Elbow_method_(clustering)))

(The elbow of a curve is conceptually similar to the idea of [compressor knees](https://theproaudiofiles.com/compressor-knee/), found in audio engineering.)

1. Try out different K values, e.g. 1-10
2. For each K, calculate distortion (how spread out the clusters are). Lower distortion is usually better.
3. Plot the distortions for each value. X axis is K vals, Y axis is distortion.
4. As K increases (as we go right on the plot), distortion goes down.
5. The Elbow point is the inflection point on the plot where distortion stops decreasing so much as we increase K. The curve bends like an elbow (sometimes called the knee of a curve). This is your optimal K.

![Elbow Example](https://upload.wikimedia.org/wikipedia/commons/8/81/Elbow_in_Inertia_on_uniform_data.png)

### Silhouette Analysis

(See [Wikipedia page](https://en.wikipedia.org/wiki/Silhouette_(clustering)))

Compares how similar each point is to its own cluster (cohesion) compared to other clusters (separation). Silhouette scores range from -1 to 1, where higher scores mean better-defined clusters (data points match their cluster well and don't match other clusters).

## Limitations

- Only works with circular (or sometimes square) clusters. See [paper](https://arxiv.org/abs/2105.08348)
- Performs poorly with unbalanced datasets
- Variability: gives different outcomes based on initial seed, so results may vary
- Often need to know K in advance. i.e., even if the data naturally forms two clusters, a K of 3 will badly (forcefully) split the data into 3 poorly defined clusters.
- When the # of clusters needed is unknown, results will usually be poor
  - The Elbow method can help find a good K value
  - Otherwise, consider if [DBSCAN](dbscan.md) is a better alternative.

### Further Reading

@TODO

- K-means++ initialization
- Centroid selection
