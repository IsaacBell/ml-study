import numpy as np

class Centroid:
    def __init__(self, location, vectors):
        self.location = location  # (D,)
        self.vectors = vectors    # (N_i, D)

class KMeans:
    def __init__(self, n_features, k):
        self.n_features = n_features
        self.centroids = [
            Centroid(
                location=np.random.randn(n_features),
                vectors=np.empty((0, n_features))
            )
            for _ in range(k)
        ]

    def distance(self, x, y):
        pass

    def fit(self, X, n_iterations):
        pass
