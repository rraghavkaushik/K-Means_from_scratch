import numpy as np
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    def distance(self, data, centroids):
        """Calculate Euclidean distance between a data point and all centroids"""
        dist = np.sqrt(np.sum((centroids - data) ** 2, axis=1))
        return dist

    def fit(self, data, max_iter=200):
        self.centroids = np.random.uniform(np.amin(data, axis=0), np.amax(data, axis=0),
                                           size=(self.k, data.shape[1]))

        for _ in range(max_iter):
            y = []
            for j in data:
                dist = self.distance(j, self.centroids)
                cluster = np.argmin(dist)
                y.append(cluster)

            y = np.array(y)
            centres = []

            for i in range(self.k):
                cluster_indices = np.argwhere(y == i).flatten()
                if len(cluster_indices) == 0:
                    centres.append(self.centroids[i])
                else:
                    centres.append(np.mean(data[cluster_indices], axis=0))

            centres = np.array(centres)

            if np.max(np.abs(self.centroids - centres)) < 0.0001:
                break
            else:
                self.centroids = centres

        return y

    def calculate_inertia(self, data):
        """Calculate the inertia for the current clustering"""
        total_inertia = 0
        for point in data:
            dist = self.distance(point, self.centroids)
            min_dist = np.min(dist)
            total_inertia += min_dist ** 2
        return total_inertia

def elbow_method(data, max_k=10):
    """Use the elbow method to find the optimal number of clusters"""

    inertias = []
    for k in range(1, max_k + 1):
        kmeans = Kmeans(k=k)
        kmeans.fit(data)
        inertia = kmeans.calculate_inertia(data)
        inertias.append(inertia)
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), inertias, 'bo-', markersize=8)
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K')
    plt.show()

    return inertias


random_points = np.random.randint(0, 100, (100, 2))

inertias = elbow_method(random_points, max_k=10)

"""
    Choose the optimum no of clusters from the elbow plot(here I have chosen 3 from my plot)
"""
optimal_k = 3 

kmeans = Kmeans(k=optimal_k)
labels = kmeans.fit(random_points)

plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='o')  
plt.show()

