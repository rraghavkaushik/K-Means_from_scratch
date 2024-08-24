import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from kmeans import KMeans
import time
import matplotlib.pyplot as plt

def compare_kmeans_performance(data, n_clusters=3):
    custom_kmeans = KMeans(k=n_clusters)
    start_time = time.time()
    custom_labels = custom_kmeans.fit(data)
    custom_time = time.time() - start_time
    custom_centroids = custom_kmeans.centroids
    custom_inertia = custom_kmeans.calculate_inertia(data)

    sklearn_kmeans = SKLearnKMeans(n_clusters=n_clusters, random_state=42)
    start_time = time.time()
    sklearn_labels = sklearn_kmeans.fit_predict(data)
    sklearn_time = time.time() - start_time
    sklearn_centroids = sklearn_kmeans.cluster_centers_
    sklearn_inertia = sklearn_kmeans.inertia_

    print("Custom KMeans:")
    print(f"Time taken: {custom_time:.4f} seconds")
    print(f"Centroids:\n{custom_centroids}")
    print(f"Inertia: {custom_inertia:.4f}")
    print()

    print("scikit-learn KMeans:")
    print(f"Time taken: {sklearn_time:.4f} seconds")
    print(f"Centroids:\n{sklearn_centroids}")
    print(f"Inertia: {sklearn_inertia:.4f}")
    print()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=custom_labels, cmap='viridis', marker='o')
    plt.scatter(custom_centroids[:, 0], custom_centroids[:, 1], c='red', marker='x', s=100, linewidths=2)
    plt.title("Custom KMeans Clustering")

    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=sklearn_labels, cmap='viridis', marker='o')
    plt.scatter(sklearn_centroids[:, 0], sklearn_centroids[:, 1], c='red', marker='x', s=100, linewidths=2)
    plt.title("scikit-learn KMeans Clustering")

    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    random_points = np.random.randint(0, 100, (100, 2))

    compare_kmeans_performance(random_points, n_clusters=3)
