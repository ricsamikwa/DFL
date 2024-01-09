
import time
import numpy as np
import sys

sys.path.append('../')
import configurations
import functions

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


np.random.seed(1337)
X = np.random.randn(100, 10)
Y = np.random.randn(100, 10) + X

print(X)

cka_from_features = functions.model_similarity_cka(X, Y)
print('Linear CKA from Features: {:.5f}'.format(cka_from_features))

# clustering features 



# Configuration options
num_samples_total = 1000
cluster_centers = [(3,3), (7,7)]
num_classes = len(cluster_centers)
epsilon = 1.0
min_samples = 13

# Generate data
X, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.5)

np.save('./clusters.npy', X)
X = np.load('./clusters.npy')

# Compute DBSCAN
db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
labels = db.labels_

no_clusters = len(np.unique(labels) )
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

# Generate scatter plot for training data
colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
plt.title('Two clusters with data')
plt.xlabel('Axis X[0]')
plt.ylabel('Axis X[1]')
plt.show()


########################################################

# from sklearn.cluster import DBSCAN
# import numpy as np
print('Here we start DBSCAN')
# Assuming you have a similarity matrix as a numpy array
similarity_matrix = np.array([
    [1.0, 0.8, 0.3, 0.2, 0.6],
    [0.8, 1.0, 0.4, 0.1, 0.7],
    [0.3, 0.4, 1.0, 0.9, 0.2],
    [0.2, 0.1, 0.9, 1.0, 0.4],
    [0.6, 0.7, 0.2, 0.4, 1.0]
])

# Convert similarity scores to distances
distance_matrix = 1 - similarity_matrix

# Create DBSCAN object and fit the data
eps = 0.3  # Epsilon parameter to define the neighborhood
min_samples = 1  # Minimum number of samples in a neighborhood to form a cluster
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
dbscan.fit(distance_matrix)

# Retrieve the labels and core samples
labels = dbscan.labels_
core_samples = np.zeros_like(labels, dtype=bool)
core_samples[dbscan.core_sample_indices_] = True

# Number of clusters in labels, ignoring noise if present
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of clusters:", n_clusters)

# Print the clusters
for cluster_id in range(n_clusters):
    cluster_indices = np.where(labels == cluster_id)[0]
    print("Cluster", cluster_id)
    for index in cluster_indices:
        print("Object", index)
    print()


###########################################################################

print('Here we start K means')

# import numpy as np

# Assuming you have a similarity matrix as a numpy array
similarity_matrix = np.array([
    [1.0, 0.8, 0.3, 0.2, 0.6],
    [0.8, 1.0, 0.4, 0.1, 0.7],
    [0.3, 0.4, 1.0, 0.9, 0.2],
    [0.2, 0.1, 0.9, 1.0, 0.4],
    [0.6, 0.7, 0.2, 0.4, 1.0]
])

# Convert similarity scores to distances
distance_matrix = 1 - similarity_matrix

# Create KMeans object and fit the data
n_clusters = 3  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(distance_matrix)

# Get the cluster labels
labels = kmeans.labels_

# Print the clusters
for cluster_id in range(n_clusters):
    cluster_indices = np.where(labels == cluster_id)[0]
    print("Cluster", cluster_id)
    for index in cluster_indices:
        print("Object", index)
    print()

##################################################

# import numpy as np
# import matplotlib.pyplot as plt

# distance_matrix = np.array([[0, 2, 3, 4],
#                             [2, 0, 5, 6],
#                             [3, 5, 0, 7],
#                             [4, 6, 7, 0]])

# mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
# positions = mds.fit_transform(distance_matrix)

perplexity = min(30, len(distance_matrix) - 1) # Set perplexity value
tsne = TSNE(n_components=2, perplexity=perplexity, metric='precomputed', random_state=42, init='random')
positions = tsne.fit_transform(distance_matrix)



plt.scatter(positions[:, 0], positions[:, 1])
plt.title('Positions of Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



def generate_non_iid_data(alpha, num_devices, num_samples_per_device):
    """
    Generate non-IID data for federated learning using the Dirichlet distribution.
    
    Parameters:
        alpha (float): Alpha parameter indicating the non-IIDness level.
        num_devices (int): Number of devices (clients).
        num_samples_per_device (int): Number of samples to generate per device.
    
    Returns:
        list of lists: List of device data, each containing a list of generated class labels.
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 (exclusive) and 1 (inclusive).")
    
    class_probabilities = np.random.dirichlet([alpha] * 10)
    
    device_data = []
    for _ in range(num_devices):
        device_samples = np.random.choice(range(10), size=num_samples_per_device, p=class_probabilities)
        device_data.append(device_samples.tolist())
    
    return device_data

# Example usage
alpha = 1  # Non-IIDness level
num_devices = 4  # Number of devices (clients)
num_samples_per_device = 100  # Number of samples per device

non_iid_data = generate_non_iid_data(alpha, num_devices, num_samples_per_device)
for i, device_samples in enumerate(non_iid_data):
    print(f"Device {i+1}: {device_samples}")



