import numpy as np
import faiss

# Generating random data
np.random.seed(42)
data = np.random.rand(10000, 128).astype('float32')

# Initializing Faiss index
index = faiss.IndexFlatL2(128)  # L2 distance index for 128-dimensional vectors

# Adding data to the index
index.add(data)

# Querying for nearest neighbors
query_vector = np.random.rand(1, 128).astype('float32')  # Random query vector
k = 5  # Number of nearest neighbors to find

distances, indices = index.search(query_vector, k)

print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)
