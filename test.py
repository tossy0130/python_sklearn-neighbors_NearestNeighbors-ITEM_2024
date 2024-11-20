import numpy as np
from sklearn.neighbors import NearestNeighbors
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]

print("テスト出力")
print(samples)

neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
neigh.fit(samples)