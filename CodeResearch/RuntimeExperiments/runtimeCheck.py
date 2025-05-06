from sklearn import datasets
import numpy as np

nSamples = 10000
nFeatures = 100
X_blobs, y_blobs = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=nFeatures, random_state=42)

runtimeTestSamples(X_blobs, y_blobs, 7, 20)
runtimeTestFeatures(X_blobs, y_blobs, 7, 20)
runtimeTestCores(X_blobs, y_blobs, [1, 2, 4, 8], 20)



