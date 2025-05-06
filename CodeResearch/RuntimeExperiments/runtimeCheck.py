from sklearn import datasets

from CodeResearch.RuntimeExperiments.runtimeTestCores import runtimeTestCores
from CodeResearch.RuntimeExperiments.runtimeTestFeatures import runtimeTestFeatures
from CodeResearch.RuntimeExperiments.runtimeTestSamples import runtimeTestSamples

nSamples = 10000
nFeatures = 10
X_blobs, y_blobs = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=nFeatures, random_state=42)

runtimeTestSamples(X_blobs, y_blobs, 7, 20, 'blobs')
runtimeTestFeatures(X_blobs, y_blobs, 3, 20, 'blobs')
runtimeTestCores(X_blobs, y_blobs, [1, 2, 4, 8], 20, 'blobs')



