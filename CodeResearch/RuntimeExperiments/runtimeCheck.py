from sklearn import datasets

from CodeResearch.DiviserCalculation.diviserHelpers import prepareDataSet
from CodeResearch.RuntimeExperiments.runtimeTestCores import runtimeTestCores
from CodeResearch.RuntimeExperiments.runtimeTestFeatures import runtimeTestFeatures
from CodeResearch.RuntimeExperiments.runtimeTestSamples import runtimeTestSamples
from CodeResearch.dataSets import loadMnist, make_random, loadCifar


def runtimeTest(x, y, points, attempts, taskName):
    x = prepareDataSet(x)
    #runtimeTestSamples(x, y, points, attempts, taskName)
    runtimeTestFeatures(x, y, points, attempts, taskName)
    runtimeTestCores(x, y, [1, 2, 4, 8, 16], attempts, taskName)

nSamples = 10000
#nFeatures = 100
#X_blobs, y_blobs = datasets.make_blobs(n_samples=nSamples, centers=2, n_features=nFeatures, random_state=42)
#runtimeTest(X_blobs, y_blobs, 10, 100, 'blobs')

#trainX, trainY = loadMnist()
#runtimeTest(trainX, trainY, 8, 30, 'mnist')

#x, y = make_random(nSamples)
#runtimeTest(x, y, 8, 30, 'random')

trainX, trainY = loadCifar()
runtimeTest(trainX, trainY, 8, 30, 'cifar')





