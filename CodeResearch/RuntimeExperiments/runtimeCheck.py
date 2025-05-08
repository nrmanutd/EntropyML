from sklearn import datasets

from CodeResearch.DiviserCalculation.diviserHelpers import prepareDataSet
from CodeResearch.RuntimeExperiments.runtimeTestCores import runtimeTestCores
from CodeResearch.RuntimeExperiments.runtimeTestFeatures import runtimeTestFeatures
from CodeResearch.RuntimeExperiments.runtimeTestSamples import runtimeTestSamples
from CodeResearch.dataSets import loadMnist, make_random, loadCifar


def runtimeTest(x, y, points, attempts, taskName):
    x = prepareDataSet(x)
    runtimeTestSamples(x, y, points, attempts, taskName)
    runtimeTestFeatures(x, y, points, attempts, taskName)
    runtimeTestCores(x, y, [1, 2, 4, 8, 16], attempts, taskName)

trainX, trainY = loadMnist()
runtimeTest(trainX, trainY, 8, 30, 'mnist')

trainX, trainY = loadCifar()
runtimeTest(trainX, trainY, 8, 30, 'cifar')





