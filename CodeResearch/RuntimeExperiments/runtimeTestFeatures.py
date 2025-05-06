import math

import numpy as np

from CodeResearch.RuntimeExperiments.plotAndSaveRuntimeSamplesResults import plotAndSaveRuntimeFeaturesResults
from CodeResearch.RuntimeExperiments.runtimeTest import runtimeTest
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays


def runtimeTestFeatures(x, y, points, attempts, taskName):
    nObjects = len(y)
    nFeatures = x.shape[1]
    powers = np.arange(1, points + 1) / points * math.log(nFeatures)
    possibleFeatures = np.exp(powers)

    results = []

    for i in range(len(possibleFeatures)):
        k = possibleFeatures[i]
        curLengthTimes = runtimeTest(x, y, nObjects, k, attempts)
        results.append(curLengthTimes)

        fileName = f'runtime_features_{i}_{taskName}.json'
        serialize_labeled_list_of_arrays(results, [str(ll) for ll in possibleFeatures[range(i + 1)]], taskName, attempts,
                                         fileName)
        plotAndSaveRuntimeFeaturesResults(fileName)

    return