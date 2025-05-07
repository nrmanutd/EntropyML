import math

import numpy as np

from CodeResearch.RuntimeExperiments.plotAndSaveRuntimeSamplesResults import plotAndSaveRuntimeSamplesResults
from CodeResearch.RuntimeExperiments.runtimeTest import runtimeTest
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays

def runtimeTestSamples(x, y, points, attempts, taskName):

    nObjects = len(y)
    nFeatures = x.shape[1]
    powers = np.arange(1, points + 1) / points * math.log(nObjects)
    possibleLength = np.floor(np.exp(powers)).astype(int)

    print(f'Sample runtime test for {taskName}')

    results = []
    for i in range(len(possibleLength)):
        l = possibleLength[i]
        curLengthTimes = runtimeTest(x, y, l, nFeatures, attempts, i == 0)
        results.append(curLengthTimes)

        fileName = f'RuntimeLogs\\runtime_samples_{i}_{taskName}.json'
        serialize_labeled_list_of_arrays(results, [str(ll) for ll in possibleLength[range(i + 1)]], taskName, attempts, fileName)

        plotAndSaveRuntimeSamplesResults(fileName)

    return