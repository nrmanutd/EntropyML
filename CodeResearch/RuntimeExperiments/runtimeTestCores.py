import math

from CodeResearch.RuntimeExperiments.plotAndSaveRuntimeSamplesResults import plotAndSaveRuntimeFeaturesResults
from CodeResearch.RuntimeExperiments.runtimeTest import runtimeTest
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays
import numba
import numpy as np

def runtimeTestCores(x, y, cores, attempts, taskName):
    nObjects = len(y)
    nFeatures = x.shape[1]
    cores = np.array(cores)

    results = []

    for i in range(len(cores)):
        core = cores[i]
        numba.set_num_threads(core)

        curLengthTimes = runtimeTest(x, y, math.floor(nObjects/4), math.floor(nFeatures/4), attempts, i == 0)
        results.append(curLengthTimes)

        fileName = f'RuntimeLogs\\runtime_cores_{i}_{taskName}.json'
        serialize_labeled_list_of_arrays(results, [str(ll) for ll in cores[range(i + 1)]], taskName,
                                         attempts,
                                         fileName)
        plotAndSaveRuntimeFeaturesResults(fileName)