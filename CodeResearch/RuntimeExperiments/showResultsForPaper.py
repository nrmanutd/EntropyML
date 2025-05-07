import os

import numpy as np

from CodeResearch.RuntimeExperiments.plotAndSaveRuntimeSamplesResults import plot_log_dependency, plot_dependency
from CodeResearch.Visualization.saveDataForVisualization import deserialize_labeles_list_of_arrays

cifarFeatures = 'SelectedLogs\\runtime_features_6_cifar.json'
cifarSamples = 'SelectedLogs\\runtime_samples_6_cifar.json'
mnistFeatures = 'SelectedLogs\\runtime_features_7_mnist.json'
mnistSamples = 'SelectedLogs\\runtime_samples_7_mnist.json'
coresFileName = 'SelectedLogs\\runtime_cores_4_mnist.json'

def filterArrays(arrays):
    result = []

    for i in range(len(arrays)):
        arr = np.array(arrays[i])
        m = np.median(arr)
        idx = np.where(arr > m)[0]
        result.append(arr[idx])

    return result

def plotFeatures(featuresFileName):
    data = deserialize_labeles_list_of_arrays(featuresFileName)
    arrays = data[0]
    labels = data[1]
    taskName = data[2].upper()

    arrays = filterArrays(arrays)

    plot_dependency(arrays, labels, taskName, r'$\log{k}$', lambda x: np.log(x), lambda x: np.log(x), lambda x: x, f'{os.path.splitext(featuresFileName)[0]}_processed.png')
    return

def plotSamples(samplesFileName):
    data = deserialize_labeles_list_of_arrays(samplesFileName)
    arrays = data[0]
    labels = data[1]
    taskName = data[2].upper()

    arrays = filterArrays(arrays)

    funcX1 = lambda x: np.log(x) + np.log(np.log(x))
    funcX2 = lambda x: np.log(x)

    plot_dependency(arrays, labels, taskName, r'$\log{\left[N \log{N}\right]}$', funcX1, lambda x: np.log(x),
                    lambda x: x[4:8], f'{os.path.splitext(samplesFileName)[0]}_loglogN_processed_v1.png')
    plot_dependency(arrays, labels, taskName, r'$\log{N}$', funcX2, lambda x: np.log(x), lambda x: x[4:8],
                    f'{os.path.splitext(samplesFileName)[0]}_logN_processed_v1.png')

    return

def plotCores(coresFileName):
    data = deserialize_labeles_list_of_arrays(coresFileName)
    arrays = data[0]
    labels = data[1]
    taskName = data[2]

    arrays = filterArrays(arrays)

    funcX1 = lambda x: np.log(x)
    plot_dependency(arrays, labels, taskName, r'$\log{Cores}$', funcX1, lambda x: np.log(x), lambda x: x[0:3],
                    f'{os.path.splitext(coresFileName)[0]}_processed.png')

    return

plotFeatures(cifarFeatures)
plotSamples(cifarSamples)

plotFeatures(mnistFeatures)
plotSamples(mnistSamples)