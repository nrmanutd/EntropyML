import numpy as np
from scipy.stats import iqr

def getOptimalBinsCount(dataSet):
    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    return np.ones(nFeatures, dtype=int) * 20

    bins = np.arange(nFeatures)
    for i in np.arange(nFeatures):
        if nObjects > 1000:
            bins[i] = math.ceil(math.log(nObjects, 2) + 1)
        else:
            width = 2 * iqr(dataSet[:, i]) /(nObjects ** (1/3))
            bins[i] = math.ceil((max(dataSet[:, i]) - min(dataSet[:, i])) / width)
    return bins