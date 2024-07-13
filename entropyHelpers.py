import numpy as np
from scipy.stats import entropy

def calcConditionalEntropy(dataSet, buckets, classIdx):
    nObjects = dataSet.shape[0]
    nClasses = len(classIdx)
    nFeatures = dataSet.shape[1]

    conditionalEntropy = 0.0

    for iFeature in np.arange(nFeatures):
        for jClass in np.arange(nClasses):
            idx = np.digitize(dataSet[classIdx[jClass], iFeature], buckets[iFeature])
            v, c = np.unique(idx, return_counts=True)
            ent = entropy(c / sum(c))
            p = sum(c) / nObjects
            conditionalEntropy += p * ent

    return conditionalEntropy
def calcConditionalMultiVarianceEntropy(dataSet, objectBins, classIdx):
    nObjects = dataSet.shape[0]
    nClasses = len(classIdx)

    conditionalMultiEntropy = np.zeros(nClasses, dtype=float)

    for iClass in np.arange(nClasses):
        mve = calcMultiVarianceEntropy(dataSet[classIdx[iClass], :], objectBins)
        conditionalMultiEntropy[iClass] = len(classIdx[iClass])/nObjects * mve

    return sum(conditionalMultiEntropy)

def calcMultiVarianceEntropy(dataSet, objectBins):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    objectBinsIndexes = np.zeros((nObjects, nFeatures))

    for iFeature in np.arange(nFeatures):
        objectBinsIndexes[:, iFeature] = np.digitize(dataSet[:, iFeature], objectBins[iFeature])

    bucketIdx = [''] * nObjects

    for iObject in np.arange(nObjects):
        bucketIdx[iObject] = ''.join(str(x) for x in objectBinsIndexes[iObject, :])

    eBins, cBins = np.unique(bucketIdx, return_counts=True)
    ent = entropy(cBins/sum(cBins))

    return ent