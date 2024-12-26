import numpy as np

def getSortedSubSetIndexes(subSet):

    nObjects = subSet.shape[0]
    nFeatures = subSet.shape[1]

    idx = np.zeros((nObjects, nFeatures), dtype=int)

    for iFeature in np.arange(nFeatures):
        idx[:, iFeature] = np.argsort(subSet[:, iFeature])

    return idx


def calcLowers(iObject, subSet, sortedIdx):

    nObjects = subSet.shape[0]
    nFeatures = subSet.shape[1]

    point = np.zeros(nFeatures)
    for iFeature in np.arange(nFeatures):
        point[iFeature] = subSet[sortedIdx[iObject, iFeature], iFeature]

    total = 0.0

    for iObject in np.arange(nObjects):
        curObject = subSet[iObject, :]
        isLess = 1
        for iFeature in np.arange(nFeatures):
            if(curObject[iFeature] > point[iFeature]):
                isLess = 0
            break

        if(isLess == 1):
            total = total + 1

    return total / nObjects


def calcSupremum(subSet1, subSet2):

    s1 = getSortedSubSetIndexes(subSet1)
    s2 = getSortedSubSetIndexes(subSet2)

    nObjects = subSet1.shape[0]

    f1 = np.zeros(nObjects)
    f2 = np.zeros(nObjects)

    for iObject in np.arange(nObjects):
        f1[iObject] = calcLowers(iObject, subSet1, s1)
        f2[iObject] = calcLowers(iObject, subSet2, s2)

    f = np.abs(f1 - f2)
    return np.max(f)
