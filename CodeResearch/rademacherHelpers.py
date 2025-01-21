import numpy as np

def GetSortedData(subSet):
    nObjects = subSet.shape[0]
    nFeatures = subSet.shape[1]

    sortedSetIdx = np.zeros((nObjects, nFeatures), dtype=int)
    sortedSet = np.zeros((nObjects, nFeatures))

    for iFeature in np.arange(nFeatures):
        sIdx = np.argsort(subSet[:, iFeature], stable=True)
        sortedSetIdx[:, iFeature] = sIdx
        sortedSet[:, iFeature] = subSet[sIdx, iFeature]

    return sortedSetIdx, sortedSet

def GetSortedDataLists(subSet):
    nObjects = subSet.shape[0]
    nFeatures = subSet.shape[1]

    sortedSetIdx = []
    sortedSet = []
    mapOrigToSorted = []

    for iFeature in range(0, nFeatures):
        sIdx = np.argsort(subSet[:, iFeature], stable=True)
        sortedSetIdx.append(sIdx)
        sortedSet.append(subSet[sIdx, iFeature])

        map = np.zeros(nObjects)
        for iIdx in range(0, nObjects):
            map[sIdx[iIdx]] = iIdx

        mapOrigToSorted.append(map)

    return sortedSetIdx, sortedSet, mapOrigToSorted


def ConvertVector(v, p1, p2):

    res = np.zeros(len(v))
    res[0] = v[0] * p1
    idx = np.arange(1, len(v))
    res[idx] = v[idx] * p2

    return res

def GetObjectsPerClass(target, seekingClass, nObjects):
    idx = np.where(target == seekingClass)[0]

    mask = np.zeros(len(idx))
    mask[np.arange(nObjects)] = 1

    mask = np.random.permutation(mask)
    idxM = np.where(mask > 0)[0]

    return idx[idxM].tolist()


def GetSubSet(dataSet, target, nObjects):
    vClasses, parts = np.unique(target, return_counts=True)
    parts = parts / len(target)

    nParts = np.floor(nObjects * parts).astype(int)

    objectsPerClass = 2 * np.maximum(np.ones(len(nParts), dtype=int), nParts)

    subSetIdx = []

    for iClass in np.arange(len(vClasses)):
        idx = GetObjectsPerClass(target, vClasses[iClass], objectsPerClass[iClass])
        subSetIdx = subSetIdx + idx

    return dataSet[subSetIdx], target[subSetIdx]