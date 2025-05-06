import math

import numpy as np

def extractDataSet(x, y, nObjects, nFeatures):
    x, y = GetSubSet(x, y, math.floor(nObjects / 2))
    x = GetSubSetOnFeatures(x, nFeatures)

    return x.copy(), y.copy()

def permuteDataSet(newSet, newTarget):
    idx = range(0, len(newTarget))
    newIdx = np.random.permutation(idx)

    return newSet[newIdx], newTarget

def GetObjectsPerClass(target, seekingClass, nObjects):
    idx = np.nonzero(target == seekingClass)[0]

    mask = np.zeros(len(idx))
    mask[0: min(len(idx) - 1, nObjects)] = 1

    mask = np.random.permutation(mask)
    idxM = np.nonzero(mask)[0]

    return idx[idxM]

def GetSubSet(dataSet, target, nObjects):
    vClasses, parts = np.unique(target, return_counts=True)
    parts = parts / len(target)

    nParts = np.floor(nObjects * parts).astype(int)

    objectsPerClass = 2 * np.maximum(np.ones(len(nParts), dtype=int), nParts)

    subSetIdx = []

    for iClass in np.arange(len(vClasses)):
        idx = GetObjectsPerClass(target, vClasses[iClass], objectsPerClass[iClass])
        subSetIdx = subSetIdx + idx.tolist()

    return dataSet[subSetIdx], target[subSetIdx]


def GetSubSetOnFeatures(x, nFeatures):
    totalFeatures = x.shape[1]
    selectedFeatures = np.random.choice(np.arange(totalFeatures), size=nFeatures, replace=False)
    x = x[:, selectedFeatures]

    return x