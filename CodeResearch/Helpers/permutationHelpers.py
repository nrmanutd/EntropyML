import math

import numpy as np

def extractDataSet(x, y, nObjects, nFeatures):
    xx, yy = GetSubSet(x, y, math.floor(nObjects / 2))
    xx = GetSubSetOnFeatures(xx, nFeatures)

    return xx, yy

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
    return x[:, selectedFeatures]


def getDataSetIndexesOfTwoClasses(currentObjects, target, iClass, jClass):
    iClassIdx = np.where(target == iClass)[0]
    jClassIdx = np.where(target == jClass)[0]

    # print('Total objects: {:}, iClass: {:}, jClass: {:}, currentObjects: {:}'.format(dataSet.shape[0], len(iClassIdx), len(jClassIdx), currentObjects))

    partIClass = len(iClassIdx) / (len(iClassIdx) + len(jClassIdx))

    iObjectsCount = math.ceil(partIClass * currentObjects) if partIClass < 0.5 else math.floor(
        partIClass * currentObjects)
    jObjectsCount = currentObjects - iObjectsCount

    iClassObjects = GetObjectsPerClass(target, iClass, iObjectsCount)
    jClassObjects = GetObjectsPerClass(target, jClass, jObjectsCount)

    return iClassObjects, jClassObjects


def getDataSetOfTwoClassesCore(dataSet, target, iClassObjects, jClassObjects):

    iObjectsCount = len(iClassObjects)
    jObjectsCount = len(jClassObjects)

    nFeatures = dataSet.shape[1]
    newSet = np.zeros((iObjectsCount + jObjectsCount, nFeatures))

    newSet[0:iObjectsCount, :] = dataSet[iClassObjects, :]
    newSet[iObjectsCount:(iObjectsCount + jObjectsCount), :] = dataSet[jClassObjects, :]

    newTarget = np.zeros(iObjectsCount + jObjectsCount)
    newTarget[0:iObjectsCount] = target[iClassObjects]
    newTarget[iObjectsCount: (iObjectsCount + jObjectsCount)] = target[jClassObjects]

    return newSet, newTarget
