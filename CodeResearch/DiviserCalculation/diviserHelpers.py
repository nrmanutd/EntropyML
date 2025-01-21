import numpy as np
import numba
from sortedcontainers import SortedDict

def GetValuedTarget(target, c1, c1p, c2p):

    res = np.zeros(len(target))
    for iObject in range(0, len(target)):
        res[iObject] = c1p if target[iObject] == c1 else c2p

    return res

def GetSortedDict(dataSet):
    res = []
    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    for iFeature in range(0, nFeatures):
        curDict = SortedDict()
        res.append(curDict)

        for iObject in range(0, nObjects):
            v = -dataSet[iObject, iFeature]
            if v not in curDict:
                curDict[v] = {iObject}
            else:
                curDict[v].add(iObject)

    return res

def GetSortedDictByIndex(sortedIdx, objects):
    res = []
    nFeatures = objects.shape[1]
    nObjects = objects.shape[0]

    for iFeature in range(0, nFeatures):
        curDict = dict()
        res.append(curDict)

        for iObject in range(nObjects - 1, -1, -1):
            origIdx = sortedIdx[iObject, iFeature]
            curDict[origIdx] = objects[origIdx, iFeature]

    return res

def GetSortedDictList(dataSet):
    res = []
    nFeatures = dataSet.shape[1]
    nObjects = dataSet.shape[0]

    for iFeature in range(0, nFeatures):
        curDict = SortedDict()
        res.append(curDict)

        for iObject in range(0, nObjects):
            v = -dataSet[iObject, iFeature]
            if v not in curDict:
                curDict[v] = {iObject}
            else:
                curDict[v].add(iObject)

    return res