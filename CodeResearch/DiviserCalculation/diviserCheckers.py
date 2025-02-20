import numpy as np
import numba as nb
from numba import jit, prange


def calculateDeltaIndependently(curState, sortedIdx, target, cClass, cClassKoeff, oClassKoeff):
    nObjects = len(target)
    allIdx = set(range(0, nObjects))

    for iFeature in range(0, len(curState)):
        indexes = sortedIdx[0:(curState[iFeature] + 1), iFeature]
        #curSet = set(indexes)
        allIdx = allIdx.intersection(indexes)

    curSum = 0
    for tIdx in allIdx:
        curSum += cClassKoeff if target[tIdx] == cClass else oClassKoeff

    totalIdx = set(range(0, nObjects))
    for tIdx in allIdx:
        totalIdx.remove(tIdx)

    curSum2 = 0
    for tIdx in totalIdx:
        curSum2 += cClassKoeff if target[tIdx] == cClass else oClassKoeff

    #print('Cursum: {:}, cursum2: {:}'.format(curSum, curSum2))

    return max(abs(curSum), abs(curSum2))

def calculateDeltaIndependently2(dataSet, targetValues, point):
    nObjects = len(targetValues)
    allIdx = set(range(0, nObjects))
    nFeatures = dataSet.shape[1]

    for iFeature in range(0, nFeatures):
        indexes = np.where(dataSet[:, iFeature] <= point[iFeature])[0]
        allIdx = allIdx.intersection(indexes)

    curSum = 0
    for tIdx in allIdx:
        curSum += targetValues[tIdx]

    return abs(curSum)

@jit(nopython=True, parallel=True)
def calculateDeltaIndependently3(dataSet, targetValues, point):
    nObjects = len(targetValues)
    allIdx = np.zeros(nObjects, dtype=nb.int64)
    nFeatures = dataSet.shape[1]

    for iObject in prange(nObjects):
        objectOut = False
        for iFeature in range(nFeatures):
            if dataSet[iObject, iFeature] > point[iFeature]:
                objectOut = True
                break

        if not objectOut:
            allIdx[iObject] = 1

    curSum = 0
    for tIdx in prange(nObjects):
        curSum += targetValues[tIdx] * allIdx[tIdx]

    return abs(curSum)