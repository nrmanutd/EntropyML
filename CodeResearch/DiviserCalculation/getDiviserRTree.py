import bisect

import numpy as np

from rtree import index

from CodeResearch.DiviserCalculation.diviserCheckers import calculateDeltaIndependently2
from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget, GetSortedDict
from CodeResearch.rademacherHelpers import GetSortedData


def getConcreteScore(iFeature, iObject, sortedIdx, sortedValues, bestScore, valuedTarget):
    pass


def getIdx(dataSet, id):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    p = index.Property()
    p.dimension = nFeatures

    idx = index.Index(properties=p)
    idx.properties.dimension = nFeatures

    query = np.zeros(2 * nFeatures)
    query[0:nFeatures] = dataSet[0, :]
    query[nFeatures:2 * nFeatures] = dataSet[0, :]

    for iObject in range(0, nObjects):
        point = np.zeros(2 * nFeatures)
        point[0:nFeatures] = dataSet[iObject, :]
        point[nFeatures:2 * nFeatures] = dataSet[iObject, :]
        idx.insert(id[iObject], point)

    return idx

def GetRTreeIndex(dataSet, valuedTarget):
    positiveIdx = np.where(valuedTarget > 0)[0]
    negativeIdx = np.where(valuedTarget < 0)[0]

    positive = getIdx(dataSet[positiveIdx, :], positiveIdx)
    negative = getIdx(dataSet[negativeIdx, :], negativeIdx)

    return positive, negative


def getPointsUnderDiviser(idx, currentDiviser, basePoint):

    nFeatures = len(currentDiviser)
    query = np.zeros(2 * nFeatures)

    query[0:nFeatures] = basePoint
    query[nFeatures:2*nFeatures] = currentDiviser

    res = list(idx.intersection(query))

    return len(res)


def updateDiviserConcrete(newDiviser, value, sortedIdx, sortedValues, valuedTarget):
    nFeatures = sortedIdx.shape[1]

    for fFeature in range(0, nFeatures):
        if newDiviser[fFeature] < value[fFeature]:
            continue

        idx = bisect.bisect_left(sortedValues[:, fFeature], value[fFeature])
        for iObject in range(idx, -1, -1):
            if valuedTarget[sortedIdx[iObject, fFeature]] < 0:
                newDiviser[fFeature] = sortedValues[iObject, fFeature]
                break

    return newDiviser

def updateDiviser(currentDiviser, origIdxs, sortedIdx, sortedValues, valuedTarget, dataSet):

    newDiviser = currentDiviser.copy()#probably place for optimization

    for idx in origIdxs:
        if valuedTarget[idx] > 0:
            continue

        newDiviser = updateDiviserConcrete(newDiviser, dataSet[idx, :], sortedIdx, sortedValues, valuedTarget)

    return newDiviser


def getBestStartDiviser(sortedIdx, sortedValues, targetValue):

    nObjects = sortedIdx.shape[0]
    nFeatures = sortedIdx.shape[1]
    diviser = np.zeros(nFeatures)

    bestScore = 0.0
    addedIdxes = set()

    for iFeature in range(0, nFeatures):
        for iObject in range(nObjects - 1, -1, -1):
            if targetValue[sortedIdx[iObject, iFeature]] < 0:
                diviser[iFeature] = sortedValues[iObject, iFeature]
                break

            if sortedIdx[iObject, iFeature] not in addedIdxes:
                bestScore += targetValue[sortedIdx[iObject, iFeature]]
                addedIdxes.add(sortedIdx[iObject, iFeature])

    return diviser, bestScore

def getMaximumDiviserPerClassRT(dataSet, valuedTarget):

    nClasses, counts = np.unique(valuedTarget, return_counts=True)

    negativeScore = nClasses[0] if nClasses[0] < 0 else nClasses[1]
    negativeCount = counts[0] if nClasses[0] < 0 else counts[1]
    positiveScore = nClasses[0] if nClasses[0] > 0 else nClasses[1]
    positiveCount = counts[0] if nClasses[0] > 0 else counts[1]

    nFeatures = dataSet.shape[1]

    sortedIdx, sortedValues = GetSortedData(dataSet)
    sortedDataSet = GetSortedDict(dataSet)

    positiveIdx, negativeIdx = GetRTreeIndex(dataSet, valuedTarget)

    bestDiviser, bestScore = getBestStartDiviser(sortedIdx, sortedValues, valuedTarget)
    basePoint = sortedValues[0, :]

    for iFeature in range(0, nFeatures):
        currentDiviser, currentScore = getBestStartDiviser(sortedIdx, sortedValues, valuedTarget)
        currentIdx = sortedDataSet[iFeature]

        for iValue in currentIdx:
            origIdxs = currentIdx[iValue]

            negativeDetected = False
            for idx in origIdxs:
                if valuedTarget[idx] < 0:
                    negativeDetected = True
                    break

            if not negativeDetected:
                continue

            currentDiviser = updateDiviser(currentDiviser, origIdxs, sortedIdx, sortedValues, valuedTarget, dataSet)

            positivePoints = getPointsUnderDiviser(positiveIdx, currentDiviser, basePoint)
            negativePoints = getPointsUnderDiviser(negativeIdx, currentDiviser, basePoint)

            currentScore = (positiveCount - positivePoints) * positiveScore + (negativeCount - negativePoints) * negativeScore

            #добавить проверку на раннюю остановку
            if currentScore > bestScore:
                bestScore = currentScore
                bestDiviser = currentDiviser.copy()

    mb = calculateDeltaIndependently2(dataSet, valuedTarget, bestDiviser)

    if abs(mb - bestScore) > 0.00001:
        print(dataSet)
        print(valuedTarget)
        print('Error!!! Correct != independent: {:} vs {:}, {:}'.format(bestScore, mb, bestDiviser))

        raise ValueError('Error!!! Correct != independent: {:} vs {:}, {:}'.format(bestScore, mb, bestDiviser))

    return bestScore, bestDiviser

def getMaximumDiviserRTree(dataSet, target):
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    valuedTarget1 = GetValuedTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    c1Banalce, c1diviser = getMaximumDiviserPerClassRT(dataSet, valuedTarget1)

    valuedTarget2 = GetValuedTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    c2Banalce, c2diviser = getMaximumDiviserPerClassRT(dataSet, valuedTarget2)

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser