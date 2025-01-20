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


def updateDiviserConcrete(newDiviser, value, sortedNegIdx, sortedNegValues):
    nFeatures = sortedNegIdx.shape[1]

    for fFeature in range(0, nFeatures):
        if newDiviser[fFeature] < value[fFeature]:
            continue

        idx = bisect.bisect_left(sortedNegValues[:, fFeature], value[fFeature])
        newDiviser[fFeature] = sortedNegValues[idx, fFeature]

    return newDiviser

def updateDiviser(currentDiviser, values, sortedNegIdx, sortedNegValues):
    newDiviser = currentDiviser
    nValues = values.shape[0]

    for iValue in range(0, nValues):
        newDiviser = updateDiviserConcrete(newDiviser, values[iValue, :], sortedNegIdx, sortedNegValues)

    return newDiviser

def getBestStartDiviser(sortedNegValues, positiveScore, positiveCount, basePoint, positiveIdx):

    nNegObjects = sortedNegValues.shape[0]
    bestDiviser = sortedNegValues[nNegObjects - 1, :]

    positivePointsUnderDiviser = getPointsUnderDiviser(positiveIdx, bestDiviser, basePoint)
    bestScore = (positiveCount - positivePointsUnderDiviser) * positiveScore

    return bestDiviser, bestScore

def getMaximumDiviserPerClassRT(dataSet, valuedTarget):

    nClasses, counts = np.unique(valuedTarget, return_counts=True)

    negativeScore = nClasses[0] if nClasses[0] < 0 else nClasses[1]
    negativeCount = counts[0] if nClasses[0] < 0 else counts[1]
    positiveScore = nClasses[0] if nClasses[0] > 0 else nClasses[1]
    positiveCount = counts[0] if nClasses[0] > 0 else counts[1]

    nFeatures = dataSet.shape[1]
    positiveIdx, negativeIdx = GetRTreeIndex(dataSet, valuedTarget)

    negativeObjectsIdx = np.where(valuedTarget < 0)[0]
    negativeObjects = dataSet[negativeObjectsIdx, :]
    sortedNegIdx, sortedNegValues = GetSortedData(negativeObjects)
    baseNegPoint = sortedNegValues[0, :]

    positiveObjectsIdx = np.where(valuedTarget > 0)[0]
    positiveObjects = dataSet[positiveObjectsIdx, :]
    sortedPosIdx, sortedPosValues = GetSortedData(positiveObjects)
    basePosPoint = sortedPosValues[0, :]

    sortedNegDataSet = GetSortedDict(negativeObjects)

    bestStartDiviser, bestStartScore = getBestStartDiviser(sortedNegValues, positiveScore, positiveCount, basePosPoint, positiveIdx)

    bestDiviser = bestStartDiviser
    bestScore = bestStartScore

    for iFeature in range(0, nFeatures):
        currentDiviser = bestStartDiviser
        currentIdx = sortedNegDataSet[iFeature]

        for iValue in currentIdx:
            idx = list(currentIdx[iValue])
            currentDiviser = updateDiviser(currentDiviser, negativeObjects[idx, :], sortedNegIdx, sortedNegValues)

            positivePoints = getPointsUnderDiviser(positiveIdx, currentDiviser, basePosPoint)
            negativePoints = getPointsUnderDiviser(negativeIdx, currentDiviser, baseNegPoint)

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