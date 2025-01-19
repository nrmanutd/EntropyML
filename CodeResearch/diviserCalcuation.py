import math

import numpy as np

from sortedcontainers import SortedList, SortedSet, SortedDict
from CodeResearch.rademacherHelpers import GetSortedData

def getDeltaForFeature(sortedIdx, target, featureState, cClass, cClassKoeff, oClassKoeff, curOmitedObjects):

    newIdx = -1
    delta = 0
    curStatePositive = False

    for iObject in range(featureState, 0, -1):
        if sortedIdx[iObject] in curOmitedObjects:
            newIdx = iObject - 1
            continue

        d = cClassKoeff if target[sortedIdx[iObject]] == cClass else oClassKoeff

        if d < 0 and curStatePositive:
            newIdx = iObject
            break

        if d > 0 and not curStatePositive:
            curStatePositive = True

        delta += d
        newIdx = iObject - 1

    return newIdx, delta


def getNextStep(curState, curOmitedObjects, sortedIdx, sortedValues, target, cClass, cClassKoeff, oClassKoeff):

    nFeatures = sortedValues.shape[1]

    bestDelta = -1000
    bestFeature = -1
    bestNewIdx = -1
    idxToOmit = []

    for iFeature in range(0, nFeatures):
        if curState[iFeature] == 0:
            continue

        newIdx, delta = getDeltaForFeature(sortedIdx[:, iFeature], target, curState[iFeature], cClass, cClassKoeff, oClassKoeff, curOmitedObjects)

        if newIdx == 0 and delta <= 0:
            continue

        if delta > bestDelta:
            bestDelta = delta
            bestFeature = iFeature
            bestNewIdx = newIdx
            idxToOmit = sortedIdx[range(curState[iFeature], bestNewIdx, -1), iFeature]

    return bestFeature, bestNewIdx, bestDelta, idxToOmit

def getMaximumDiviserPerClass(sortedIdx, sortedValues, target, cClass, cClassKoeff, oClassKoeff):
    nObjects = sortedValues.shape[0]
    nFeatures = sortedValues.shape[1]

    curState = np.ones(nFeatures, dtype=int) * (nObjects - 1)
    curOmitedObjects = set()
    curBalance = 0
    maxBalance = 0
    maxState = []

    while True:
        iFeature, newIdx, delta, IdxToOmit = getNextStep(curState, curOmitedObjects, sortedIdx, sortedValues, target, cClass, cClassKoeff, oClassKoeff)

        if iFeature == -1:
            break

        curBalance += delta
        curState[iFeature] = newIdx
        curOmitedObjects.update(IdxToOmit)

        if curBalance > maxBalance:
            maxBalance = curBalance
            maxState = curState.copy()

    diviserValues = np.zeros(nFeatures)
    for kFeature in range(0, nFeatures):
        curIdx = maxState[kFeature]
        if curIdx < 0 or curIdx >= nObjects:
            print ('Cur Idx: {:}, ifeature: {:}, maxState: {:}'.format(curIdx, kFeature, maxState))

        diviserValues[kFeature] = sortedValues[maxState[kFeature], kFeature]

    mb = calculateDeltaIndependently(maxState, sortedIdx, target, cClass, cClassKoeff, oClassKoeff)

    if abs(mb - maxBalance) > 0.00001:
        print('!!!Error!!! Difference occured between actual and independent calculation: a {:}, i {:}'.format(maxBalance, mb))

    return maxBalance, diviserValues

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

def getMaximumDiviserProd(dataSet, target):

    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    sortedIdx, sortedValues = GetSortedData(dataSet)

    c1Banalce, c1diviser = getMaximumDiviserPerClass(sortedIdx, sortedValues, target, nClasses[0], 1 / counts[0], -1 / counts[1])
    c2Banalce, c2diviser = getMaximumDiviserPerClass(sortedIdx, sortedValues, target, nClasses[1], 1 / counts[1], -1 / counts[0])

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser

def getMaximumDiviser(dataSet, target):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    sortedIdx, sortedValues = GetSortedData(dataSet)

    curSet = set()
    totalBalance = 0

    diviser = np.zeros(nFeatures)
    diviserIdx = np.zeros(nFeatures, dtype=int)

    for iFeature in range(0, nFeatures):
        curBalance = totalBalance
        maxBalance = totalBalance
        curObject = nObjects

        for iObject in range(nObjects - 1, 0, -1):

            if iFeature > 0 and sortedIdx[iObject, iFeature] not in curSet:
                continue

            delta = 1/counts[0] if target[sortedIdx[iObject, iFeature]] == nClasses[0] else -1/counts[1]
            curBalance -= delta

            if abs(curBalance) > abs(maxBalance):
                maxBalance = curBalance
                curObject = iObject

        #print('Max diviser]: {:} of {:}, value]: {:}'.format(curObject - 1, nObjects, sortedValues[curObject - 1, iFeature]))

        curIdxes = set(sortedIdx[0:curObject, iFeature])
        diviser[iFeature] = sortedValues[curObject - 1, iFeature]
        diviserIdx[iFeature] = curObject - 1

        if iFeature == 0:
            curSet = curIdxes
        else:
            curSet = curSet.intersection(curIdxes)

        totalBalance = maxBalance

    mb1 = calculateDeltaIndependently(diviserIdx, sortedIdx, target, nClasses[0], 1 / counts[0], -1 / counts[1])
    mb2 = calculateDeltaIndependently(diviserIdx, sortedIdx, target, nClasses[1], 1 / counts[1], -1 / counts[0])

    if abs(abs(mb1) - abs(totalBalance)) > 0.00001 or abs(abs(mb2) - abs(totalBalance)) > 0.00001:
        print(
            '!!!Error!!! Difference occured between actual and independent calculation: a {:}, i1 {:}, i2: {:}'.format(maxBalance,
                                                                                                             mb1, mb2))

    return totalBalance, diviser


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

def calcDelta(curSet, valuedTarget1):
    delta = 0
    itemsToOmit = []
    alreadyPositive = False
    setPositive = False

    lastElement = curSet.peekitem(-1)[0]

    for keyPair in curSet:
        if keyPair == lastElement:
            return delta, itemsToOmit

        d = 0
        curItemsToOmit = curSet[keyPair]

        if len(curItemsToOmit) == 0:
            raise ValueError('Items to omit shouldnt be empty')

        for iObj in curItemsToOmit:
            curDelta = valuedTarget1[iObj]

            if curDelta < 0 and alreadyPositive:
                return delta, itemsToOmit

            if curDelta > 0:
                setPositive = True

            d += curDelta

        alreadyPositive |= setPositive
        delta += d
        itemsToOmit += curItemsToOmit

    return delta, itemsToOmit


def getNextStepFast(sortedDataSet, valuedTarget1):
    nFeatures = len(sortedDataSet)

    bestFeature = -1
    bestDelta = 0
    bestItemsToOmit = []

    for iFeature in range(0, nFeatures):
        curSet = sortedDataSet[iFeature]
        curDelta, curItemsToOmit = calcDelta(curSet, valuedTarget1)

        #print('Feature {:}: delta = {:}, items to omit = {:}'.format(iFeature, curDelta, curItemsToOmit))

        if len(curItemsToOmit) == 0:
            continue

        if curDelta > bestDelta:
            bestDelta = curDelta
            bestFeature = iFeature
            bestItemsToOmit = curItemsToOmit

    return bestFeature, bestDelta, bestItemsToOmit

def getMaximumDiviserPerClassFast(dataSet, valuedTarget1):
    sortedDataSet = GetSortedDict(dataSet)
    nFeatures = dataSet.shape[1]
    nClasses, counts = np.unique(valuedTarget1, return_counts=True)
    maxLeft = max(nClasses[0] * counts[0], nClasses[1] * counts[1])

    curBalance = 0
    maxBalance = 0
    maxState = np.zeros(nFeatures)

    while True:
        iFeature, delta, itemsToOmit = getNextStepFast(sortedDataSet, valuedTarget1)

        if iFeature == -1:
            break

        curBalance += delta
        #print('Step: i#{:}, d{:}, to omit:{:}. Cur balance: {:}, best balance: {:}'.format(iFeature, delta, itemsToOmit, curBalance, maxBalance))

        for item in itemsToOmit:
            curDecrment = valuedTarget1[item]
            if curDecrment > 0:
                maxLeft -= curDecrment

            for kFeature in range(0, nFeatures):
                vv = -dataSet[item, kFeature]
                curItem = sortedDataSet[kFeature][vv]
                if len(curItem) == 1:
                    sortedDataSet[kFeature].pop(vv)
                else:
                    curItem.remove(item)

        if curBalance > maxBalance:
            maxBalance = curBalance

            for kFeature in range(0, nFeatures):
                maxState[kFeature] = -sortedDataSet[kFeature].peekitem(0)[0]

        if maxLeft + curBalance <= maxBalance:
            break

    #mb = calculateDeltaIndependently2(dataSet, valuedTarget1, maxState)

    #if abs(mb - maxBalance) > 0.00001:
    #    raise ValueError('Error!!! Fast != independent: {:} vs {:}'.format(maxBalance, mb))

    return abs(maxBalance), maxState

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

def getMaximumDiviserFast(dataSet, target):
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    valuedTarget1 = GetValuedTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    c1Banalce, c1diviser = getMaximumDiviserPerClassFast(dataSet, valuedTarget1)

    valuedTarget2 = GetValuedTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    c2Banalce, c2diviser = getMaximumDiviserPerClassFast(dataSet, valuedTarget2)

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser