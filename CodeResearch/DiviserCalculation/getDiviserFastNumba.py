import numpy as np
from numba import jit, prange

from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget, prepareDataSet, getSortedSet


@jit(nopython=True)
def calcDelta(curSet, valuedTarget1):
    delta = 0
    itemsToOmit = []
    alreadyPositive = False
    setPositive = False

    totalElements = len(curSet)
    currentElementIndex = 0

    for keyPair in curSet:
        if currentElementIndex == totalElements - 1:
            return delta, itemsToOmit

        d = 0
        curItemsToOmit = curSet[keyPair]

        if len(curItemsToOmit) == 0:
            raise ValueError('Items to omit shouldnt be empty')

        for iObj in curItemsToOmit:
            curDelta = valuedTarget1[iObj]

            if curDelta < 0 and alreadyPositive:  #todo: fix bug for different objects on same values. Not finishing...
                return delta, itemsToOmit

            if curDelta > 0:
                setPositive = True

            d += curDelta

        alreadyPositive |= setPositive
        delta += d
        itemsToOmit += curItemsToOmit
        currentElementIndex += 1

    return delta, itemsToOmit


@jit(nopython=True)
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


@jit(nopython=True)
def updateToNewState(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta):
    nFeatures = len(currentState)
    delta = 0

    for iFeature in prange(0, nFeatures):
        for iSortedObject in range(currentState[iFeature], -1, -1):
            curObject = sortedDataSet[iSortedObject, iFeature]

            if omitedObjects[curObject] is False and valuedTarget[curObject] < 0:
                currentState[iFeature] = iSortedObject
                break

            if omitedObjects[curObject] is False:
                delta += valuedTarget[curObject]
                omitedObjects[curObject] = True

    return currentState, delta, omitedObjects

@jit(nopython=True)
def updateOmitedObjects(currentState, sortedFeature, feature, valuedTarget, omitedObjects):

    omitedDelta = np.zeros(len(omitedObjects))
    iOmitedIdx = 0
    wasPositive = False
    positiveValue = 0
    delta = 0

    for iSortedObject in prange(currentState, -1, -1):
        iObject = sortedFeature[iSortedObject]

        if wasPositive and valuedTarget[iObject] < 0 and feature[iObject] != positiveValue:
            break

        if omitedObjects[iSortedObject] is True:
            continue

        omitedDelta[iOmitedIdx] = iObject
        iOmitedIdx += 1
        delta += valuedTarget[iObject]

        if valuedTarget[iObject] > 0:
            positiveValue = feature[iObject] if wasPositive is False else positiveValue
            wasPositive |= True

    return omitedDelta, delta


@jit(nopython=True)
def getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget, nClasses, counts):
    sortedDataSet = getSortedSet(dataSet)
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]
    maxLeft = max(nClasses[0] * counts[0], nClasses[1] * counts[1])

    curBalance = 0
    maxBalance = 0
    maxState = np.zeros(nFeatures)
    currentState = np.ones(nFeatures) * (nObjects - 1)
    omitedObjects = np.zeros(nObjects, dtype=bool)

    while True:
        iFeature = getNextStepFast(sortedDataSet, valuedTarget, currentState)

        if iFeature == -1:
            break

        omitedDelta, delta = updateOmitedObjects(currentState[iFeature], sortedDataSet[:, iFeature],
                                                   dataSet[:, iFeature], valuedTarget, omitedObjects)
        curBalance += delta

        currentState, delta, omitedDelta = updateToNewState(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta)
        curBalance += delta

        for iObject in prange(0, len(omitedObjects)):
            if omitedDelta[iObject] == 0:
                break

            omitedObjects[omitedDelta[iObject]] = True

        #print('Step: i#{:}, d{:}, to omit:{:}. Cur balance: {:}, best balance: {:}'.format(iFeature, delta, itemsToOmit, curBalance, maxBalance))

        if curBalance > maxBalance:
            maxBalance = curBalance

            for kFeature in prange(0, nFeatures):
                objectIdx = sortedDataSet[currentState[kFeature], kFeature]
                maxState[kFeature] = dataSet[objectIdx, kFeature]

        if maxLeft + curBalance <= maxBalance:
            break

    #mb = calculateDeltaIndependently2(dataSet, valuedTarget1, maxState)

    #if abs(mb - maxBalance) > 0.00001:
    #    raise ValueError('Error!!! Fast != independent: {:} vs {:}'.format(maxBalance, mb))

    return abs(maxBalance), maxState


@jit(nopython=True)
def getMaximumDiviserFastNumba(dataSet, target):
    dataSet = prepareDataSet(dataSet)
    nClasses = np.unique(target)
    counts = np.zeros(2)
    counts[0] = len(np.where(target == nClasses[0])[0])
    counts[1] = len(np.where(target == nClasses[1])[0])

    if len(nClasses) != 2:
        #raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))
        print('Error!!! Number of classes should be equal to two, instead ', len(nClasses))

    valuedTarget1 = GetValuedTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    c1Banalce, c1diviser = getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget1, nClasses, counts)

    valuedTarget2 = GetValuedTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    c2Banalce, c2diviser = getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget2, nClasses, counts)

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser
