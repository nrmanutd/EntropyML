import numba as nb
import numpy as np
from numba import jit, prange

from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget, prepareDataSet, getSortedSet, doubleDataSet


@jit(nopython=True)
def updateStateOnOtherFeatures(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta, updateState):
    nFeatures = len(currentState)

    for iFeature in range(0, nFeatures):
        for iSortedObject in range(currentState[iFeature], -1, -1):
            iObject = sortedDataSet[iSortedObject, iFeature]

            if not omitedObjects[iObject] and not omitedDelta[iObject]:
                if valuedTarget[iObject] < 0:
                    if updateState:
                        currentState[iFeature] = iSortedObject
                    break

                omitedDelta[iObject] = True

    return currentState, omitedDelta

@jit(nopython=True)
def makeStepForConcreteFeature(currentState, sortedFeature, feature, valuedTarget, omitedObjects, omitedDelta):

    wasPositive = False
    positiveValue = 0

    for iSortedObject in range(currentState, -1, -1):
        iObject = sortedFeature[iSortedObject]

        if wasPositive and valuedTarget[iObject] < 0 and feature[iObject] != positiveValue:
            return omitedDelta, iSortedObject

        if omitedObjects[iObject]:
            continue

        omitedDelta[iObject] = True

        if valuedTarget[iObject] > 0:
            positiveValue = positiveValue if wasPositive else feature[iObject]
            wasPositive |= True

    return omitedDelta, -1

@jit(nopython=True)
def calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedDelta, updateOmited):

    omitedDelta, idx = makeStepForConcreteFeature(currentState[iFeature], sortedDataSet[:, iFeature],
                                      dataSet[:, iFeature], valuedTarget, omitedObjects, omitedDelta)

    if idx == -1:
        return currentState, omitedObjects, -1, -2

    currentState, omitedDelta = updateStateOnOtherFeatures(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta, updateOmited)

    delta = 0
    addedPositives = 0
    for iObject in range(0, len(omitedDelta)):
        if not omitedDelta[iObject]:
            continue

        omitedDelta[iObject] = False

        if updateOmited:
            omitedObjects[iObject] = True

        delta += valuedTarget[iObject]
        addedPositives += valuedTarget[iObject] if valuedTarget[iObject] > 0 else 0

    return currentState, omitedObjects, addedPositives, delta

@jit(nopython=True, parallel=True)
def getNextStepFast(dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedMatrix):
    nFeatures = dataSet.shape[1]
    res = np.full(nFeatures, -2, dtype=nb.float64)

    for iFeature in prange(0, nFeatures):
        delta = calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedMatrix[:, iFeature], False)[3]
        res[iFeature] = delta

    bestIndex = np.argmax(res)

    if res[bestIndex] < -1:
        return -1

    return bestIndex

@jit(nopython=True, parallel=True)
def getMinPositives(sortedDataSet, valuedTarget):

    nFeatures = sortedDataSet.shape[1]
    nObjects = sortedDataSet.shape[0]
    firstPositiveObjects = np.zeros(nFeatures, dtype=nb.int64)

    for iFeature in prange(nFeatures):
        for iSortedObject in range(nObjects):
            iObject = sortedDataSet[iSortedObject, iFeature]

            if valuedTarget[iObject] > 0:
                firstPositiveObjects[iFeature] = iSortedObject
                break

    return firstPositiveObjects

@jit(nopython=True, parallel=True)
def getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget, sortedDataSet):
    minPositives = getMinPositives(sortedDataSet, valuedTarget)

    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    if nFeatures == 0:
        return 0, np.zeros(0)

    curBalance = 0
    maxBalance = 0
    maxState = np.zeros(nFeatures)
    currentState = np.ones(nFeatures, dtype=nb.int64) * (nObjects - 1)
    omitedObjects = np.full(nObjects, False, dtype=nb.bool)
    omitedDelta = np.full(nObjects, False,  dtype=nb.bool)
    omitedMatrix = np.full((nObjects, nFeatures), False, dtype=nb.bool)
    stoppingCriteria = False
    maxLeft = 1

    iSteps = 0

    while True:
        iSteps += 1

        for iFeature in range(nFeatures):
            if minPositives[iFeature] > currentState[iFeature]:
                stoppingCriteria = True
                break

        if stoppingCriteria:
            break

        iFeature = getNextStepFast(dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedMatrix)
        if iFeature == -1:
            break

        currentState, omitedObjects, addedPositives, delta = calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedDelta, True)

        maxLeft -= addedPositives

        curBalance += delta
        #print('Feature: #{' + str(iFeature) + '}, d{' + f2s(delta, 20) + '}. Cur balance: {' + f2s(curBalance, 20) + '}, best balance: {' + f2s(maxBalance, 10) + '}')

        if curBalance > maxBalance:
            maxBalance = curBalance

            for kFeature in range(0, nFeatures):
                objectIdx = sortedDataSet[currentState[kFeature], kFeature]
                maxState[kFeature] = dataSet[objectIdx, kFeature]

        if maxLeft + curBalance <= maxBalance:
            break

    return abs(maxBalance), maxState

@jit(nopython=True)
def getMaximumDiviserFastNumba(dataSet, target):
    dataSet = prepareDataSet(dataSet)
    dataSet = doubleDataSet(dataSet)
    nClasses = np.unique(target)

    counts = np.zeros(2, dtype=nb.int64)
    counts[0] = len(np.where(target == nClasses[0])[0])
    counts[1] = len(np.where(target == nClasses[1])[0])

    if len(nClasses) != 2:
        #raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))
        print('Error!!! Number of classes should be equal to two, instead ', len(nClasses))

    valuedTarget1 = GetValuedTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    valuedTarget2 = GetValuedTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])

    sds1 = getSortedSet(dataSet, valuedTarget1)
    sds2 = getSortedSet(dataSet, valuedTarget2)

    return getMaximumDiviserFastNumbaCore(dataSet, target, valuedTarget1, sds1, valuedTarget2, sds2)

@jit(nopython=True)
def getMaximumDiviserFastNumbaCore(dataSet, target, valuedTarget1, sortedSet1, valuedTarget2, sortedSet2):

    nClasses = np.unique(target)
    counts = np.zeros(2, dtype=nb.int32)
    counts[0] = len(np.where(target == nClasses[0])[0])
    counts[1] = len(np.where(target == nClasses[1])[0])

    if len(nClasses) != 2:
        #raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))
        print('Error!!! Number of classes should be equal to two, instead ', len(nClasses))

    c1Banalce, c1diviser = getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget1, sortedSet1)
    c2Banalce, c2diviser = getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget2, sortedSet2)

    firstClass = target[0]
    secondClass = nClasses[0] if firstClass != nClasses[0] else nClasses[1]

    classUnderDiviser = firstClass if valuedTarget1[0] < 0 else secondClass

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser, classUnderDiviser

    classUnderDiviser = firstClass if classUnderDiviser != firstClass else secondClass

    return c2Banalce, c2diviser, classUnderDiviser
