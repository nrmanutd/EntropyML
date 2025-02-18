import numpy as np
import numba as nb
from numba import jit, prange

from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget, prepareDataSet, getSortedSet, f2s, bv2s, \
    iv2s


@jit(nopython=True, parallel=True)
def updateToNewState(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta, updateState):
    nFeatures = len(currentState)

    for iFeature in prange(0, nFeatures):
        for iSortedObject in range(currentState[iFeature], -1, -1):
            iObject = sortedDataSet[iSortedObject, iFeature]

            if not omitedObjects[iObject]:
                if valuedTarget[iObject] < 0:
                    if updateState:
                        currentState[iFeature] = iSortedObject
                    break

                omitedDelta[iObject] = True

    return currentState, omitedDelta

@jit(nopython=True)
def updateOmitedObjects(currentState, sortedFeature, feature, valuedTarget, omitedObjects, omitedDelta):

    wasPositive = False
    positiveValue = 0

    for iSortedObject in range(currentState, -1, -1):
        iObject = sortedFeature[iSortedObject]

        if wasPositive and valuedTarget[iObject] < 0 and feature[iObject] != positiveValue:#todo: fix bug when same value and positive was first but then i'm stopping at this value. Should just make sorting in sortedDataSet so negative values are going first and then positive (targetValues)
            break

        if omitedObjects[iObject]:
            continue

        omitedDelta[iObject] = True

        if valuedTarget[iObject] > 0:
            positiveValue = positiveValue if wasPositive else feature[iObject]
            wasPositive |= True

    return omitedDelta

@jit(nopython=True, parallel=True)
def getNextStepFast(dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedMatrix):
    nFeatures = dataSet.shape[1]
    deltas = np.zeros(nFeatures)

    print('Total features: ', str(nFeatures))
    for iFeature in prange(0, nFeatures):
        delta = calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedMatrix[:, iFeature], False)[2]
        print('Feature: {' + str(iFeature) + '} Delta: {' + f2s(delta) + '}')
        deltas[iFeature] = delta

    bestFeature = np.argmax(deltas)
    print('Best Feature: ', bestFeature)
    return np.argmax(deltas)

@jit(nopython=True, parallel=True)
def calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedDelta, updateOmited):
    print('Valued target: ', valuedTarget)
    print('Before: ' + str(iFeature) + ' ', str(len(omitedDelta)))

    omitedDelta = updateOmitedObjects(currentState[iFeature], sortedDataSet[:, iFeature],
                                      dataSet[:, iFeature], valuedTarget, omitedObjects, omitedDelta)

    print('After: ' + str(iFeature) + ' ', bv2s(omitedDelta))
    currentState, omitedDelta = updateToNewState(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta, updateOmited)

    delta = 0
    for iObject in prange(0, len(omitedDelta)):
        if not omitedDelta[iObject]:
            continue

        omitedDelta[iObject] = False

        if updateOmited:
            omitedObjects[iObject] = True

        delta += valuedTarget[iObject]

    print('Delta: ', delta)

    return currentState, omitedObjects, delta

@jit(nopython=True, parallel=True)
def getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget, nClasses, counts):
    sortedDataSet = getSortedSet(dataSet)
    print(dataSet)
    print(sortedDataSet)
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]
    maxLeft = max(nClasses[0] * counts[0], nClasses[1] * counts[1])

    curBalance = 0
    maxBalance = 0
    maxState = np.zeros(nFeatures)
    currentState = np.ones(nFeatures, dtype=nb.int64) * (nObjects - 1)
    omitedObjects = np.zeros(nObjects, dtype=nb.bool)
    omitedDelta = np.zeros(nObjects, dtype=nb.bool)
    omitedMatrix = np.zeros((nObjects, nFeatures), dtype=nb.int64)
    iStep = 0

    while True:
        print('Inside while true...')
        iFeature = getNextStepFast(dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedMatrix)
        print('Selected feature: ', iFeature)
        print('Current state: ', iv2s(currentState))

        if iFeature == -1:
            break

        currentState, omitedObjects, delta = calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedDelta, True)

        curBalance += delta
        print('Omited objects: ', bv2s(omitedObjects))
        print('Feature: i#{' + str(iFeature) + '}, d{' + f2s(delta) + '}. Cur balance: {' + f2s(curBalance) + '}, best balance: {' + f2s(maxBalance) + '}')

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
