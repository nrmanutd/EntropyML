import time

import numba as nb
import numpy as np
from numba import jit, prange, objmode

from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget, prepareDataSet, getSortedSet, f2s, fv2s


@jit(nopython=True, parallel=True)
def updateStateOnOtherFeatures(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta, updateState):
    nFeatures = len(currentState)

    for iFeature in prange(0, nFeatures):
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

    for iFeature in range(0, nFeatures):
        delta = calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedMatrix[:, iFeature], False)[3]
        deltas[iFeature] = delta

    return np.argmax(deltas)

@jit(nopython=True, parallel=True)
def calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedDelta, updateOmited):
    omitedDelta = makeStepForConcreteFeature(currentState[iFeature], sortedDataSet[:, iFeature],
                                      dataSet[:, iFeature], valuedTarget, omitedObjects, omitedDelta)

    currentState, omitedDelta = updateStateOnOtherFeatures(currentState, sortedDataSet, valuedTarget, omitedObjects, omitedDelta, updateOmited)

    delta = 0
    addedPositives = 0
    for iObject in prange(0, len(omitedDelta)):
        if not omitedDelta[iObject]:
            continue

        omitedDelta[iObject] = False

        if updateOmited:
            omitedObjects[iObject] = True

        delta += valuedTarget[iObject]
        addedPositives += valuedTarget[iObject] if valuedTarget[iObject] > 0 else 0

    return currentState, omitedObjects, addedPositives, delta

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
def getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget):
    sortedDataSet = getSortedSet(dataSet, valuedTarget)
    minPositives = getMinPositives(sortedDataSet, valuedTarget)

    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    curBalance = 0
    maxBalance = 0
    maxState = np.zeros(nFeatures)
    currentState = np.ones(nFeatures, dtype=nb.int64) * (nObjects - 1)
    omitedObjects = np.zeros(nObjects, dtype=nb.bool)
    omitedDelta = np.zeros(nObjects, dtype=nb.bool)
    omitedMatrix = np.zeros((nObjects, nFeatures), dtype=nb.int64)
    stoppingCriteria = False
    maxLeft = 1

    #print('Current minPositives: ', minPositives)

    getFastTime = 0
    makeStepTime = 0
    iSteps = 0

    while True:
        iSteps += 1
        #print('Current state: ', currentState)

        for iFeature in range(nFeatures):
            if minPositives[iFeature] > currentState[iFeature]:
                stoppingCriteria = True
                break

        if stoppingCriteria:
            break

        with objmode(t1='f8'):
            t1 = time.time()

        iFeature = getNextStepFast(dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedMatrix)
        with objmode(t2='f8'):
            t2 = time.time()

        getFastTime += (t2 - t1)

        with objmode(t1='f8'):
            t1 = time.time()
        currentState, omitedObjects, addedPositives, delta = calcDelta(iFeature, dataSet, sortedDataSet, valuedTarget, currentState, omitedObjects, omitedDelta, True)

        with objmode(t2='f8'):
            t2 = time.time()
        makeStepTime += (t2 - t1)

        maxLeft -= addedPositives

        curBalance += delta
        #print('Feature: #{' + str(iFeature) + '}, d{' + f2s(delta) + '}. Cur balance: {' + f2s(curBalance) + '}, best balance: {' + f2s(maxBalance) + '}')

        if curBalance > maxBalance:
            maxBalance = curBalance

            for kFeature in prange(0, nFeatures):
                objectIdx = sortedDataSet[currentState[kFeature], kFeature]
                maxState[kFeature] = dataSet[objectIdx, kFeature]

        if maxLeft + curBalance <= maxBalance:
            break

    print('Steps: ' + str(iSteps) + ' Getting feature: ' + f2s(getFastTime) + ' Making step: ' + f2s(makeStepTime))
    #print('Max state = ', fv2s(maxState))

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
    c1Banalce, c1diviser = getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget1)

    valuedTarget2 = GetValuedTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    c2Banalce, c2diviser = getMaximumDiviserPerClassFastNumba(dataSet, valuedTarget2)

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser
