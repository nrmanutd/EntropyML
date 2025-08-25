import bisect

import numpy as np

from CodeResearch.DiviserCalculation.diviserCheckers import calculateDeltaIndependently2
from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget
from CodeResearch.Helpers.rademacherHelpers import GetSortedData

def initOmited(nObjects):

    omited = dict()
    for iObject in range(0, nObjects):
        omited[iObject] = set()

    return omited


def checkObjectIsUnder(iObject, curFeature, localState, dataSet):

    resIdxes = set()

    for iFeature in range(0, curFeature):
        if dataSet[iObject, iFeature] > localState[iFeature]:
            resIdxes.add(iFeature)

    return resIdxes


def calcBalanceForNewState(newState, sortedValues, sortedSetIdx, valuedTarget):

    nObjects = sortedValues.shape[0]
    nFeatures = len(newState)

    curSet = set(range(0, nObjects))

    for iFeature in range(0, nFeatures):
        v = newState[iFeature]
        idx = bisect.bisect_right(sortedValues[:, iFeature], v)
        curFeaturesLessIdx = sortedSetIdx[0:idx, iFeature]
        curSet = curSet.intersection(curFeaturesLessIdx)

    res = 0.0
    for idx in curSet:
        res += valuedTarget[idx]

    return sum(valuedTarget) - res #todo optimize, don't calculate every time sum

def updateForPositive(iObject, localState, sortedValues, sortedIdx, dataSet, idxesObjectIsOver, valuedTarget):

    nObjects = dataSet.shape[0]
    toOmit = dict()
    newState = localState.copy()

    for iiFeature in idxesObjectIsOver:
        curBorder = localState[iiFeature]
        lowerIdx = bisect.bisect_right(sortedValues[:, iiFeature], curBorder)#отдельно проверить логику на -1 и на конец массива

        objectDetected = False
        startedWorseObjects = False

        #curOmit = set()
        conditionedBreak = False

        for iIdx in range(lowerIdx, nObjects):
            #curOmit.add(iIdx)

            originalIndex = sortedIdx[iIdx, iiFeature]

            if objectDetected and not startedWorseObjects:
                if valuedTarget[originalIndex] < 0:
                    startedWorseObjects = True

                    if iIdx == nObjects - 1: #когда последний из наказанных оказался в начале
                        newState[iiFeature] = sortedValues[iIdx, iiFeature]
                        break

                    continue

            if objectDetected and startedWorseObjects:
                if valuedTarget[originalIndex] > 0 and sortedValues[iIdx, iiFeature] != sortedValues[iIdx - 1, iiFeature]:
                    newState[iiFeature] = dataSet[sortedIdx[iIdx - 1, iiFeature], iiFeature]
                    conditionedBreak = True
                    break

            if originalIndex == iObject:
                objectDetected = True

        if not conditionedBreak and startedWorseObjects:
            newState[iiFeature] = sortedValues[nObjects - 1, iiFeature]

    newBalance = calcBalanceForNewState(newState, sortedValues, sortedIdx, valuedTarget)
    return newBalance, newState

def updateForObject(iObject, curFeature, localState, localBalance, valuedTarget, sortedValues, sortedIdx, dataSet):
    delta = valuedTarget[iObject]

    idxesObjectIsOver = checkObjectIsUnder(iObject, curFeature, localState, dataSet)
    objectIsUnder = len(idxesObjectIsOver) == 0

    if delta > 0:
        if objectIsUnder:
            return localBalance + delta, localState
        else:
            newBalance, newState = updateForPositive(iObject, localState, sortedValues, sortedIdx, dataSet, idxesObjectIsOver, valuedTarget)
            return newBalance, newState
    if delta < 0:
        if objectIsUnder:
            return localBalance + delta, localState #доработать лишний шаг сделать, по спуску
        else:
            return localBalance, localState

    if delta == 0:
        raise ValueError('Delta shouldnt be equal to zero!')

def getOptimalState(curObjects, curFeature, localState, localBalance, valuedTarget, sortedValues, sortedIdx, dataSet):

    alreadyProcessed = False

    for iObject in curObjects:
        if alreadyProcessed and valuedTarget[iObject] > 0:
            continue

        localBalance, localState = updateForObject(iObject, curFeature, localState, localBalance, valuedTarget, sortedValues, sortedIdx, dataSet)

        if valuedTarget[iObject] > 0:
            alreadyProcessed = True

    return localBalance, localState


def updateDiviser(iFeature, valuedTarget, sortedValues, sortedIdx, curState, curBalance, dataSet, maxLeft):
    nObjects = len(valuedTarget)
    newBalance = curBalance
    newState = curState.copy()

    localBalance = curBalance
    localState = curState

    iIndex = nObjects - 1
    curObjects = set()

    while True:
        if iIndex == 0 or sortedValues[iIndex, iFeature] != sortedValues[iIndex - 1, iFeature]:
            curObjects.add(sortedIdx[iIndex, iFeature])
            curValue = valuedTarget[sortedIdx[iIndex, iFeature]]
            maxLeft -= curValue if curValue > 0 else 0

            localState[iFeature] = sortedValues[max(iIndex - 1, 0), iFeature]#отработка кейса последнего элемента
            localBalance, localState = getOptimalState(curObjects, iFeature, localState, localBalance, valuedTarget, sortedValues,
                                                       sortedIdx, dataSet)#текущий объект iIndex - Это кандидат на удаление

            if localBalance > newBalance:
                newBalance = localBalance
                newState = localState.copy()

            if iIndex == 0 or localBalance + maxLeft <= newBalance:
                break

            curObjects = set()
            iIndex -= 1
            continue

        curObjects.add(sortedIdx[iIndex, iFeature])
        curValue = valuedTarget[sortedIdx[iIndex, iFeature]]
        maxLeft -= curValue if curValue > 0 else 0

        iIndex -= 1
        continue

    return newState, newBalance

def getMaximumDiviserPerClassCorrect(dataSet, valuedTarget1):
    sortedIdx, sortedValues = GetSortedData(dataSet)
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]

    curBalance = sum(valuedTarget1)
    maxLeft = 0
    for iObject in range(0, nObjects):
        curValue = valuedTarget1[iObject]
        maxLeft += curValue if curValue > 0 else 0

    curState = np.zeros(nFeatures)
    for iFeature in range(0, nFeatures):
        curState[iFeature] = sortedValues[nObjects - 1, iFeature]

    for iFeature in range(0, nFeatures):
        curState, curBalance = updateDiviser(iFeature, valuedTarget1, sortedValues, sortedIdx, curState, curBalance, dataSet, maxLeft)

    mb = calculateDeltaIndependently2(dataSet, valuedTarget1, curState)

    if abs(mb - curBalance) > 0.00001:
        print(dataSet)
        print(valuedTarget1)
        print('Error!!! Correct != independent: {:} vs {:}, {:}'.format(curBalance, mb, curState))

        raise ValueError('Error!!! Correct != independent: {:} vs {:}, {:}'.format(curBalance, mb, curState))

    return abs(curBalance), curState

def getMaximumDiviserCorrect(dataSet, target):
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    valuedTarget1 = GetValuedTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    c1Banalce, c1diviser = getMaximumDiviserPerClassCorrect(dataSet, valuedTarget1)

    #print(c1Banalce, c1diviser)
    #print('Second attempt...')

    valuedTarget2 = GetValuedTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    c2Banalce, c2diviser = getMaximumDiviserPerClassCorrect(dataSet, valuedTarget2)

    #print(c2Banalce, c2diviser)
    #print('Finished...')

    if c1Banalce >= c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser