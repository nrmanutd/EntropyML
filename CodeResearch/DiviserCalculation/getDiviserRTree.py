import bisect

import numpy as np

from CodeResearch.DiviserCalculation.diviserCheckers import calculateDeltaIndependently2
from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget, GetSortedDictList, \
    getIdx, GetPowerOfSet, getPointsUnderDiviser, getBestStartDiviser, prepareDataSet, \
    getPointsIdxUnderDiviser
from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast
from CodeResearch.rademacherHelpers import GetSortedData


def GetRTreeIndex(dataSet, valuedTarget):
    positiveIdx = np.where(valuedTarget > 0)[0]
    negativeIdx = np.where(valuedTarget < 0)[0]

    positive = getIdx(dataSet[positiveIdx, :], positiveIdx)
    negative = getIdx(dataSet[negativeIdx, :], negativeIdx)

    return positive, negative

def updateDiviserConcrete(newDiviser, value, sortedNegIdx, sortedNegValues):
    nFeatures = sortedNegIdx.shape[1]

    for fFeature in range(0, nFeatures):
        if newDiviser[fFeature] < value[fFeature]:
            continue

        idx = bisect.bisect_left(sortedNegValues[:, fFeature], value[fFeature]) - 1
        newDiviser[fFeature] = sortedNegValues[idx, fFeature]

    return newDiviser

def updateDiviser(currentDiviser, idx, sortedNegIdx, sortedNegValues, omitIdx):
    newDiviser = currentDiviser
    nFeatures = sortedNegIdx.shape[1]

    for fFeature in range(0, nFeatures):
        #s1 = time.time()

        idxl = bisect.bisect_left(sortedNegValues[:, fFeature], newDiviser[fFeature])
        idxr = bisect.bisect_right(sortedNegValues[:, fFeature], newDiviser[fFeature])
        #e1 = time.time()
        #timeForSearch += e1 - s1

        if idxr - idxl - len(idx) - len(omitIdx) > 0:
            continue

        #s1 = time.time()
        eSet = set(sortedNegIdx[idxl: idxr, fFeature])
        #print('len eset {:}, len idx {:}, len omit {:}'.format(len(eSet), len(idx), len(omitIdx)))
        #print('max Eset {:}, min idx {:}'.format(max(eSet), min(idx)))
        #print('min Eset {:}, max idx {:}'.format(min(eSet), max(idx)))
        eSet = eSet - idx

        #eSet = set(sortedNegIdx[idxl: idxr, fFeature])
        #for e in idx:
        #    eSet.discard(e)

        #e1 = time.time()
        #timeForInternalCycle += e1 - s1

        lenBefore = len(eSet)
        #s1 = time.time()
        lenAfter = len(eSet.intersection(omitIdx))
        #e1 = time.time()
        #timeForIntersection += e1 - s1

        #s1 = time.time()
        if lenBefore == lenAfter:
            curIdx = idxl - 1

            while sortedNegIdx[curIdx, fFeature] in omitIdx:
                curIdx -= 1

            newDiviser[fFeature] = sortedNegValues[curIdx, fFeature]

        #e1 = time.time()
        #timeForSearchingNext += e1 - s1

    #print('Diviser updated, search{:}\ discard {:}\intersect {:}\\next {:}'.format(timeForSearch, timeForInternalCycle, timeForIntersection, timeForSearchingNext))
    #total = timeForSearch + timeForInternalCycle + timeForIntersection + timeForSearchingNext
    #print('Parts: Diviser updated, search{:.2f}\ discard {:.2f}\intersect {:.2f}\\next {:.2f}'.format(timeForSearch/total, timeForInternalCycle/total,
                                                                                   #timeForIntersection/total,
                                                                                   #timeForSearchingNext/total))
    return newDiviser

def getPointsBeforeDiviserIntersection(sortedPosValues, sortedPosIdx, currentDiviser):
    nFeatures = sortedPosValues.shape[1]
    nObjects = sortedPosValues.shape[0]

    curSet = set()

    for fNumber in range(0, nFeatures):
        iFeature = fNumber
        # s1 = time.time()
        idx = bisect.bisect_right(sortedPosValues[:, iFeature], currentDiviser[iFeature])
        curFeaturesLessIdx = sortedPosIdx[idx: nObjects, iFeature]
        curSet.update(curFeaturesLessIdx)

    return len(curSet)

def getPointsUnderDiviserIntersection(sortedPosValues, sortedPosIdx, currentDiviser):
    nFeatures = sortedPosValues.shape[1]

    featuresIdx = np.zeros(nFeatures)

    for iFeature in range(0, nFeatures):
        idx = bisect.bisect_right(sortedPosValues[:, iFeature], currentDiviser[iFeature])
        featuresIdx[iFeature] = idx

    featuresIdx = np.argsort(featuresIdx)

    idx = bisect.bisect_right(sortedPosValues[:, featuresIdx[0]], currentDiviser[featuresIdx[0]])
    curFeaturesLessIdx = sortedPosIdx[0:idx, featuresIdx[0]]
    curSet = set(curFeaturesLessIdx)

    for fNumber in range(1, len(featuresIdx)):
        iFeature = featuresIdx[fNumber]
        # s1 = time.time()
        idx = bisect.bisect_right(sortedPosValues[:, iFeature], currentDiviser[iFeature])
        curFeaturesLessIdx = sortedPosIdx[0:idx, iFeature]
        curSet = curSet.intersection(curFeaturesLessIdx)

    return len(curSet)

def getUnreachablePositives(negObjects, positiveIdx, basePoint):
    lowest = np.min(negObjects, axis=0)

    return getPointsUnderDiviser(positiveIdx, lowest, basePoint)

def updateDiviserViaRTTree(negativeIdx, negativeObjects, idx, nFeatures):

    for id in idx:
        negativeIdx.delete(id, negativeObjects[id, :])

    return negativeIdx.bounds[nFeatures:2*nFeatures]


def updateDiviserFast(idx, negIdx, nFeatures):

    divisor = np.zeros(nFeatures)
    for iFeature in range(0, nFeatures):
        for iObject in idx:
            del negIdx[iFeature][iObject]

        if len(negIdx[iFeature]) == 0:
            print('Dict in zero capacity')

        divisor[iFeature] = next(iter(negIdx[iFeature].items()))[1]

    return divisor


def updateDiviserGreedy(currentDiviser, nextIdx, negativeIdx, positiveIdx, negativeObjects, omit, bestScore):

    for idx in nextIdx:
        curPoint = negativeObjects[idx, :]
        id = np.where(currentDiviser < curPoint)[0]
        if len(id) > 0:
            print('Error')
        betweenDiviserAndNegative = getPointsIdxUnderDiviser(negativeIdx, currentDiviser, curPoint)

        collection = dict()

        for negCandidate in betweenDiviserAndNegative:
            positivePoints = getPointsIdxUnderDiviser(positiveIdx, currentDiviser, negativeObjects[negCandidate, :])
            collection[negCandidate] = positivePoints

        print(collection)

    return bestScore, currentDiviser, omit


def getMaximumDiviserPerClassRT(dataSet, valuedTarget, subError, balance, diviser):

    nClasses = np.unique(valuedTarget)

    negativeScore = nClasses[0] if nClasses[0] < 0 else nClasses[1]
    positiveScore = nClasses[0] if nClasses[0] > 0 else nClasses[1]

    nFeatures = dataSet.shape[1]

    negativeObjectsIdx = np.where(valuedTarget < 0)[0]
    negativeCount = len(negativeObjectsIdx)
    negativeObjects = dataSet[negativeObjectsIdx, :]
    sortedNegIdx, sortedNegValues = GetSortedData(negativeObjects)

    positiveObjectsIdx = np.where(valuedTarget > 0)[0]
    positiveCount = len(positiveObjectsIdx)
    positiveObjects = dataSet[positiveObjectsIdx, :]
    sortedPosIdx, sortedPosValues = GetSortedData(positiveObjects)

    positiveIdx = getIdx(dataSet[positiveObjectsIdx, :], range(0, len(positiveObjectsIdx)))
    negativeIdx = getIdx(dataSet[negativeObjectsIdx, :], range(0, len(negativeObjectsIdx)))

    basePoint = np.zeros(nFeatures)
    for i in range(0, nFeatures):
        basePoint[i] = min(sortedPosValues[0, i], sortedNegValues[0, i])

    sortedNegDataSet = GetSortedDictList(negativeObjects)
    axesPower = GetPowerOfSet(sortedNegDataSet)
    axesSortedPowerIdx = np.argsort(axesPower)

    bestStartDiviser, bestStartScore = getBestStartDiviser(sortedNegValues, positiveScore, positiveCount, positiveIdx, basePoint)

    if bestStartScore > balance:
        bestDiviser = bestStartDiviser
        bestScore = bestStartScore
    else:
        bestDiviser = diviser
        bestScore = balance

    #if 1 - bestScore < subError:
    #    return bestScore, bestDiviser

    iFeature = axesSortedPowerIdx[-1]

    currentDiviser = bestDiviser.copy()
    currentIdx = sortedNegDataSet[iFeature]

    omit = set()
    for iValue in currentIdx:
        idx = currentIdx[iValue]
        currentDiviser = updateDiviser(currentDiviser, idx, sortedNegIdx, sortedNegValues, omit)
        omit.update(idx)

        nextIdx = currentIdx[-currentDiviser[iFeature]]
        currentScore, currentDiviser, omitted = updateDiviserGreedy(currentDiviser, nextIdx, negativeIdx, positiveIdx, negativeObjects, omit, bestScore)

        omit.update(omitted)

        if currentScore > bestScore:
            bestScore = currentScore
            bestDiviser = currentDiviser.copy()

            if 1 - bestScore < subError:
                return bestScore, bestDiviser

    mb = calculateDeltaIndependently2(dataSet, valuedTarget, bestDiviser)

    if abs(mb - bestScore) > 0.00001:
        print(dataSet)
        print(valuedTarget)
        print('Error!!! Correct != independent: {:} vs {:}, {:}'.format(bestScore, mb, bestDiviser))

        raise ValueError('Error!!! Correct != independent: {:} vs {:}, {:}'.format(bestScore, mb, bestDiviser))

    return bestScore, bestDiviser

def getMaximumDiviserRTree(dataSet, target):
    dataSet, target = prepareDataSet(dataSet, target)
    nClasses, counts = np.unique(target, return_counts=True)

    subError = 0.01

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    fastBalance, fastDiviser = getMaximumDiviserFast(dataSet, target)

    valuedTarget1 = GetValuedTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    c1Banalce, c1diviser = getMaximumDiviserPerClassRT(dataSet, valuedTarget1, subError, fastBalance, fastDiviser)

    if 1 - c1Banalce < subError:
        return c1Banalce, c1diviser

    valuedTarget2 = GetValuedTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    c2Banalce, c2diviser = getMaximumDiviserPerClassRT(dataSet, valuedTarget2, subError, c1Banalce, c1diviser)

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser