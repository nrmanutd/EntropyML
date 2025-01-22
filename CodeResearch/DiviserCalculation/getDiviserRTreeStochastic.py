import math
import time
from random import randrange

import numpy as np

from CodeResearch.DiviserCalculation.diviserCheckers import calculateDeltaIndependently2
from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget, getIdx, GetSortedDictList, GetPowerOfSet, \
    getBestStartDiviser, getPointsUnderDiviser, prepareDataSet
from CodeResearch.DiviserCalculation.getDiviserRTree import updateDiviser
from CodeResearch.rademacherHelpers import GetSortedData

def getTopBorderCandidates(currentDiviser, sortedNegDataSet, topPoints, omit):
    nFeatures = len(currentDiviser)

    total = []
    for iFeature in range(0, nFeatures):
        curElemIdx = sortedNegDataSet[iFeature][-currentDiviser[iFeature]]
        total += list(curElemIdx)

    total = set(total)
    total -= omit

    borderIdxs, borderCounts = np.unique(list(total), return_counts=True)
    borderCountsIdx = np.argsort(borderCounts)

    parts = borderCounts / sum(borderCounts)
    entropy = - np.sum(parts * np.log(parts))

    res = borderIdxs[borderCountsIdx][len(borderIdxs) - topPoints:len(borderIdxs)]
    return res, entropy, len(borderCounts)

def getMaximumDiviserPerClassRTStochastic(dataSet, valuedTarget, subError):

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

    bestStartDiviser, bestStartScore, possibleBestScore = getBestStartDiviser(sortedNegValues, positiveScore, positiveCount, positiveIdx, basePoint)

    bestDiviser = bestStartDiviser
    bestScore = bestStartScore

    if 1 - bestScore < subError:
        return bestScore, bestDiviser, possibleBestScore

    topPoints = 1
    nAttempts = 1

    for iAttempt in range(0, nAttempts):
        #if iAttempt%10 == 0:
        #    print('#Attempt (Stochastic)', iAttempt)

        currentDiviser = bestStartDiviser.copy()
        omit = set()
        curIterations = 0

        candidates = 0
        updating = 0
        under = 0

        while curIterations < negativeCount:
            #s1 = time.time()
            objectsToOmit, entropy, diffObjects = getTopBorderCandidates(currentDiviser, sortedNegDataSet, topPoints, omit)
            #e1 = time.time()
            #candidates += (e1 - s1)

            if diffObjects == 0:
                print('Error!')

            if abs((entropy - diffObjects) / diffObjects) < 0.2:
                print('Exit by entropy...e = {:}, d = {:}'.format(entropy, diffObjects))
                break

            omittingObject = objectsToOmit[randrange(min(topPoints, len(objectsToOmit)))]

            #s1 = time.time()
            currentDiviser = updateDiviser(currentDiviser, {omittingObject}, sortedNegIdx, sortedNegValues, omit)
            #e1 = time.time()
            #updating += (e1 - s1)

            #s1 = time.time()
            positivePoints = getPointsUnderDiviser(positiveIdx, currentDiviser, basePoint)
            #e1 = time.time()
            #under += e1 - s1

            negativePoints = negativeCount - len(omit) - 1

            currentScore = (positiveCount - positivePoints) * positiveScore + (
                        negativeCount - negativePoints) * negativeScore

            currentPossibleBestScore = currentScore + positivePoints * positiveScore

            if currentPossibleBestScore <= bestScore:
                break

            # добавить проверку на раннюю остановку
            if currentScore > bestScore:
                bestScore = currentScore
                bestDiviser = currentDiviser.copy()
                possibleBestScore = currentPossibleBestScore

                if 1 - bestScore < subError:
                    return bestScore, bestDiviser, possibleBestScore

            omit.add(omittingObject)
            curIterations += 1
            #e1 = time.time()

        #print('Attempt #{:}, current score: {:} / best score: {:}'.format(iAttempt, currentScore, bestScore))
        #print('Attempt #{:}, cand: {:} / updating: {:} / under: {:}'.format(iAttempt, candidates, updating, under))

    #mb = calculateDeltaIndependently2(dataSet, valuedTarget, bestDiviser)

    #if abs(mb - bestScore) > 0.00001:
    #    print(dataSet)
    #    print(valuedTarget)
    #    print('Error!!! Correct != independent: {:} vs {:}, {:}'.format(bestScore, mb, bestDiviser))

    #    raise ValueError('Error!!! Correct != independent: {:} vs {:}, {:}'.format(bestScore, mb, bestDiviser))

    return bestScore, bestDiviser, possibleBestScore

def getMaximumDiviserRTreeStochastic(dataSet, target):

    dataSet, target = prepareDataSet(dataSet, target)

    nClasses, counts = np.unique(target, return_counts=True)

    subError = 0.01

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    valuedTarget1 = GetValuedTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    c1Banalce, c1diviser, c1PossibleBest = getMaximumDiviserPerClassRTStochastic(dataSet, valuedTarget1, subError)

    valuedTarget2 = GetValuedTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    c2Banalce, c2diviser, c2PossibleBest = getMaximumDiviserPerClassRTStochastic(dataSet, valuedTarget2, subError)

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser, c1PossibleBest

    return c2Banalce, c2diviser, c2PossibleBest