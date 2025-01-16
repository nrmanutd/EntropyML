import numpy as np

from CodeResearch.calculateDistributionDelta import getMaximumDiviser, GetSortedData


def getDeltaForFeature(sortedIdx, target, featureState, cClass, cClassKoeff, oClassKoeff, curOmitedObjects):

    newIdx = 0
    delta = 0
    curStatePositive = False

    for iObject in range(featureState, -1, -1):

        if sortedIdx[iObject] in curOmitedObjects:
            continue

        d = cClassKoeff if target[sortedIdx[iObject]] == cClass else oClassKoeff

        if d < 0 and curStatePositive:
            newIdx = sortedIdx[iObject]
            break

        if d > 0 and not curStatePositive:
            curStatePositive = True

        delta += d
        newIdx = sortedIdx[iObject]

    return newIdx, delta


def getNextStep(curState, curOmitedObjects, sortedIdx, sortedValues, target, cClass, cClassKoeff, oClassKoeff):

    nFeatures = sortedValues.shape[1]

    bestDelta = -1000
    bestFeature = -1
    bestNewIdx = -1
    idxToOmit = []

    for iFeature in range(0, nFeatures):
        newIdx, delta = getDeltaForFeature(sortedIdx[:, iFeature], target, curState[iFeature], cClass, cClassKoeff, oClassKoeff, curOmitedObjects)

        if newIdx == -1:
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
            maxState = curState

    return maxBalance, maxState

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

s = np.array([[1,0,0], [0,1,0], [0,0,1], [3, 2, -1], [2, 3, -1], [2, 2, 0]])
c = np.array([1, 1, 1, -1, -1, -1])

diviser, values = getMaximumDiviserProd(s, c)

print('Prod diviser...')
print(diviser)
print(values)

diviser, values = getMaximumDiviser(s, c)

print('Stable diviser...')
print (diviser)
print(values)