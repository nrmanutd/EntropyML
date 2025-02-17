import numpy as np

from CodeResearch.DiviserCalculation.diviserHelpers import GetValuedTarget, GetSortedDict, prepareDataSet


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

            if curDelta < 0 and alreadyPositive:#todo: fix bug for different objects on same values. Not finishing...
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

def getMaximumDiviserFast(dataSet, target):
    dataSet = prepareDataSet(dataSet)
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))
        #print('Error!!! Number of classes should be equal to two, instead ', len(nClasses))

    valuedTarget1 = GetValuedTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])
    c1Banalce, c1diviser = getMaximumDiviserPerClassFast(dataSet, valuedTarget1)

    valuedTarget2 = GetValuedTarget(target, nClasses[1], 1 / counts[1], -1 / counts[0])
    c2Banalce, c2diviser = getMaximumDiviserPerClassFast(dataSet, valuedTarget2)

    if c1Banalce > c2Banalce:
        return c1Banalce, c1diviser

    return c2Banalce, c2diviser