import numpy as np

from CodeResearch.calculateDistributionDelta import getMaximumDiviser, GetSortedData

def getMaximumDiviserPerClass(sortedIdx, sortedValues, target, cClass, cClassKoeff, oClassKoeff):
    nObjects = sortedValues.shape[0]
    nFeatures = sortedValues.shape[1]

    curSet = set()
    totalBalance = 0

    diviser = np.zeros(nFeatures)

    for iFeature in range(0, nFeatures):
        curBalance = totalBalance
        maxBalance = totalBalance
        curObject = nObjects

        for iObject in range(nObjects - 1, 0, -1):

            if iFeature > 0 and sortedIdx[iObject, iFeature] not in curSet:
                continue

            delta = cClassKoeff if target[sortedIdx[iObject, iFeature]] == cClass else oClassKoeff
            curBalance -= delta

            if abs(curBalance) > abs(maxBalance):
                maxBalance = curBalance
                curObject = iObject

        # print('Max diviser]: {:} of {:}, value]: {:}'.format(curObject - 1, nObjects, sortedValues[curObject - 1, iFeature]))

        curIdxes = set(sortedIdx[0:curObject, iFeature])
        diviser[iFeature] = sortedValues[curObject - 1, iFeature]

        if iFeature == 0:
            curSet = curIdxes
        else:
            curSet = curSet.intersection(curIdxes)

        totalBalance = maxBalance

    return totalBalance, diviser

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