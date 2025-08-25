import numpy as np

from CodeResearch.DiviserCalculation.diviserCheckers import calculateDeltaIndependently
from CodeResearch.Helpers.rademacherHelpers import GetSortedData



def getMaximumDiviser(dataSet, target):
    nObjects = dataSet.shape[0]
    nFeatures = dataSet.shape[1]
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    sortedIdx, sortedValues = GetSortedData(dataSet)

    curSet = set()
    totalBalance = 0

    diviser = np.zeros(nFeatures)
    diviserIdx = np.zeros(nFeatures, dtype=int)

    for iFeature in range(0, nFeatures):
        curBalance = totalBalance
        maxBalance = totalBalance
        curObject = nObjects

        for iObject in range(nObjects - 1, 0, -1):

            if iFeature > 0 and sortedIdx[iObject, iFeature] not in curSet:
                continue

            delta = 1/counts[0] if target[sortedIdx[iObject, iFeature]] == nClasses[0] else -1/counts[1]
            curBalance -= delta

            if abs(curBalance) > abs(maxBalance):
                maxBalance = curBalance
                curObject = iObject

        #print('Max diviser]: {:} of {:}, value]: {:}'.format(curObject - 1, nObjects, sortedValues[curObject - 1, iFeature]))

        curIdxes = set(sortedIdx[0:curObject, iFeature])
        diviser[iFeature] = sortedValues[curObject - 1, iFeature]
        diviserIdx[iFeature] = curObject - 1

        if iFeature == 0:
            curSet = curIdxes
        else:
            curSet = curSet.intersection(curIdxes)

        totalBalance = maxBalance

    mb1 = calculateDeltaIndependently(diviserIdx, sortedIdx, target, nClasses[0], 1 / counts[0], -1 / counts[1])
    mb2 = calculateDeltaIndependently(diviserIdx, sortedIdx, target, nClasses[1], 1 / counts[1], -1 / counts[0])

    if abs(abs(mb1) - abs(totalBalance)) > 0.00001 or abs(abs(mb2) - abs(totalBalance)) > 0.00001:
        print(
            '!!!Error!!! Difference occured between actual and independent calculation: a {:}, i1 {:}, i2: {:}'.format(maxBalance,
                                                                                                             mb1, mb2))

    return totalBalance, diviser