import numpy as np

from sortedcontainers import SortedDict
from CodeResearch.DiviserCalculation.diviserHelpers import GetSortedDict, GetValuedTarget
from CodeResearch.DiviserCalculation.getDiviserFast import getMaximumDiviserFast


def getMaximumDiviserStat(dataSet, valuedTarget1, maxDistance):
    sd = GetSortedDict(dataSet)

    possibleValues = []

    totalCombinations = 1

    for featureDict in sd:
        curDict = SortedDict()
        curValue = 0

        totalCombinations *= len(featureDict)

        for iValued in valuedTarget1:
            curValue += -iValued if iValued < 0 else 0

        for iValue in range(0, len(featureDict)):
            curSet = featureDict.peekitem(iValue)

            curSum = 0
            for iCurElement in curSet[1]:
                curSum += -valuedTarget1[iCurElement] if valuedTarget1[iCurElement] < 0 else 0

            if curSum > 0 or iValue == len(featureDict) - 1:
                curDict[-curSet[0]] = curValue

            curValue -= curSum

            if curValue < maxDistance:
                break

        possibleValues.append(curDict)

    curCombintations = 1

    for pv in possibleValues:
        curCombintations *= len(pv)

    print('Total vs cur, {:} vs {:}'.format(totalCombinations, curCombintations))

    return possibleValues

def getMaximumPossibleByAnalysis(dataSet, target):
    nClasses, counts = np.unique(target, return_counts=True)

    if len(nClasses) != 2:
        raise ValueError('Number of classes should be equal to two, instead {:}'.format(len(nClasses)))

    valuedTarget1 = GetValuedTarget(target, nClasses[0], 1 / counts[0], -1 / counts[1])

    maxDistance, values = getMaximumDiviserFast(dataSet, target)
    possibleValues = getMaximumDiviserStat(dataSet, valuedTarget1, maxDistance)

    return possibleValues

