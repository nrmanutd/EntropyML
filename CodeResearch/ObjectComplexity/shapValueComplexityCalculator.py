import math

import numpy as np
import scipy.stats
from numba import njit, prange
from scipy import stats

from CodeResearch.Helpers.commonHelpers import calculateNormalityTest, calculateNormalityWithMeanTest
from CodeResearch.ObjectComplexity.baseComplexityCalculator import BaseComplexityCalculator

@njit
def ifObjectIsUnder(diviser, curObject):
    for i in np.arange(len(diviser)):
        if diviser[i] < curObject[i]:
            return False

    return True

class ShapValueComplexityCalculator(BaseComplexityCalculator):
    def __init__(self, dataSet, target, objectIdx):
        self.objectIdx = objectIdx
        self.target = target
        self.dataSet = dataSet

        self.usedObjects = []
        self.accuracy = []

    @staticmethod
    @njit
    def calculateAggregateScore(classUnderDivisier, currentUsedObjects, diviser, totalObjects, dataSet, target):
        positiveObjects = 0
        negativeObjects = 0
        positiveObjectsCount = 0
        negativeObjectsCount = 0
        for i in prange(0, totalObjects):
            if currentUsedObjects[i] == 1:
                continue

            newObject = dataSet[i, :]

            isObjectUnderDiviser = ifObjectIsUnder(diviser, newObject)
            objectClass = target[i]

            if objectClass == classUnderDivisier:
                positiveObjectsCount += 1
                positiveObjects += (1 if isObjectUnderDiviser else 0)
            else:
                negativeObjectsCount += 1
                negativeObjects += (1 if isObjectUnderDiviser else 0)
                # negativeObjects += (0 if isObjectUnderDiviser else 1)

        aggregateAccuracy = positiveObjects / positiveObjectsCount - negativeObjects / negativeObjectsCount
        return aggregateAccuracy

    def updateComplexity(self, diviser, classUnderDivisier, idx):

        totalObjects = len(self.target)
        currentUsedObjects = np.zeros(totalObjects)
        currentUsedObjects[idx] = 1
        self.usedObjects.append(currentUsedObjects)

        aggregateAccuracy = ShapValueComplexityCalculator.calculateAggregateScore(classUnderDivisier, currentUsedObjects, diviser, totalObjects, self.dataSet, self.target)
        #aggregateAccuracy = (positiveObjects + negativeObjects) / (positiveObjectsCount + negativeObjectsCount)
        self.accuracy.append(aggregateAccuracy)

    @staticmethod
    def checkConcreteClassIsNormal(shapValues, target, c):
        cIdx = np.where(target == c)[0]
        shaps = shapValues[cIdx]
        noNanShaps = shaps[np.where(~np.isnan(shaps))[0]]

        return calculateNormalityWithMeanTest(noNanShaps)

    @staticmethod
    def checkIfClassesAreNormal(shapValues, target):
        classes = np.unique(target)
        alpha = 0.01

        pv1 = ShapValueComplexityCalculator.checkConcreteClassIsNormal(shapValues, target, classes[0])
        pv2 = ShapValueComplexityCalculator.checkConcreteClassIsNormal(shapValues, target, classes[1])

        print(f'pv1 = {pv1}, pv2 = {pv2}, alpha = {alpha}')
        if pv1 > alpha and pv2 > alpha:
            return True

        return False

    @staticmethod
    def calculateObjectImportance(shapValues):

        return shapValues

        delta = 0.1

        noNanIndexes = np.where(~np.isnan(shapValues))[0]
        shaps = shapValues[noNanIndexes]

        std = np.std(shaps)

        sortIndex = np.argsort(-shaps)
        totalObjects = len(sortIndex)

        threshold = math.sqrt(math.log(2 / delta) / (2 * totalObjects))
        curCumulative = 1

        norm_dist = stats.norm(loc=0, scale=std)
        selectedObjects = np.zeros(totalObjects)
        distributionWeight = 1 / totalObjects

        epsilon = 0.1 / totalObjects
        ad = np.zeros(totalObjects)

        for i in range(totalObjects):
            value = shaps[sortIndex[i]]
            normDistributionValue = norm_dist.cdf(value)

            if normDistributionValue < epsilon or (1 - normDistributionValue) < epsilon:
                normalCoefff = (epsilon * (1 - epsilon))
            else:
                normalCoefff = normDistributionValue * (1 - normDistributionValue)

            ad[sortIndex[i]] = (curCumulative - normDistributionValue) ** 2 / normalCoefff

            #print(f'{abs(curCumulative - normDistributionValue)} vs {threshold}')
            #if abs(curCumulative - normDistributionValue) > threshold:
            #    if (curCumulative > normDistributionValue) and abs(curCumulative - distributionWeight - normDistributionValue) < abs(curCumulative - normDistributionValue):
            #        selectedObjects[sortIndex[i]] = 1

            curCumulative -= distributionWeight

        adSortedIdx = np.argsort(-ad)
        selectedObjects[adSortedIdx[0]] = 1
        prevRatio = ad[adSortedIdx[1]] / ad[adSortedIdx[0]]

        for i in range(1, totalObjects - 1):
            ratio = ad[adSortedIdx[i + 1]] / ad[adSortedIdx[i]]
            if abs(ratio / prevRatio - 1) < 0.01:
                break

            selectedObjects[adSortedIdx[i]] = 1
            prevRatio = ratio

        #print(ad[np.argsort(-ad)])
        pValuesToReturn = np.zeros(len(shapValues))
        for i in prange(totalObjects):
            if selectedObjects[i] == 1:
                pValuesToReturn[noNanIndexes[i]] = shaps[i]

        return pValuesToReturn

    def getShapValues(self):

        totalObjects = len(self.target)
        totalAttempts = len(self.accuracy)

        shapValues = np.zeros(totalObjects)
        accuracy = np.array(self.accuracy)

        for i in np.arange(totalObjects):
            withObjectIdx = []
            noObjectIdx = []

            for j in np.arange(totalAttempts):
                if self.usedObjects[j][i] == 1:
                    withObjectIdx.append(j)
                else:
                    noObjectIdx.append(j)

            shapValues[i] = np.mean(accuracy[withObjectIdx]) - np.mean(accuracy[noObjectIdx])

        if ShapValueComplexityCalculator.checkIfClassesAreNormal(shapValues, self.target):
            pValue = np.zeros(len(shapValues))
        else:
            pValue = ShapValueComplexityCalculator.calculateObjectImportance(shapValues)

        #return shapValues, pValue
        return shapValues, shapValues

    def getObjectsIndex(self):
        return np.array(self.objectIdx)
