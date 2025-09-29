import numpy as np
from numba import njit, prange
from scipy import stats

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
    def calculateObjectImportance(shapValues):
        std = np.std(shapValues)

        sortIndex = np.argsort(shapValues)
        curCumulative = 0
        totalObjects = len(sortIndex)
        deltas = np.zeros(totalObjects)

        totalNotNaN = totalObjects - np.isnan(shapValues).sum()

        norm_dist = stats.norm(loc=0, scale=std)

        for i in np.arange(len(sortIndex)):
            if np.isnan(shapValues[sortIndex[i]]):
                continue

            curCumulative += 1/totalNotNaN
            normDistribution = norm_dist.cdf(shapValues[sortIndex[i]])

            deltas[sortIndex[i]] = -abs(normDistribution - curCumulative)

        deltasIdx = np.argsort(deltas)
        maximumElement = max(1, totalObjects // 10)

        pValues = np.zeros(totalObjects)

        for i in range(maximumElement):
            curIdx = deltasIdx[i]
            if deltas[curIdx] == 0:
                continue

            pValues[curIdx] = 1

        ks_statistic_before, p_value_before = stats.shapiro(shapValues)
        ks_statistic_after, p_value_after = stats.shapiro(shapValues[maximumElement::])

        print(f'Before: {ks_statistic_before:.4f}, {p_value_before:.4f}')
        print(f'After: {ks_statistic_after:.4f}, {p_value_after:.4f}')

        return pValues

    def getShapValues(self):

        totalObjects = len(self.target)
        totalAttempts = len(self.accuracy)

        shapValues = np.zeros(totalObjects)
        pvalue = np.zeros(totalObjects)
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

        pvalue = ShapValueComplexityCalculator.calculateObjectImportance(shapValues)

        return shapValues, pvalue

    def getObjectsIndex(self):
        return np.array(self.objectIdx)
