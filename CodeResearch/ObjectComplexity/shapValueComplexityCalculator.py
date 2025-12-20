import numpy as np
from numba import njit, prange

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

        self.goodObject = np.zeros(len(target), dtype=np.int32)
        self.objectAttempts = np.zeros(len(target), dtype=np.int32)

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

        aggregateAccuracy = positiveObjects / positiveObjectsCount - negativeObjects / negativeObjectsCount
        return aggregateAccuracy

    def updateInstanceGoodness(self, diviser, classUnderDiviser, idx):
        idxToSkip = set(idx)

        for i in np.arange(len(self.target)):
            if i in idxToSkip:
                continue

            newObject = self.dataSet[i, :]

            isObjectUnderDiviser = ifObjectIsUnder(diviser, newObject)
            objectClass = self.target[i]

            if isObjectUnderDiviser and objectClass == classUnderDiviser:
                self.goodObject[i] += 1

            if not isObjectUnderDiviser and objectClass != classUnderDiviser:
                self.goodObject[i] += 1

            self.objectAttempts[i] += 1

        pass

    def getInstanceHardness(self):
        return np.array([self.goodObject[i] / self.objectAttempts[i] for i in range(len(self.target))])

    def updateComplexity(self, diviser, classUnderDivisier, idx):

        totalObjects = len(self.target)
        currentUsedObjects = np.zeros(totalObjects)
        currentUsedObjects[idx] = 1
        self.usedObjects.append(currentUsedObjects)

        aggregateAccuracy = ShapValueComplexityCalculator.calculateAggregateScore(classUnderDivisier, currentUsedObjects, diviser, totalObjects, self.dataSet, self.target)

        self.accuracy.append(aggregateAccuracy)

        self.updateInstanceGoodness(diviser, classUnderDivisier, idx)

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

        instanceHardness = self.getInstanceHardness()
        return shapValues, instanceHardness

    def getObjectsIndex(self):
        return np.array(self.objectIdx)
