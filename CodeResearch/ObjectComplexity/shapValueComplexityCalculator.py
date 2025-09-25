import numpy as np
from numba import njit, prange

from CodeResearch.ObjectComplexity.baseComplexityCalculator import BaseComplexityCalculator


class ShapValueComplexityCalculator(BaseComplexityCalculator):
    def __init__(self, dataSet, target, objectIdx):
        self.objectIdx = objectIdx
        self.target = target
        self.dataSet = dataSet

        self.usedObjects = []
        self.accuracy = []

    @staticmethod
    @njit
    def ifObjectIsUnder(diviser, curObject):
        for i in prange(0, len(diviser)):
            if diviser[i] < curObject[i]:
                return False

        return True

    def calculateKS(self, diviser, idx):

        firstClass = self.target[0]

        firstClassCountUnder = 0
        secondClassCountUnder = 0

        firstClassCount = 0
        secondClassCount = 0

        for i in np.arange(len(self.target)):
            if i in idx:
                continue

            curObject = self.dataSet[i, :]

            curObjectIsUnder = self.ifObjectIsUnder(diviser, curObject)

            if self.target[i] == firstClass:
                firstClassCount += 1
            else:
                secondClassCount += 1

            if curObjectIsUnder:
                if self.target[i] == firstClass:
                    firstClassCountUnder += 1
                else:
                    secondClassCountUnder += 1

        return abs(firstClassCountUnder/firstClassCount - secondClassCountUnder/secondClassCount)


    def updateComplexity(self, diviser, classUnderDivisier, idx):

        totalObjects = len(self.target)
        currentUsedObjects = np.zeros(totalObjects)
        currentUsedObjects[idx] = 1
        self.usedObjects.append(currentUsedObjects)

        aggregateAccuracy = 0

        for i in np.arange(totalObjects):
            newObject = self.dataSet[i, :]

            isObjectUnderDiviser = self.ifObjectIsUnder(diviser, newObject)
            objectClass = self.target[i]

            if (isObjectUnderDiviser and objectClass == classUnderDivisier) or (not isObjectUnderDiviser and objectClass != classUnderDivisier):
                aggregateAccuracy += 1 / totalObjects

        self.accuracy.append(aggregateAccuracy)

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

            shapValues[i] = np.sum(accuracy[withObjectIdx]) - np.sum(accuracy[noObjectIdx])

        return shapValues / totalAttempts

    def getObjectsIndex(self):
        return np.array(self.objectIdx)
