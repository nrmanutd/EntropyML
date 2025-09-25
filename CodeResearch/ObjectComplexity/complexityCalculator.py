import math
import numpy as np
from scipy.stats import entropy

from CodeResearch.ObjectComplexity.baseComplexityCalculator import BaseComplexityCalculator


class KSComplexityCalculator(BaseComplexityCalculator):
    def __init__(self, dataset, target, objectsIdx=None):
        self.dataset = dataset
        self.target = target

        self.objectsIdx = objectsIdx if objectsIdx is not None else np.arange(len(target))

        diffClasses = np.unique(target)
        if len(diffClasses) != 2:
            raise ValueError(f'Number of classes should be 2, instead: {diffClasses}')

        self.firstClass = diffClasses[0]
        self.secondClass = diffClasses[1]

        self.firstClassCount = len(np.where(self.target == self.firstClass)[0])
        self.secondClassCount = len(target) - self.firstClassCount

        self.goodObject = np.zeros(len(target), dtype=np.int32)
        self.objectAttempts = np.zeros(len(target), dtype=np.int32)

    def updateComplexity(self, d, c, idx):
        self.addComplexityOutOfIdx(d, idx)
        return self.calculateKSOutOfIdx(d, idx)

    @staticmethod
    def estimateObjectIsOver(obj, div):
        nFeatures = len(obj)

        for j in np.arange(nFeatures):
            if obj[j] > div[j]:
                return True

        return False

    @staticmethod
    def estimateObjectIsGood(objectIsOver, objectClass, goodClassForOver):
        if objectIsOver and objectClass == goodClassForOver:
            return 1

        if (not objectIsOver) and objectClass != goodClassForOver:
            return 1

        return 0

    def addComplexity(self, diviser):
        self.addComplexityOutOfIdx(diviser, [])

    def addComplexityOutOfIdx(self, diviser, idx):
        idxToSkip = set(idx)

        firstClass = self.firstClass
        secondClass = self.secondClass

        firstClassCount = 0
        secondClassCount = 0

        firstClassOver = 0
        secondClassOver = 0

        isObjectOver = np.full(len(self.goodObject), False, dtype=np.bool)
        for i in np.arange(len(isObjectOver)):
            if i in idxToSkip:
                continue

            isObjectOver[i] = self.estimateObjectIsOver(self.dataset[i, :], diviser)

            if self.target[i] == firstClass:
                firstClassCount += 1
            else:
                secondClassCount += 1

            if isObjectOver[i]:
                if self.target[i] == firstClass:
                    firstClassOver += 1
                else:
                    secondClassOver += 1

        goodClassForOver = firstClass if firstClassOver/firstClassCount > secondClassOver/secondClassCount else secondClass

        for i in np.arange(len(isObjectOver)):
            if i in idxToSkip:
                continue

            goodnessEstimation = self.estimateObjectIsGood(isObjectOver[i], self.target[i], goodClassForOver)
            self.goodObject[i] += goodnessEstimation
            self.objectAttempts[i] += 1
        pass

    def calculateKSOutOfIdx(self, diviser, idx):
        idxToSkip = set(idx)
        firstClass = self.firstClass

        firstClassOver = 0
        secondClassOver = 0

        firstClassCount = 0
        secondClassCount = 0

        for i in np.arange(len(self.target)):
            if i in idxToSkip:
                continue

            isObjectOver = self.estimateObjectIsOver(self.dataset[i, :], diviser)
            if isObjectOver:
                if self.target[i] == firstClass:
                    firstClassOver += 1
                else:
                    secondClassOver += 1

            if self.target[i] == firstClass:
                firstClassCount += 1
            else:
                secondClassCount += 1

        balance = firstClassOver/firstClassCount - secondClassOver/secondClassCount
        return abs(balance)

    def calculateComplexity(self):
        result = np.zeros(len(self.goodObject))
        for i in np.arange(len(result)):
            p = self.goodObject[i] / self.objectAttempts[i]

            result[i] = entropy([p, 1-p], base=2)

        return result

    def getObjectsFrequences(self):
        return np.array([self.goodObject[i] / self.objectAttempts[i] for i in range(len(self.target))])

    def getErrorExpectation(self):
        p = self.getObjectsFrequences()
        p = p[~np.isnan(p)]
        errors = np.minimum(p, 1 - p)
        return np.mean(errors)

    def getObjectsIndex(self):
        return np.array(self.objectsIdx)