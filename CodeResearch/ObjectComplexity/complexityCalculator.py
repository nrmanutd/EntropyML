import math
import numpy as np
from scipy.stats import entropy

class KSComplexityCalculator:
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

        diffClasses = np.unique(target)
        if len(diffClasses) != 2:
            raise ValueError(f'Number of classes should be 2, instead: {diffClasses}')

        self.firstClass = diffClasses[0]
        self.secondClass = diffClasses[1]

        self.firstClassCount = len(np.where(self.target == self.firstClass)[0])
        self.secondClassCount = len(target) - self.firstClassCount

        self.goodObject = np.zeros(len(target), dtype=np.int32)
        self.totalAttempts = 0

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
        self.totalAttempts += 1

        firstClass = self.firstClass
        secondClass = self.secondClass

        firstClassCount = 0
        secondClassCount = 0

        isObjectOver = np.full(len(self.goodObject), False, dtype=np.bool)
        for i in np.arange(len(isObjectOver)):
            isObjectOver[i] = self.estimateObjectIsOver(self.dataset[i, :], diviser)

            if isObjectOver[i]:
                if self.target[i] == firstClass:
                    firstClassCount += 1
                else:
                    secondClassCount += 1

        goodClassForOver = firstClass if firstClassCount/self.firstClassCount > secondClassCount/self.secondClassCount else secondClass

        for i in np.arange(len(isObjectOver)):
            goodnessEstimation = self.estimateObjectIsGood(isObjectOver[i], self.target[i], goodClassForOver)
            self.goodObject[i] += goodnessEstimation
        pass

    def calculateComplexity(self):
        result = np.zeros(len(self.goodObject))
        for i in np.arange(len(result)):
            p = self.goodObject[i] / self.totalAttempts

            result[i] = entropy([p, 1-p], base=2)

        return result

    def getObjectsFrequences(self):
        return self.goodObject / self.totalAttempts

    def getErrorExpectation(self):
        p = self.getObjectsFrequences()
        errors = np.minimum(p, 1 - p)
        return np.mean(errors)