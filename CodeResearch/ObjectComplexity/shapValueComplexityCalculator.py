import math

import numpy as np
import random

from CodeResearch.DataSeparationFramework.Metrics import BaseMetricCalculator
from CodeResearch.ObjectComplexity.baseComplexityCalculator import BaseComplexityCalculator


class ShapValueComplexityCalculator(BaseComplexityCalculator):
    def __init__(self, dataSet, target, objectIdx, limit, KSCalculator: BaseMetricCalculator):
        self.KSCalculator = KSCalculator
        self.objectIdx = objectIdx
        self.target = target
        self.dataSet = dataSet
        self.limit = limit

        self.shapValues = np.array(len(target))

    def custom_shuffle(self, arr):
        if len(arr) < 2:
            return arr

        first = arr[0]
        last = arr[-1]
        middle = arr[1:-1]

        random.shuffle(middle)

        return [first, last] + middle

    def ifObjectIsUnder(self, diviser, curObject):
        for i in np.arange(len(diviser)):
            if diviser[i] > curObject[i]:
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


    def updateComplexity(self, d, idx):

        shuffledIdx = self.custom_shuffle(idx)
        curIdx = set(idx)

        prevKS, prevDiviser = self.KSCalculator.calculateMetric(self.dataSet[shuffledIdx[0:2], :], self.target[shuffledIdx[0:2], :])
        oosPrevKS = self.calculateKS(prevDiviser, curIdx)

        counter = 1
        for i in np.arange(2, math.ceil(len(shuffledIdx) * self.limit)):

            newIdx = shuffledIdx[0:i+1]
            newKS, d = self.KSCalculator.calculateMetric(self.dataSet[newIdx, :], self.target[newIdx, :])
            oosNewKS = self.calculateKS(d, curIdx)

            if counter == 1:
                self.shapValues[shuffledIdx[i]] = oosNewKS - oosPrevKS
            else:
                self.shapValues[shuffledIdx[i]] = self.shapValues[shuffledIdx[i]] * (counter - 1)/counter + 1 / counter * (oosNewKS - oosPrevKS)

            oosPrevKS = oosNewKS

            counter += 1

        pass

    def getShapValues(self):
        return self.shapValues
