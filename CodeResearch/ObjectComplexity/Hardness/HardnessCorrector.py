import numpy as np
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class HardnessCorrector(BaseHardnessCalculator):

    def __init__(self, hardnessCalculator: BaseHardnessCalculator):
        self.hardnessCalculator = hardnessCalculator

    def calculateHardness(self, dataSet, target):
        importance, easyness = self.hardnessCalculator.calculateHardness(dataSet, target)

        importance = self.convertToECDF(importance)
        return importance, easyness


    def convertToECDF(self, importance):
        idx = np.argsort(importance)

        result = np.zeros(len(importance))
        totalElements = len(idx)

        for i in range(len(idx)):
            originalIdx = idx[i]
            result[originalIdx] = i / totalElements

        return result