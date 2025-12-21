import numpy as np
from scipy.stats import rankdata

from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class HardnessCorrector(BaseHardnessCalculator):

    def __init__(self, hardnessCalculator: BaseHardnessCalculator):
        self.hardnessCalculator = hardnessCalculator

    def calculateHardness(self, dataSet, target):
        importance, easyness = self.hardnessCalculator.calculateHardness(dataSet, target)

        #importance = self.convertToPositive(importance)
        #importance = self.convertToECDF(importance)
        #importance = self.convertToUniform(importance)
        importance = self.convertToECDF(importance)
        easyness = self.convertToECDF(easyness)

        return importance, easyness

    def convertToPositive(self, importance):
        minimum = min(importance)
        return importance - minimum

    def convertToUniform(self, importance):
        minimum = min(importance)
        maximum = max(importance)

        return (importance - minimum) / (maximum - minimum)

    def convertToECDF(self, importance):
        idx = np.argsort(importance)

        result = np.zeros(len(importance))
        totalElements = len(idx)

        for i in range(len(idx)):
            originalIdx = idx[i]
            result[originalIdx] = i / totalElements

        return result

    def convertToECDFMidRank(self, vector):
        r = rankdata(vector, method="average")
        return (r - 0.5) / len(vector)