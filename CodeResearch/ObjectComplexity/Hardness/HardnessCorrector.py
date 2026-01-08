import numpy as np
from scipy.stats import rankdata

from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class HardnessCorrector(BaseHardnessCalculator):

    def __init__(self, hardnessCalculator: BaseHardnessCalculator):
        self.hardnessCalculator = hardnessCalculator

    def calculateHardness(self, dataSet, target, baseDataSet, baseTarget, alpha):
        importance, easiness = self.hardnessCalculator.calculateHardness(dataSet, target, baseDataSet, baseTarget, alpha)

        importance = self.convertToECDFMidRank(importance)
        #easiness = self.convertToECDFMidRank(easiness)

        #easiness = 1 - 2 * np.abs(easiness - 0.5)

        return importance, easiness

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