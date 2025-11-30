import numpy as np
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator

class ExpandingDatasetHardnessCalculator(BaseHardnessCalculator):
    def calculateHardness(self, dataSet, target):
        xx = np.hstack((dataSet, -dataSet))
        return self.hardnessCalculator.calculateHardness(xx, target)

    def __init__(self, hardnessCalculator: BaseHardnessCalculator):
        self.hardnessCalculator = hardnessCalculator

