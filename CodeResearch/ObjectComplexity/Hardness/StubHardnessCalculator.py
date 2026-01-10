import numpy as np
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class StubHardnessCalculator(BaseHardnessCalculator):
    def calculateHardness(self, dataSet, target, baseDataSet, baseTarget, alpha):
        return np.zeros(len(target)), np.zeros(len(target))