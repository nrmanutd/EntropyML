import numpy as np
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class StandardPriorityCalculator(BasePriorityCalculator):
    def calculatePriority(self, dataSet, target):
        n = len(target)
        return [range(n)], [np.full(n, 1.0/n)]