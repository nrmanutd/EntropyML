import numpy as np

from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class RandomPrioritizer(BasePriorityCalculator):
    def __init__(self, seed=None):
        self.seed = seed

    def calculatePriority(self, dataSet, target):

        if self.seed is not None:
            np.random.seed(self.seed)

        n = len(target)
        newIdx = np.random.permutation(n)

        return newIdx, np.full(n, 1.0/n)