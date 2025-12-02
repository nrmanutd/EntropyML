import numpy as np

from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class RandomComplexityBasedPrioritizer(BasePriorityCalculator):
    def __init__(self, probs, seed=None):
        self.seed = seed
        self.probs = probs

        if not np.isclose(np.asarray(probs).sum(), 1.0):
            raise ValueError('Probs should be in sum = 1.0')

    def calculatePriority(self, dataSet, target):
        weights = np.asarray(self.probs)

        if self.seed is not None:
            np.random.seed(self.seed)

        # случайные ключи по формуле Эфраимида–Спиракиса
        keys = np.random.random(len(weights)) ** (1.0 / weights)

        # сортировка — чем меньше key, тем выше позиция
        return np.argsort(-keys), self.probs
