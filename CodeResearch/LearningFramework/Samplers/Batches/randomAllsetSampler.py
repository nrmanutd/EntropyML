import numpy as np

from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class RandomAllsetSampler(BaseSampler):
    def __init__(self, dataset, target, batchsize, prioritizer: BasePriorityCalculator):
        self.prioritizer = prioritizer
        self.batchsize = batchsize
        self.dataset = dataset
        self.target = target

    def sample(self, seed=None):
        batches = []

        newIdx = self.prioritizer.calculatePriority(self.dataset, self.target)
        n = len(self.target)

        for i in range(0, n, self.batchsize):
            subIdx = newIdx[i:i + self.batchsize]

            x_batch = self.dataset[subIdx]
            y_batch = self.target[subIdx]

            batches.append((x_batch, y_batch))

        return batches
