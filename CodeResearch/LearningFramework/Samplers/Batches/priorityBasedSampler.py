import numpy as np
from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler


class PriorityBasedSampler(BaseSampler):
    def __init__(self, dataset, target, batchsize, priority):

        self.originalIdx = np.argsort(-priority)
        self.batchsize = batchsize
        self.dataset = dataset
        self.target = target
        self.currentIndex = 0

    def sample(self, seed=None):
        batches = []
        n = len(self.target)

        for i in range(0, n, self.batchsize):
            subIdx = self.originalIdx[i:i + self.batchsize]

            x_batch = self.dataset[subIdx]
            y_batch = self.target[subIdx]

            batches.append((x_batch, y_batch))

        return batches