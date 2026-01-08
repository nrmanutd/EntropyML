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
        finishIndex = min(len(self.target), self.currentIndex + self.batchsize)
        idx = range(self.currentIndex, finishIndex)

        idx = self.originalIdx[idx]
        xBatch = self.dataset[idx]
        yBatch = self.target[idx]

        self.currentIndex = finishIndex

        return [(xBatch, yBatch)]