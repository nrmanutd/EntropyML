from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler


class SequentialSampler(BaseSampler):
    def __init__(self, dataset, target, batchsize):
        self.batchsize = batchsize
        self.dataset = dataset
        self.target = target
        self.currentIndex = 0

    def sample(self, seed=None):
        finishIndex = min(len(self.target), self.currentIndex + self.batchsize)

        idx = range(self.currentIndex, finishIndex)
        xBatch = self.dataset[idx]
        yBatch = self.target[idx]

        self.currentIndex = finishIndex

        return [(xBatch, yBatch)]