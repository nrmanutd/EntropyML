import numpy as np
from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler


class RandomSubsetSampler(BaseSampler):

    def __init__(self, dataset, target, alpha):
        self.target = target
        self.dataset = dataset
        self.alpha = alpha

    def sample(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        n = self.dataset.shape[0]
        indices = np.random.permutation(n)

        test_count = int(n * self.alpha)
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]

        return self.dataset[train_idx], self.target[train_idx], self.dataset[test_idx], self.target[test_idx], np.full(n, 1.0/n)
