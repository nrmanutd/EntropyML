import numpy as np

from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min
from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class RandomWithFixedLengthSampler(BaseSampler):
    def __init__(self, dataset, target, priorityCalculator: BasePriorityCalculator, trainAlpha, testAlpha):
        self.priorityCalculator = priorityCalculator
        self.target = target
        self.dataset = dataset
        self.testAlpha = testAlpha
        self.trainAlpha = trainAlpha

        if trainAlpha < 0:
            raise ValueError("Train part should be > 0")

        if testAlpha < 0:
            raise ValueError("Test part should be > 0")

        if trainAlpha + testAlpha > 1:
            raise ValueError("Train and test parts should be <= 1 in common")

    def sample(self, seed=None):
        """
            Разбивает X и y на тест / трейн с заданными долями.
            test_size и train_size — доли от общего числа объектов.
            train_size + test_size <= 1.
            """
        if seed is not None:
            np.random.seed(seed)

        test_idx, remaining_idx = stratified_split_indices_with_min(self.target, self.testAlpha)
        n = self.dataset.shape[0]
        n_train = int(n * self.trainAlpha)

        if len(remaining_idx) < n_train:
            raise ValueError("Not enough train objects according to desired part.")

        priority, probs = self.priorityCalculator.calculatePriority(self.dataset[remaining_idx], self.target[remaining_idx])

        trainX = []
        trainY = []
        testX = []
        testY = []

        for p in priority:
            prioritized_idx = remaining_idx[p]

            curTrain = len(p) if n_train == 0 else n_train
            train_idx = prioritized_idx[:curTrain]

            trainX.append(self.dataset[train_idx])
            trainY.append(self.target[train_idx])
            testX.append(self.dataset[test_idx])
            testY.append(self.target[test_idx])

        return trainX, trainY, testX, testY, probs
