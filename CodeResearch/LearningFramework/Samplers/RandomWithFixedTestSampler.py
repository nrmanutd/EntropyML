import numpy as np

from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min
from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class RandomWithFixedTestSampler(BaseSampler):
    def __init__(self, dataset, target, xtest, ytest, priorityCalculator: BasePriorityCalculator, trainAlpha):
        self.ytest = ytest
        self.xtest = xtest
        self.priorityCalculator = priorityCalculator
        self.target = target
        self.dataset = dataset
        self.trainAlpha = trainAlpha

        if trainAlpha < 0 or trainAlpha > 1:
            raise ValueError("Test part should be between 0 and 1")

    def sample(self, seed=None):
        """
            Разбивает X и y на тест / трейн с заданными долями.
            test_size и train_size — доли от общего числа объектов.
            train_size + test_size <= 1.
            """
        if seed is not None:
            np.random.seed(seed)

        train_idx, remaining_idx = stratified_split_indices_with_min(self.target, self.trainAlpha)

        priority, probs = self.priorityCalculator.calculatePriority(self.dataset[train_idx],
                                                                    self.target[train_idx])

        trainX = []
        trainY = []
        testX = []
        testY = []

        for p in priority:
            curTrain_idx = train_idx[p]

            trainX.append(self.dataset[curTrain_idx])
            trainY.append(self.target[curTrain_idx])
            testX.append(self.xtest)
            testY.append(self.ytest)

        return trainX, trainY, testX, testY, probs