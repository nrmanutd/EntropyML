from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class PredefinedWithFixedTestSampler(BaseSampler):
    def __init__(self, dataset, target, xtest, ytest, priorityCalculator: BasePriorityCalculator, logger: BaseLogger):
        self.logger = logger
        self.ytest = ytest
        self.xtest = xtest
        self.priorityCalculator = priorityCalculator
        self.target = target
        self.dataset = dataset

    def sample(self, seed=None):

        priority, probs = self.priorityCalculator.calculatePriority(self.dataset, self.target)

        trainX = []
        trainY = []
        testX = []
        testY = []

        for k in range(len(priority)):
            p = priority[k]
            curTrain_idx = p

            self.logger.logDebug(f'Calculating objects priority for {len(p)} ({len(p) / len(self.target)}%) objects of {k} ({len(priority)}) priorities. test: {len(self.ytest)} objects')

            trainX.append(self.dataset[curTrain_idx])
            trainY.append(self.target[curTrain_idx])
            testX.append(self.xtest)
            testY.append(self.ytest)

        return trainX, trainY, testX, testY, probs