import time

import numpy as np

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler


class GeneralLearningEstimator:

    def __init__(self, iterations, logger: BaseLogger):
        self.logger = logger
        self.iterations = iterations

    def estimateLearner(self, sampler: BaseSampler, learner: BaseLearner):
        losses = []

        for i in range(self.iterations):

            if i %10 == 0:
                self.logger.logDebug(f'Iteration {i} of {self.iterations}')

            self.logger.logDebug('Sampling...')
            trainX, trainY, testX, testY, probs = sampler.sample()
            self.logger.logDebug('Training model...')
            model = learner.train(trainX, trainY, probs)
            self.logger.logDebug(f'Testing {len(model)} models...')
            modelAccuracy = learner.test(model, testX, testY)
            self.logger.logDebug(f'Tested models.')

            losses.append(modelAccuracy)

            self.logger.logConcreteObject(losses)

        return losses