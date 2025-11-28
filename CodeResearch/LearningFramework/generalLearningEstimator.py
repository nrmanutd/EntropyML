import time

import numpy as np

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler


class GeneralLearningEstimator:

    def __init__(self, iterations):
        self.iterations = iterations

    def estimateLearner(self, sampler: BaseSampler, learner: BaseLearner):
        losses = np.zeros(self.iterations)

        t0 = time.time()
        t1 = t0
        for i in range(self.iterations):

            if i %10 == 0:
                print(f'Iteration {i} of {self.iterations}, time: {time.time() - t0} s, delta time: {time.time() - t1}')
                t1 = time.time()

            trainX, trainY, testX, testY = sampler.sample()
            model = learner.train(trainX, trainY)
            modelAccuracy = learner.test(model, testX, testY)

            losses[i] = modelAccuracy

        return losses