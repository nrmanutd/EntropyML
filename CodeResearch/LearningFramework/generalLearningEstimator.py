import time

import numpy as np

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.LearningFramework.Samplers.baseSampler import BaseSampler


class GeneralLearningEstimator:

    def __init__(self, iterations):
        self.iterations = iterations

    def estimateLearner(self, sampler: BaseSampler, learner: BaseLearner):
        losses = []

        t0 = time.time()
        t1 = t0
        for i in range(self.iterations):

            if i %10 == 0:
                print(f'Iteration {i} of {self.iterations}, time: {time.time() - t0} s, delta time: {time.time() - t1}')
                t1 = time.time()

            print('sampling...')
            trainX, trainY, testX, testY, probs = sampler.sample()
            print('training model...')
            model = learner.train(trainX, trainY, probs)
            print('testing model...')
            modelAccuracy = learner.test(model, testX, testY)

            losses.append(modelAccuracy)

        return losses