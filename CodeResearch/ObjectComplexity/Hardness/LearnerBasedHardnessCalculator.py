import math

import numpy as np

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.ObjectAssessment.BaseObjectAssesor import BaseObjectAssesor
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger


class LearnerBasedHardnessCalculator(BaseHardnessCalculator):
    def __init__(self, learner: BaseLearner, assesor: BaseObjectAssesor, nAttempts, alpha, logger: BaseLogger):
        self.logger = logger
        self.assesor = assesor
        self.alpha = alpha
        self.nAttempts = nAttempts
        self.learner = learner

    def calculateHardness(self, dataSet, target):

        n = len(target)
        trainObjects = math.ceil(n * self.alpha)

        trainIdxes = []
        testIdxes = []
        testResponds = []

        self.logger.logDebug('Start hardness calculation...')

        for i in range(self.nAttempts):
            if i%10 == 0:
                self.logger.logDebug(f'Attempt #{i}...')

            randomIdx = np.random.permutation(n)

            trainIdx = randomIdx[:trainObjects]
            testIdx = randomIdx[trainObjects:]

            res = self.learner.trainAndTest(dataSet[trainIdx, :], target[trainIdx], np.full(trainObjects, fill_value=1.0/trainObjects),dataSet[testIdx, :], target[testIdx]) #todo: make res in all learners tuple - accuracy and learner responds

            trainIdxes.append(trainIdx)
            testIdxes.append(testIdx)
            testResponds.append(res)

        self.logger.logDebug('Assesing results...')
        importance, easyness = self.assesor.estimate(trainIdxes, testIdxes, testResponds, target)

        self.logger.logDebug('Finished calculating hardness')

        return importance, easyness