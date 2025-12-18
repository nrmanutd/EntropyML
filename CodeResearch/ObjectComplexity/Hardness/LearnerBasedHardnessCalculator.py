import math

import numpy as np

from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min
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

        logger.logDebug(f'Alpha = {alpha}')

    def calculateHardness(self, dataSet, target):

        trainIdxes = []
        testIdxes = []
        testResponds = []

        self.logger.logDebug('Start hardness calculation...')

        for i in range(self.nAttempts):
            if i%10 == 0:
                self.logger.logDebug(f'Attempt #{i}...')

            trainIdx, testIdx = stratified_split_indices_with_min(target, self.alpha)

            x = dataSet[trainIdx, :]
            y = target[trainIdx]

            xtest = dataSet[testIdx, :]
            ytest = target[testIdx]

            res = self.learner.trainAndTest(x, y, np.full(len(trainIdx), fill_value=1.0/len(trainIdx)), xtest, ytest)

            trainIdxes.append(trainIdx)
            testIdxes.append(testIdx)
            testResponds.append(res)

        self.logger.logDebug('Assesing results...')
        importance, easiness = self.assesor.estimate(trainIdxes, testIdxes, testResponds, target)

        self.logger.logDebug('Finished calculating hardness')

        return importance, easiness