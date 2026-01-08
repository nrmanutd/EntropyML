import math

import numpy as np

from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min, \
    stratified_split_indices_from_ks_native
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.ObjectAssessment.BaseObjectAssesor import BaseObjectAssesor
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger


class LearnerBasedHardnessCalculator(BaseHardnessCalculator):
    def __init__(self, learner: BaseLearner, assesor: BaseObjectAssesor, nAttempts, logger: BaseLogger):
        self.logger = logger
        self.assesor = assesor
        self.nAttempts = nAttempts
        self.learner = learner

        logger.logDebug(f'nAttempts = {nAttempts}')

    def calculateHardness(self, dataSet, target, baseDataSet, baseTarget, alpha):

        trainIdxes = []
        testIdxes = []
        testResponds = []

        ds = dataSet
        t = target

        self.logger.logDebug(f'Start hardness calculation for alpha = {alpha}...')

        for i in range(self.nAttempts):
            if i%10 == 0:
                self.logger.logDebug(f'Attempt #{i} of {self.nAttempts}...')

            trainIdx, testIdx = stratified_split_indices_with_min(target, alpha)

            x = ds[trainIdx, :]
            y = t[trainIdx]

            xtest = ds[testIdx, :]
            ytest = t[testIdx]

            extended_x = np.concatenate([x, baseDataSet], axis = 0) if baseDataSet is not None else x
            extended_y = np.concatenate([y, baseTarget]) if baseDataSet is not None else y

            res = self.learner.trainAndTest(extended_x, extended_y, np.full(len(extended_y), fill_value=1.0/len(extended_y)), xtest, ytest)

            trainIdxes.append(trainIdx)
            testIdxes.append(testIdx)
            testResponds.append(res)

        self.logger.logDebug('Assesing results...')
        importance, easiness = self.assesor.estimate(trainIdxes, testIdxes, testResponds, t)

        self.logger.logDebug('Finished calculating hardness')

        return importance, easiness