import copy

import numpy as np

from CodeResearch.CurriculumLearning.clServices.commonCLHelpers import should_stop
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min
from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor
from CodeResearch.LearningFramework.Learners.DataTransformationParametersLearner import \
    DataTransformationParametersLearner
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.ObjectAssessment.BaseObjectAssesor import BaseObjectAssesor

class LearnerBasedHardnessCalculator(BaseHardnessCalculator):
    def __init__(self, learner: BaseLearner, assesor: BaseObjectAssesor, nAttempts, dataTransformer: BaseDataProcessor, logger: BaseLogger, minimumIterations: int = 2):
        self.dataTransformer = dataTransformer
        self.minimumIterations = minimumIterations
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
        easinessList = []
        allEasiness = np.zeros(len(target))
        allCounts = np.zeros(len(target))

        if baseDataSet is not None and len(baseTarget) != 0:
            p = self.dataTransformer.estimateDataTransformationParameters(baseDataSet, baseTarget)
            learner = DataTransformationParametersLearner(self.learner, p, self.dataTransformer)
            baseModel = learner.train(baseDataSet, baseTarget, np.full(len(baseTarget), 1.0/len(baseTarget)))
        else:
            learner = self.learner
            baseModel = None

        for i in range(self.nAttempts):
            if i%10 == 0:
                self.logger.logDebug(f'Attempt #{i} of {self.nAttempts}...')

            trainIdx, testIdx = stratified_split_indices_with_min(target, alpha)

            x = ds[trainIdx]
            y = t[trainIdx]

            xtest = ds[testIdx]
            ytest = t[testIdx]

            if baseModel is not None:
                baseModelCopy = copy.deepcopy(baseModel)
                baseModelCopy = learner.update(baseModelCopy, x, y)
            else:
                p = self.dataTransformer.estimateDataTransformationParameters(x, y)
                learner = DataTransformationParametersLearner(self.learner, p, self.dataTransformer)
                baseModelCopy = learner.train(x, y, np.full(len(y), 1.0/len(y)))

            res = learner.test(baseModelCopy, xtest, ytest)

            trainIdxes.append(trainIdx)
            testIdxes.append(testIdx)
            testResponds.append(res)

            curEasiness, curCounts = self.estimateEasiness(testIdx, t, res)
            allEasiness += curEasiness
            allCounts += curCounts

            uptodateEasiness = self.calculateEasiness(allEasiness, allCounts)
            easinessList.append(uptodateEasiness)

            shouldStop = should_stop(easinessList, self.logger)
            if shouldStop and i >= self.minimumIterations:
                self.logger.logDebug(f'Stop criteria based on rank correlation at iteration {i} of {self.nAttempts}')
                break

            del baseModelCopy

        self.logger.logDebug('Assesing results...')
        importance, easiness = self.assesor.estimate(trainIdxes, testIdxes, testResponds, t)

        if baseModel is not None:
            del baseModel

        self.logger.logDebug('Finished calculating hardness')

        return importance, easiness

    def estimateEasiness(self, testIdx, t, res):
        result = np.zeros(len(t))
        objectCounts = np.zeros(len(t))

        for j in range(len(testIdx)):
            originalIdx = testIdx[j]
            objectCounts[originalIdx] = 1

            if res[1][j] == t[originalIdx]:
                result[originalIdx] += 1

        return result, objectCounts

    def calculateEasiness(self, easiness, counts):
        resultEasiness = np.zeros(len(easiness))

        for j in range(len(easiness)):
            if counts[j] == 0:
                if easiness[j] != 0:
                    raise ValueError('Incorrect match between easiness and counts!')
                continue

            resultEasiness[j] = easiness[j] / counts[j]

        return resultEasiness