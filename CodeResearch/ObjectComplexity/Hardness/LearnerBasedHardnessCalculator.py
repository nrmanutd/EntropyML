import math

import numpy as np

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.ObjectAssessment.BaseObjectAssesor import BaseObjectAssesor


class LearnerBasedHardnessCalculator(BaseHardnessCalculator):
    def __init__(self, learner: BaseLearner, assesor: BaseObjectAssesor, nAttempts, alpha):
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

        for i in range(self.nAttempts):

            randomIdx = np.random.permutation(n)

            trainIdx = randomIdx[:trainObjects]
            testIdx = randomIdx[trainObjects:]

            res = self.learner.trainAndTest(dataSet[trainIdx, :], target[trainIdx], np.full(trainObjects, fill_value=1.0/trainObjects),dataSet[testIdx, :], target[testIdx]) #todo: make res in all learners tuple - accuracy and learner responds

            trainIdxes.append(trainIdx)
            testIdxes.append(testIdx)
            testResponds.append(res)

        importance, easyness = self.assesor.estimate(trainIdxes, testIdxes, testResponds, target)

        return importance, easyness