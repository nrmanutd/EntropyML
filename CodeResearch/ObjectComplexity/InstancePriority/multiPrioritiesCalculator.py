import math

import numpy as np
from scipy.special import softmax

from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, hardnessCalculator: BaseHardnessCalculator, alphas, useBasedPriority, useImportance, useHardness,
                 useBoth):
        self.hardnessCalculator = hardnessCalculator
        self.alphas = alphas
        self.useBoth = useBoth
        self.useHardness = useHardness
        self.useImportance = useImportance
        self.useBasedPriority = useBasedPriority

    def calculatePriority(self, dataSet, target):

        hardnessResult = self.hardnessCalculator.calculateHardness(dataSet, target)
        importance = hardnessResult[0]
        easiness = hardnessResult[1]

        importanceIdx = np.argsort(importance)[::-1]
        hardnessIdx = np.argsort(easiness)[::-1]

        minImportance = np.min(importance)
        maxImportance = np.max(importance)

        correctedImportance = (importance - minImportance) / (maxImportance - minImportance)

        bothIdx = np.argsort(easiness * correctedImportance)[::-1]

        resultPriorities = []
        probs = []

        for alpha in self.alphas:
            nTrain = math.ceil(alpha * len(target))

            if self.useBasedPriority:
                resultPriorities.append(range(nTrain))
                probs.append(np.full(nTrain, 1.0 / nTrain))

            if self.useImportance:
                cutIdx = importanceIdx[:nTrain]
                resultPriorities.append(cutIdx)
                probs.append(softmax(importance[cutIdx]))

            if self.useHardness:
                cutIdx = hardnessIdx[:nTrain]
                resultPriorities.append(cutIdx)
                probs.append(softmax(easiness[cutIdx]))

            if self.useBoth:
                cutIdx = bothIdx[:nTrain]
                resultPriorities.append(cutIdx)
                values = easiness * correctedImportance
                probs.append(softmax(values[cutIdx]))

        return resultPriorities, probs
