import math

import numpy as np

from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, hardnessCalculator: BaseHardnessCalculator, alphas, useBasedPriority, useImportance, useHardness, useBoth):
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

        for alpha in self.alphas:
            nTrain = math.ceil(alpha * len(target))

            if self.useBasedPriority:
                resultPriorities.append(range(nTrain))

            if self.useImportance:
                resultPriorities.append(importanceIdx[:nTrain])

            if self.useHardness:
                resultPriorities.append(hardnessIdx[:nTrain])

            if self.useBoth:
                resultPriorities.append(bothIdx[:nTrain])

        return resultPriorities