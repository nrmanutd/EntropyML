import math

import numpy as np
from scipy.special import softmax

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min_and_priority, \
    stratified_split_indices_with_min
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.UsefulObjectsCalculator import UsefulObjectsCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, hcs, logger: BaseLogger, alphas, betas, repeats, useBasedPriority, useImportance, useHardness,
                 useBoth):
        self.repeats = repeats
        self.hcs = hcs
        self.logger = logger
        self.alphas = alphas
        self.betas = betas
        self.useBoth = useBoth
        self.useHardness = useHardness
        self.useImportance = useImportance
        self.useBasedPriority = useBasedPriority
        self.usefullObjectsCalculator = UsefulObjectsCalculator()

    def calculatePriority(self, dataSet, target):

        importances = []
        easinesses = []

        for hc in self.hcs:
            hardnessResult = hc.calculateHardness(dataSet, target)
            importance = hardnessResult[0]
            easiness = hardnessResult[1]

            importances.append(importance)
            easinesses.append(easiness)

        easinessThreshold = self.usefullObjectsCalculator.evaluate(easinesses[0], easinesses[-1])

        resultPriorities = []
        probs = []

        for alpha in self.alphas:
            nTrain = math.ceil(alpha * len(target))

            if self.useBasedPriority:
                for beta in self.betas:
                    curNTrain = math.ceil(beta * nTrain)
                    for r in range(self.repeats):
                        rIdx = np.random.permutation(len(target))
                        resultPriorities.append(rIdx[range(curNTrain)])
                        probs.append(np.full(curNTrain, 1.0 / curNTrain))

            if self.useImportance:
                for k in range(len(self.betas)):
                    beta = self.betas[k]
                    currentMetrices = easinesses
                    currentMetricIdx = min(k, len(currentMetrices) - 1)
                    currentMetric = currentMetrices[currentMetricIdx]

                    for r in range(self.repeats):
                        cutIdx = stratified_split_indices_with_min_and_priority(target, currentMetric, beta * alpha)
                        curProbs = softmax(currentMetric[cutIdx])

                        resultPriorities.append(cutIdx)
                        probs.append(curProbs)

            if self.useHardness:
                for k in range(len(self.betas)):
                    beta = self.betas[k]
                    easiness = easinesses[min(k, len(easinesses) - 1)]

                    subIdx = np.where(easiness >= easinessThreshold)[0]

                    #cutIdx = stratified_split_indices_with_min_and_priority(target[subIdx], easiness[subIdx], beta * alpha * len(easiness) / len(subIdx))
                    part = beta * alpha * len(easiness) / len(subIdx)
                    if part > 1:
                        self.logger.logDebug(f'Part = {part}, beta = {beta}, alpha = {alpha}')
                        part = 1

                    cutIdx, testIdx = stratified_split_indices_with_min(target[subIdx], part)
                    cutIdx = subIdx[cutIdx]
                    curProbs = softmax(easiness[cutIdx])

                    for r in range(self.repeats):
                        resultPriorities.append(cutIdx)
                        probs.append(curProbs)

            if self.useBoth:
                for k in range(len(self.betas)):
                    beta = self.betas[k]
                    curIdx = min(k, len(easinesses) - 1)
                    easiness = easinesses[curIdx]
                    importance = importances[curIdx]

                    product = self.calculateProductBasedPriority(importance, easiness, 0.5)
                    idx = stratified_split_indices_with_min_and_priority(target, product, beta * alpha)
                    curProbs = softmax(product[idx])

                    for r in range(self.repeats):
                        resultPriorities.append(idx)
                        probs.append(curProbs)

        return resultPriorities, probs

    def calculateProductBasedPriority(self, importance, easiness, alpha = 0.5):
        n = len(importance)
        eps = 1 / (2 * n)

        score = np.exp(alpha * np.log(eps + importance) + (1 - alpha) * np.log(eps + easiness))

        return score