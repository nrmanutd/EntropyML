import math

import numpy as np
from scipy.special import softmax

from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min_and_priority
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, hcs, alphas, betas, repeats, useBasedPriority, useImportance, useHardness,
                 useBoth):
        self.repeats = repeats
        self.hcs = hcs
        self.alphas = alphas
        self.betas = betas
        self.useBoth = useBoth
        self.useHardness = useHardness
        self.useImportance = useImportance
        self.useBasedPriority = useBasedPriority

    def calculatePriority(self, dataSet, target):

        importances = []
        easinesses = []

        for hc in self.hcs:
            hardnessResult = hc.calculateHardness(dataSet, target)
            importance = hardnessResult[0]
            easiness = hardnessResult[1]

            importances.append(importance)
            easinesses.append(easiness)

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
                    importanceIdx = min(k, len(importances) - 1)
                    importance = importances[importanceIdx]

                    for r in range(self.repeats):
                        cutIdx = stratified_split_indices_with_min_and_priority(target, importance, beta * alpha)
                        curProbs = softmax(importance[cutIdx])

                        resultPriorities.append(cutIdx)
                        probs.append(curProbs)

            if self.useHardness:
                for k in range(len(self.betas)):
                    beta = self.betas[k]
                    easiness = easinesses[min(k, len(easinesses) - 1)]
                    cutIdx = stratified_split_indices_with_min_and_priority(target, easiness, beta * alpha)
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