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
                    importanceIdx = k - 1 if k > 0 else 0
                    importance = importances[importanceIdx]

                    product = self.calculateProductBasedPriority(importance, np.random.permutation(len(importance)), beta)

                    cutIdx = stratified_split_indices_with_min_and_priority(target, product, beta*alpha)
                    curProbs = softmax(importance[cutIdx])
                    for r in range(self.repeats):
                        resultPriorities.append(cutIdx)
                        probs.append(curProbs)

            if self.useHardness:
                for k in range(len(self.betas)):
                    beta = self.betas[k]
                    easiness = easinesses[k]
                    cutIdx = stratified_split_indices_with_min_and_priority(target, easiness, beta * alpha)
                    curProbs = softmax(easiness[cutIdx])
                    for r in range(self.repeats):
                        resultPriorities.append(cutIdx)
                        probs.append(curProbs)

            if self.useBoth:
                for k in range(len(self.betas)):
                    beta = self.betas[k]
                    easiness = easinesses[-1]
                    importanceIdx = k - 1 if k > 0 else 0
                    importance = importances[importanceIdx]

                    product = self.calculateProductBasedPriority(importance, easiness, beta)

                    idx = stratified_split_indices_with_min_and_priority(target, product, beta * alpha)
                    curProbs = np.full(product[idx], 1.0 / len(idx))

                    for r in range(self.repeats):
                        resultPriorities.append(idx)
                        probs.append(curProbs)

        return resultPriorities, probs

    def calculateProductBasedPriority(self, importance, easiness, beta):
        totalObjects = len(importance)

        easinessIdx = np.argsort(-easiness)
        topEasiestCount = math.ceil(beta / 2 * totalObjects)

        topEasiestIdx = easinessIdx[:topEasiestCount]
        topEasiestIdxSet = set(topEasiestIdx)

        importanceIdx = np.argsort(-importance)

        resultPriority = np.zeros(totalObjects)

        for k in range(topEasiestCount):
            originalIdx = topEasiestIdx[k]
            resultPriority[originalIdx] = totalObjects - k

        counter = topEasiestCount
        for k in range(len(importance)):
            originalIdx = importanceIdx[k]
            if originalIdx in topEasiestIdxSet:
                continue

            resultPriority[originalIdx] = totalObjects - counter
            counter += 1

        return resultPriority