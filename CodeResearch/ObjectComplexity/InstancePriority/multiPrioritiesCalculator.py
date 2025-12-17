import math

import numpy as np
from scipy.special import softmax

from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min_and_priority
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, hardnessCalculator: BaseHardnessCalculator, alphas, betas, repeats, useBasedPriority, useImportance, useHardness,
                 useBoth):
        self.repeats = repeats
        self.hardnessCalculator = hardnessCalculator
        self.alphas = alphas
        self.betas = betas
        self.useBoth = useBoth
        self.useHardness = useHardness
        self.useImportance = useImportance
        self.useBasedPriority = useBasedPriority

    def calculatePriority(self, dataSet, target):

        hardnessResult = self.hardnessCalculator.calculateHardness(dataSet, target)
        importance = hardnessResult[0]
        easyness = hardnessResult[1]

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
                for beta in self.betas:
                    cutIdx = stratified_split_indices_with_min_and_priority(target, importance, beta*alpha)
                    curProbs = softmax(importance[cutIdx])
                    for r in range(self.repeats):
                        resultPriorities.append(cutIdx)
                        probs.append(curProbs)

            if self.useHardness:
                for beta in self.betas:
                    cutIdx = stratified_split_indices_with_min_and_priority(target, easyness, beta * alpha)
                    curProbs = softmax(easyness[cutIdx])
                    for r in range(self.repeats):
                        resultPriorities.append(cutIdx)
                        probs.append(curProbs)

            product = self.calculateProductBasedPriority(importance, easyness, 0.2)

            if self.useBoth:
                for beta in self.betas:
                    idx = stratified_split_indices_with_min_and_priority(target, product, beta * alpha)
                    curProbs = softmax(product[idx])
                    for r in range(self.repeats):
                        resultPriorities.append(idx)
                        probs.append(curProbs)

        return resultPriorities, probs

    def calculateProductBasedPriority(self, importance, easyness, alpha):
        return (3 - easyness) * (importance - 2)

        beta = 1 - alpha
        epsilon = 1e-12
        x_safe = np.maximum(importance, epsilon)
        y_safe = np.maximum(easyness, epsilon)

        min_ie = np.minimum(x_safe, y_safe)

        product = ((x_safe * y_safe) ** alpha) * (min_ie ** beta)
        #return product

        return importance * easyness
        #return 0.5 * (importance + easyness)