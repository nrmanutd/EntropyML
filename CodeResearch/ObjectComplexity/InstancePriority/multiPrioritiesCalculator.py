import math

import numpy as np
from scipy.special import softmax

from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, hardnessCalculator: BaseHardnessCalculator, alphas, repeats, useBasedPriority, useImportance, useHardness,
                 useBoth):
        self.repeats = repeats
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

        resultPriorities = []
        probs = []
        betas = [0.05, 0.1, 0.5, 1]

        for alpha in self.alphas:
            nTrain = math.ceil(alpha * len(target))

            if self.useBasedPriority:
                for beta in betas:
                    curNTrain = math.ceil(beta * nTrain)
                    for r in range(self.repeats):
                        rIdx = np.random.permutation(len(target))
                        resultPriorities.append(rIdx[range(curNTrain)])

                        probs.append(np.full(curNTrain, 1.0 / curNTrain))

            #if self.useImportance:#beta = 0.1
            #    cutIdx = importanceIdx[:nTrain]
            #    resultPriorities.append(cutIdx)
            #    probs.append(softmax(importance[cutIdx]))

            #if self.useHardness:#beta = 0.5
            #    cutIdx = hardnessIdx[:nTrain]
            #    resultPriorities.append(cutIdx)
            #    probs.append(softmax(easiness[cutIdx]))

            if self.useBoth:
                curIdx, curProbs = MultiPrioritiesCalculator.assign_weights(importance, easiness)

                for beta in betas:
                    curNTrain = math.ceil(nTrain * beta)
                    idx = curIdx[:curNTrain]
                    for r in range(self.repeats):
                        resultPriorities.append(idx)
                        probs.append(curProbs[:curNTrain] / sum(curProbs[:curNTrain]))

        return resultPriorities, probs

    @staticmethod
    def assign_weights(importance, easiness):

        xy_product = importance * easiness
        sortedIdx = np.argsort(-xy_product)

        return sortedIdx, softmax(xy_product[sortedIdx])
