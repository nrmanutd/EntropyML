import math

import numpy as np
import torch
from scipy.special import softmax

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min_and_priority, \
    stratified_split_indices_with_min
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.Hardness.UsefulObjectsCalculator import UsefulObjectsCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, hcBuilder, logger: BaseLogger, alphas, betas, repeats, useBasedPriority, useImportance, useHardness,
                 useBoth):
        self.repeats = repeats
        self.hcBuilder = hcBuilder
        self.logger = logger
        self.alphas = alphas
        self.betas = betas
        self.useBoth = useBoth
        self.useHardness = useHardness
        self.useImportance = useImportance
        self.useBasedPriority = useBasedPriority
        self.usefullObjectsCalculator = UsefulObjectsCalculator()

    def calculatePriority(self, dataSet, target):

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
                resIdxes, resProbs = self.calculateChain(dataSet, target, alpha, 'importance')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

            if self.useHardness:
                resIdxes, resProbs = self.calculateChain(dataSet, target, alpha, 'easiness_delta')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

            if self.useBoth:
                resIdxes, resProbs = self.calculateChain(dataSet, target, alpha, 'easiness_delta&importance')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

                resIdxes, resProbs = self.calculateChain(dataSet, target, alpha, 'easiness&importance')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

        return resultPriorities, probs

    def calculateChain(self, dataSet, target, alpha, priorityType: str):
        self.logger.logDebug(f'Calculating chain for {priorityType}...')

        resultPriorities = []
        probs = []

        nObjects = len(target)

        currentDataSetIdx = []
        prevBeta = 0
        prevEasiness = np.zeros(nObjects)

        np.random.seed(42)

        for k in range(len(self.betas)):
            beta = self.betas[k]

            if abs(beta - 1) < 0.001:
                for r in range(self.repeats):
                    resultPriorities.append(np.arange(nObjects))
                    probs.append(np.full(nObjects, 1.0 / nObjects))

                break

            deltaBeta = beta - prevBeta
            prevBeta = beta

            currentIdx = np.array(currentDataSetIdx, dtype=np.int64)
            ci = set(currentDataSetIdx)
            restIdx = np.array([i for i in range(nObjects) if i not in ci], dtype=np.int64)
            fraction = deltaBeta * alpha * nObjects / len(restIdx)
            hc = self.hcBuilder()

            importance, easiness = hc.calculateHardness(dataSet[restIdx, :], target[restIdx],
                                                             dataSet[currentIdx, :], target[currentIdx], fraction)

            easinessDelta = easiness - prevEasiness

            if priorityType == 'importance':
                priority = importance
            elif priorityType == 'easiness':
                priority = easiness
            elif priorityType == 'easiness_delta':
                priority = easinessDelta
            elif priorityType == 'easiness_delta&importance':
                priority = self.calculateProductBasedPriority(importance, easinessDelta, 0.5)
            elif priorityType == 'easiness&importance':
                priority = self.calculateProductBasedPriority(importance, easiness, 0.5)
            else:
                raise ValueError(f'Unknown priorityType: {priorityType}')

            cutIdx = stratified_split_indices_with_min_and_priority(target[restIdx], priority, fraction)
            currentDataSetIdx.extend(restIdx[cutIdx])

            ci = set(cutIdx)
            prevEasiness = easiness[np.array([i for i in range(len(easiness)) if i not in ci], dtype=np.int64)]

            #curProbs = softmax(np.arange(len(currentDataSetIdx), 0, -1))
            curProbs = (np.full(len(currentDataSetIdx), 1.0/len(currentDataSetIdx)))

            for r in range(self.repeats):
                resultPriorities.append(np.array(currentDataSetIdx))
                probs.append(curProbs)

        return resultPriorities, probs

    def calculateChainNoState(self, dataSet, target, alpha, priorityType: str):
        self.logger.logDebug(f'Calculating chain no state for {priorityType}...')

        nObjects = len(target)
        resultPriorities = []
        probs = []

        for k in range(len(self.betas)):
            beta = self.betas[k]

            if abs(beta - 1) < 0.001:
                for r in range(self.repeats):
                    resultPriorities.append(np.arange(nObjects))
                    probs.append(np.full(nObjects, 1.0 / nObjects))

                break

            fraction = beta * alpha
            hc = self.hcBuilder()
            importance, easiness = hc.calculateHardness(dataSet, target, None, None, fraction)

            if priorityType == 'importance':
                priority = importance
            elif priorityType == 'easiness':
                priority = easiness
            else:
                priority = self.calculateProductBasedPriority(importance, easiness, 0.5)

            cutIdx = stratified_split_indices_with_min_and_priority(target, priority, fraction)
            curProbs = easiness[cutIdx] / sum(easiness[cutIdx])

            for r in range(self.repeats):
                resultPriorities.append(np.array(cutIdx))
                probs.append(curProbs)

        return resultPriorities, probs

    def calculateProductBasedPriority(self, importance, easiness, alpha = 0.5):
        n = len(importance)
        eps = 1 / (2 * n)

        score = np.exp(alpha * np.log(eps + importance) + (1 - alpha) * np.log(eps + easiness))

        return score

