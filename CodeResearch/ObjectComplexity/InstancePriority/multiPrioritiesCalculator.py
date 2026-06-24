import math

import numpy as np
from scipy.special import softmax

from CodeResearch.CurriculumLearning.clServices.commonCLHelpers import calculateProductBasedPriority
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min_and_priority
from CodeResearch.ObjectComplexity.Hardness.UsefulObjectsCalculator import UsefulObjectsCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator


class MultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, hcBuilder, logger: BaseLogger, alphas, betas, shouldEstimateForFullSet, useBasedPriority, useImportance, useHardness,
                 useBoth):
        self.shouldEstimateForFullSet = shouldEstimateForFullSet
        self.hcBuilder = hcBuilder
        self.logger = logger
        self.alphas = alphas
        self.betas = np.sort(betas)
        self.useBoth = useBoth
        self.useHardness = useHardness
        self.useImportance = useImportance
        self.useBasedPriority = useBasedPriority
        self.usefullObjectsCalculator = UsefulObjectsCalculator()

    def calculatePriority(self, dataSet, target):
        #return self.calculatePriorityNoChain(dataSet, target)
        return self.calculateChainPriority(dataSet, target)

    def calculateChainPriority(self, dataSet, target):
        resultPriorities = []
        probs = []

        for alpha in self.alphas:
            nTrain = math.ceil(alpha * len(target))

            if self.useBasedPriority:
                for beta in self.betas:
                    curNTrain = math.ceil(beta * nTrain)
                    rIdx = np.random.permutation(len(target))
                    resultPriorities.append(rIdx[range(curNTrain)])
                    probs.append(np.full(curNTrain, 1.0 / curNTrain))

            if self.useImportance:
                resIdxes, resProbs = self.calculateChain(dataSet, target, alpha, 'importance', 'easiness')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

            if self.useHardness:
                resIdxes, resProbs = self.calculateChain(dataSet, target, alpha, 'easiness', 'easiness')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

            if self.useBoth:
                resIdxes, resProbs = self.calculateChain(dataSet, target, alpha, 'both')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

        return resultPriorities, probs

    def calculatePriorityNoChain(self, dataSet, target):
        resultPriorities = []
        probs = []

        for alpha in self.alphas:
            nTrain = math.ceil(alpha * len(target))
            importances, easinesses = self.calculateHardnessAndDiversityMaps(dataSet, target, alpha)

            if self.useBasedPriority:
                for beta in self.betas:
                    curNTrain = math.ceil(beta * nTrain)
                    rIdx = np.random.permutation(len(target))
                    resultPriorities.append(rIdx[range(curNTrain)])
                    probs.append(np.full(curNTrain, 1.0 / curNTrain))

            if self.useImportance:
                resIdxes, resProbs = self.calculateChainNoState(target, importances, easinesses, alpha, 'importance')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

            if self.useHardness:
                resIdxes, resProbs = self.calculateChainNoState(target, importances, easinesses, alpha, 'easiness')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

            if self.useBoth:
                resIdxes, resProbs = self.calculateChainNoState(target, importances, easinesses, alpha, 'both')

                for i in range(len(resIdxes)):
                    resultPriorities.append(resIdxes[i])
                    probs.append(resProbs[i])

        return resultPriorities, probs

    def calculateChain(self, dataSet, target, alpha, priorityType: str, firstBetaPriorityType: str = None):
        self.logger.logDebug(f'Calculating chain for {priorityType}...')

        if firstBetaPriorityType is None:
            firstBetaPriorityType = priorityType

        resultPriorities = []
        probs = []

        nObjects = len(target)

        currentDataSetIdx = []
        prevBeta = 0
        prevEasiness = np.zeros(nObjects)
        curProbs = []

        for k in range(len(self.betas)):
            beta = self.betas[k]

            if abs(beta - 1) < 0.001:
                ci = set(currentDataSetIdx)
                restIdx = np.array([i for i in range(nObjects) if i not in ci], dtype=np.int64)
                currentDataSetIdx.extend(restIdx)
                curProbs.extend(np.ones(len(restIdx)) * min(curProbs))

                resultPriorities.append(np.array(currentDataSetIdx))
                probs.append(np.full(len(curProbs), 1.0))
                break

            deltaBeta = beta - prevBeta
            prevBeta = beta

            currentIdx = np.array(currentDataSetIdx, dtype=np.int64)
            ci = set(currentDataSetIdx)
            restIdx = np.array([i for i in range(nObjects) if i not in ci], dtype=np.int64)
            fraction = deltaBeta * alpha * nObjects / len(restIdx)

            shouldUseLearnerBased = True if k == 0 else False
            hc = self.hcBuilder(shouldUseLearnerBased)

            importance, easiness = hc.calculateHardness(dataSet[restIdx], target[restIdx],
                                                             dataSet[currentIdx], target[currentIdx], fraction)

            easinessDelta = easiness - prevEasiness

            currentPriorityType = firstBetaPriorityType if k == 0 else priorityType

            if currentPriorityType == 'importance':
                priority = importance
            elif currentPriorityType == 'easiness':
                priority = easiness
            elif currentPriorityType == 'easiness_delta':
                priority = easinessDelta
            elif currentPriorityType == 'easiness_delta&importance':
                priority = calculateProductBasedPriority(importance, easinessDelta, 0.5)
            elif currentPriorityType == 'easiness&importance':
                priority = calculateProductBasedPriority(importance, easiness, 0.5)
            else:
                raise ValueError(f'Unknown priorityType: {priorityType}')

            cutIdx = stratified_split_indices_with_min_and_priority(target[restIdx], priority, fraction)
            currentDataSetIdx.extend(restIdx[cutIdx])

            multiplier = 1 if len(curProbs) == 0 else (min(curProbs) / np.max(softmax(priority[cutIdx])))
            curProbs.extend(softmax(priority[cutIdx]) * multiplier)

            ci = set(cutIdx)
            prevEasiness = easiness[np.array([i for i in range(len(easiness)) if i not in ci], dtype=np.int64)]

            if self.shouldEstimateForFullSet is False:
                resultPriorities.append(np.array(currentDataSetIdx))
                probs.append(np.full(len(currentDataSetIdx), 1.0))

        return resultPriorities, probs

    def calculateHardnessAndDiversityMaps(self, dataSet, target, alpha):
        easinesses = {}
        importances = {}

        for k in range(len(self.betas)):
            beta = self.betas[k]

            if abs(beta - 1) < 0.001:
                continue

            fraction = beta * alpha
            hc = self.hcBuilder(True)
            importance, easiness = hc.calculateHardness(dataSet, target, None, None, fraction)

            easinesses[beta] = easiness
            importances[beta] = importance

        return importances, easinesses

    def calculateChainNoState(self, target, importances, easinesses, alpha, priorityType: str):
        self.logger.logDebug(f'Calculating chain no state for {priorityType}...')

        nObjects = len(target)
        resultPriorities = []
        probs = []

        for k in range(len(self.betas)):
            beta = self.betas[k]

            if abs(beta - 1) < 0.001:
                resultPriorities.append(np.arange(nObjects))
                probs.append(np.full(nObjects, 1.0 / nObjects))#todo: add logic for collecting idxes from previous iterations
                break

            fraction = beta * alpha

            importance = importances[beta]
            easiness = easinesses[beta]

            if priorityType == 'importance':
                priority = importance
            elif priorityType == 'easiness':
                priority = easiness
            else:
                priority = calculateProductBasedPriority(importance, easiness, 0.5)

            cutIdx = stratified_split_indices_with_min_and_priority(target, priority, fraction)
            #curProbs = easiness[cutIdx] / sum(easiness[cutIdx])
            curProbs = softmax(priority[cutIdx])

            if self.shouldEstimateForFullSet is False:
                resultPriorities.append(np.array(cutIdx))
                probs.append(curProbs)

        return resultPriorities, probs