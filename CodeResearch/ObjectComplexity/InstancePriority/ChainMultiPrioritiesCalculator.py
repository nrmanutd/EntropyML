import math

import numpy as np

from CodeResearch.CurriculumLearning.clServices.commonCLHelpers import calculateProductBasedPriority
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min_and_priority
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchLearner
from CodeResearch.LearningFramework.Samplers.Batches.randomAllsetSampler import RandomAllsetSampler
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import \
    centered_grad_norm_head_linear_two_pass_entropy_loss
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator
from CodeResearch.ObjectComplexity.InstancePriority.standardPriorityCalculator import StandardPriorityCalculator


class ChainMultiPrioritiesCalculator(BasePriorityCalculator):
    def __init__(self, betas, repeats, learner: TorchLearner, hc: BaseHardnessCalculator, logger: BaseLogger):
        self.hc = hc
        self.learner = learner
        self.repeats = repeats
        self.logger = logger
        self.betas = np.sort(betas)

    def calculatePriority(self, dataSet, target):
        resultPriorities = []
        probs = []

        prioritiesIdx = self.calculatePriorities(dataSet, target, 'easiness&importance')

        for beta in self.betas:
            curNTrain = math.ceil(beta * len(target))
            for r in range(self.repeats):
                rIdx = prioritiesIdx[:curNTrain]
                resultPriorities.append(rIdx)
                probs.append(np.full(curNTrain, 1.0 / curNTrain))

        prioritiesIdx = self.calculatePriorities(dataSet, target, 'easiness')

        for beta in self.betas:
            curNTrain = math.ceil(beta * len(target))
            for r in range(self.repeats):
                rIdx = prioritiesIdx[:curNTrain]
                resultPriorities.append(rIdx)
                probs.append(np.full(curNTrain, 1.0 / curNTrain))

        return resultPriorities, probs

    def calculatePriorities(self, dataSet, target, priorityType):
        self.logger.logDebug(f'Calculating priorities via chain. Objects in dataset = {len(target)} ')

        nObjects = len(target)

        currentDataSetIdx = []
        prevBeta = 0

        model = None

        for k in range(len(self.betas)):
            beta = self.betas[k]
            self.logger.logDebug(f'Estimating chain for beta {beta} #{k} of {len(self.betas)}...')

            if abs(beta - 1) < 0.001:
                ci = set(currentDataSetIdx)
                restIdx = np.array([i for i in range(nObjects) if i not in ci], dtype=np.int64)
                currentDataSetIdx.extend(restIdx)
                break

            if k == 0:
                idx = self.GetInitialSet(dataSet, target, beta)
                currentDataSetIdx.extend(idx)
                model = self.learner.train(dataSet[idx], target[idx], None)
                prevBeta = beta
                continue

            deltaBeta = beta - prevBeta
            prevBeta = beta

            ci = set(currentDataSetIdx)
            restIdx = np.array([i for i in range(nObjects) if i not in ci], dtype=np.int64)
            fraction = deltaBeta * nObjects / len(restIdx)

            currentIdx = np.array(currentDataSetIdx)
            idx, model = self.getYetAnotherChain(model, dataSet[restIdx], target[restIdx], dataSet[currentIdx], target[currentIdx], fraction, priorityType)

            restOrigIdx = restIdx[idx]
            currentDataSetIdx.extend(restOrigIdx)

        return np.array(currentDataSetIdx)

    def GetInitialSet(self, dataSet, target, beta) -> np.ndarray:
        importance, easiness = self.hc.calculateHardness(dataSet, target, None, None, beta)
        cutIdx = stratified_split_indices_with_min_and_priority(target, easiness, beta)

        return cutIdx

    def getYetAnotherChain(self, model, dataSet, target, baseDataSet, basetTarget, fraction, priorityType: str):
        sampler = RandomAllsetSampler(dataSet, target, 128, StandardPriorityCalculator())
        batches = sampler.sample()
        self.logger.logDebug(f'Another chain calculating: {len(target)} of potential objects, {len(basetTarget)} of added objects')
        importance, easiness = centered_grad_norm_head_linear_two_pass_entropy_loss(model[0], batches, self.learner.device)
        self.logger.logDebug(f'Calculated importance and hardness')

        if priorityType == 'easiness&importance':
            priority = calculateProductBasedPriority(importance, easiness)
        elif priorityType == 'easiness':
            priority = easiness
        else:
            raise ValueError(f'Unknown priority: {priorityType}')
        nextObjectsToTrainIdx = stratified_split_indices_with_min_and_priority(target, priority, fraction)

        extended_x = np.concatenate([baseDataSet, dataSet[nextObjectsToTrainIdx]], axis=0)
        extended_y = np.concatenate([basetTarget, target[nextObjectsToTrainIdx]])

        self.logger.logDebug(f'Before training on extended dataset of length {len(extended_y)}')
        #model = self.learner.train(extended_x, extended_y, None)
        model = self.learner.update(model, extended_x, extended_y)
        self.logger.logDebug('Training finished')

        return nextObjectsToTrainIdx, model