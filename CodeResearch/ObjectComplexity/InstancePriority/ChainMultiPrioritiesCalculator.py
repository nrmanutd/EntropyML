import copy
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
    def __init__(self, betas, nAttempts, repeats, learner: TorchLearner, hc: BaseHardnessCalculator, logger: BaseLogger):
        self.nAttempts = nAttempts
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

        return resultPriorities, probs

    def calculatePriorities(self, dataSet, target, priorityType):
        self.logger.logDebug(f'Calculating priorities via chain. Objects in dataset = {len(target)} ')

        nObjects = len(target)

        currentDataSetIdx = []
        prevBeta = 0

        prioritiezedOrigIdxes = []

        for k in range(len(self.betas)):
            beta = self.betas[k]
            self.logger.logDebug(f'Estimating chain for beta {beta} #{k} of {len(self.betas)}...')

            if abs(beta - 1) < 0.001:
                ci = set(currentDataSetIdx)
                restIdx = np.array([i for i in range(nObjects) if i not in ci], dtype=np.int64)
                currentDataSetIdx.extend(restIdx)
                break

            if k == 0:
                priority = self.GetInitialPriority(dataSet, target, beta)

            deltaBeta = beta - prevBeta
            prevBeta = beta

            ci = set(currentDataSetIdx)
            restIdx = np.array([i for i in range(nObjects) if i not in ci], dtype=np.int64)
            fraction = deltaBeta * nObjects / len(restIdx)

            currentIdx = np.array(currentDataSetIdx)
            addedIdxes, priority = self.updateModelAccordingToPrioritiezed(dataSet, target, currentIdx, restIdx, priority, fraction, priorityType)

            currentDataSetIdx.extend(addedIdxes)

        return np.array(currentDataSetIdx)

    def GetInitialPriority(self, dataSet, target, beta, idxesFraction=1) -> np.ndarray:
        importance, easiness = self.hc.calculateHardness(dataSet, target, None, None, beta)

        return easiness

    def updateModelAccordingToPrioritiezed(self, dataSet, target, currentIdx, restIdx, priority, fraction, priorityType: str):
        currentAddedIdxes = stratified_split_indices_with_min_and_priority(target[restIdx], priority, fraction)
        addedIdxes = restIdx[currentAddedIdxes]

        ci = set(addedIdxes)
        restIdxes = np.array([i for i in restIdx if i not in ci], dtype=np.int64)

        importance = np.zeros(len(restIdxes))
        easiness = np.zeros(len(restIdxes))

        sampler = RandomAllsetSampler(dataSet[restIdxes], target[restIdxes], 128, StandardPriorityCalculator())
        batches = sampler.sample()

        extended_x = np.concatenate([dataSet[currentIdx], dataSet[addedIdxes]], axis=0) if len(currentIdx) > 0 else dataSet[addedIdxes]
        extended_y = np.concatenate([target[currentIdx], target[addedIdxes]]) if len(currentIdx) > 0 else target[addedIdxes]

        for i in range(self.nAttempts):
            self.logger.logDebug(f'Learning attempt {i} of {self.nAttempts}. Objects number: {len(extended_y)}')
            model = self.learner.train(extended_x, extended_y, None)
            self.logger.logDebug('Model trained')
            curImportance, curEasiness = centered_grad_norm_head_linear_two_pass_entropy_loss(model[0], batches,
                                                                                        self.learner.device)

            self.logger.logDebug('Calculated importance and easiness...')

            importance += curImportance
            easiness += curEasiness

        importance /= self.nAttempts
        easiness /= self.nAttempts

        if priorityType == 'easiness&importance':
            priority = calculateProductBasedPriority(importance, easiness)
        elif priorityType == 'easiness':
            priority = easiness
        else:
            raise ValueError(f'Unknown priority: {priorityType}')

        return addedIdxes, priority


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