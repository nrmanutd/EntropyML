import math

import numpy as np

from CodeResearch.CurriculumLearning.clServices.commonCLHelpers import calculateProductBasedPriority
from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
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

        prioritiesIdx = self.calculatePriorities(dataSet, target)

        for beta in self.betas:
            curNTrain = math.ceil(beta * len(target))
            for r in range(self.repeats):
                rIdx = prioritiesIdx[:curNTrain]
                resultPriorities.append(rIdx)
                probs.append(np.full(curNTrain, 1.0 / curNTrain))

        return resultPriorities, probs

    def calculatePriorities(self, dataSet, target):
        self.logger.logDebug(f'Calculating priorities via chain ')

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

            idx, model = self.getYetAnotherChain(model, dataSet[restIdx], target[restIdx], fraction)
            currentDataSetIdx.extend(idx)

        return np.array(currentDataSetIdx)

    def GetInitialSet(self, dataSet, target, beta) -> np.ndarray:
        importance, easiness = self.hc.calculateHardness(dataSet, target, None, None, beta)
        nObjects = math.ceil(beta * len(target))
        return np.argsort(-easiness)[:nObjects]

    def getYetAnotherChain(self, model, dataSet, target, fraction):
        sampler = RandomAllsetSampler(dataSet, target, 128, StandardPriorityCalculator())
        batches = sampler.sample()
        importance, hardness = centered_grad_norm_head_linear_two_pass_entropy_loss(model, batches, self.learner.device)

        priority = calculateProductBasedPriority(importance, np.exp(-hardness))
        nObjects = math.ceil(fraction * len(target))

        nextObjectsToTrainIdx = np.argsort(-priority)[:nObjects]

        model = self.learner.update(model, dataSet[nextObjectsToTrainIdx], target[nextObjectsToTrainIdx])

        return nextObjectsToTrainIdx, model