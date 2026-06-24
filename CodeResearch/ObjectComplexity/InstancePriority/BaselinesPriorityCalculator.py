import math

import numpy as np
import torch

from CodeResearch.LearningFramework.Samplers.Batches.randomAllsetSampler import RandomAllsetSampler
from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import centered_grad_norm_head_linear_two_pass, \
    grad_norm_head_linear_one_pass, el2n_scores
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator
from CodeResearch.ObjectComplexity.InstancePriority.standardPriorityCalculator import StandardPriorityCalculator


class BaselinesPriorityCalculator(BasePriorityCalculator):
    def __init__(self, nAttempts: int, betas, batchSize: int, metric: str, dataTransformer: BaseDataProcessor, learnerCreator=None):
        self.dataTransformer = dataTransformer
        self.metric = metric
        self.nAttempts = nAttempts
        self.betas = betas
        self.batchSize = batchSize
        self.learnerCreator = learnerCreator

    def calculatePriority(self, dataSet, target):

        p = self.dataTransformer.estimateDataTransformationParameters(dataSet, target)
        ds, t = self.dataTransformer.applyParametersToData(dataSet, target, p)

        device = self.learnerCreator().learner.device
        xb = torch.as_tensor(ds, dtype=torch.float32, device=device)
        yb = torch.as_tensor(t, dtype=torch.int64, device=device)
        
        sampler = RandomAllsetSampler(xb, yb, self.batchSize, StandardPriorityCalculator())

        scores = np.zeros(len(target), dtype=np.float32)
        for i in range(self.nAttempts):
            learner = self.learnerCreator()
            model = learner.train(dataSet, target, np.full(len(target), 1.0 / len(target)))

            batches = sampler.sample()
            if self.metric == "EL2N":
                currentScores = el2n_scores(model, batches, device)
            elif self.metric == "GraNd":
                currentScores = grad_norm_head_linear_one_pass(model, batches, device)
            else:
                raise ValueError(f'Incorrect metric type: {self.metric}')

            scores += np.asarray(currentScores, dtype=np.float32)

        idxes = np.argsort(scores)[::-1]

        resultPriorities = []
        probs = []

        for beta in self.betas:
            curNTrain = math.ceil(beta * len(idxes))
            curIdxes = idxes[:curNTrain]

            resultPriorities.append(curIdxes)
            probs.append(np.full(curNTrain, 1.0 / curNTrain))

        return resultPriorities, probs