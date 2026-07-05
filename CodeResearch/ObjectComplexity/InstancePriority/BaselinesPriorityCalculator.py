import math

import numpy as np
import torch

from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor
from CodeResearch.LearningFramework.NeuralNetwork.BaseScoreCalculator import BaseScoreCalculator
from CodeResearch.LearningFramework.Samplers.Batches.randomAllsetSampler import RandomAllsetSampler
from CodeResearch.ObjectComplexity.InstancePriority.basePriorityCalculator import BasePriorityCalculator
from CodeResearch.ObjectComplexity.InstancePriority.standardPriorityCalculator import StandardPriorityCalculator


class BaselinesPriorityCalculator(BasePriorityCalculator):
    def __init__(self, nAttempts: int, betas, batchSize: int, dataTransformer: BaseDataProcessor, scoreCalculator:BaseScoreCalculator, device, learnerCreator=None):
        self.device = device
        self.scoreCalculator = scoreCalculator
        self.dataTransformer = dataTransformer
        self.nAttempts = nAttempts
        self.betas = betas
        self.batchSize = batchSize
        self.learnerCreator = learnerCreator

    def calculatePriority(self, dataSet, target):
        p = self.dataTransformer.estimateDataTransformationParameters(dataSet, target)
        ds, t = self.dataTransformer.applyParametersToData(dataSet, target, p)

        xb = torch.as_tensor(ds, dtype=torch.float32, device=self.device)
        yb = torch.as_tensor(t, dtype=torch.int64, device=self.device)
        
        sampler = RandomAllsetSampler(xb, yb, self.batchSize, StandardPriorityCalculator())

        scores = np.zeros(len(target), dtype=np.float32)
        for i in range(self.nAttempts):
            learner = self.learnerCreator()
            model = learner.train(dataSet, target, np.full(len(target), 1.0 / len(target)))

            batches = sampler.sample()
            currentScores = self.scoreCalculator.calculateScore(model, batches, self.device)

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