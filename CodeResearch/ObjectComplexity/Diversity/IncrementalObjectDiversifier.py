import numpy as np
import torch

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchLearner
from CodeResearch.LearningFramework.Samplers.Batches.randomAllsetSampler import RandomAllsetSampler
from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import centered_grad_norm_head_linear_two_pass
from CodeResearch.ObjectComplexity.InstancePriority.standardPriorityCalculator import StandardPriorityCalculator


class IncrementalObjectDiversifier(BaseObjectDiversifier):
    def __init__(self, learner: TorchLearner, nAttempts: int, batchSize: int, logger: BaseLogger):
        self.logger = logger
        self.nAttempts = nAttempts
        self.learner = learner
        self.batchSize = batchSize

    def calculateObjectDiversity(self, ds, t, baseDataSet, baseTarget, alpha):
        device = self.learner.device

        importance = np.zeros(len(t))

        if baseDataSet is None or len(baseTarget) == 0:
            return importance
            #raise ValueError('Incremental hardness calculator shouldnt be used with empty baseDataSet ')

        xb = torch.as_tensor(ds, dtype=torch.float32, device=device)
        yb = torch.as_tensor(t, dtype=torch.int64, device=device)

        sampler = RandomAllsetSampler(xb, yb, self.batchSize, StandardPriorityCalculator())

        for i in range(self.nAttempts):
            if i%10 == 0:
                self.logger.logDebug(f'Calculating incremental step for #{i} of {self.nAttempts} attempts')

            model = self.learner.train(baseDataSet, baseTarget, np.full(len(baseTarget), 1.0 / len(baseTarget)))

            batches = sampler.sample()
            scores = centered_grad_norm_head_linear_two_pass(model, batches, device)

            importance += scores

            del model
            torch.cuda.empty_cache()

        importance /= self.nAttempts

        self.logger.logDebug(f'Finished calculating additional diversification for alpha = {alpha}')

        return importance