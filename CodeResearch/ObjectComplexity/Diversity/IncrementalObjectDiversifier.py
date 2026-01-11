import numpy as np
import torch

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchLearner
from CodeResearch.LearningFramework.Learners.TorchMLPLearner import TorchMLPLearner
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import per_sample_grads_vmap, calculateDelta


class IncrementalObjectDiversifier(BaseObjectDiversifier):
    def __init__(self, learner: TorchLearner, nAttempts: int, logger: BaseLogger):
        self.logger = logger
        self.nAttempts = nAttempts
        self.learner = learner

    def calculateObjectDiversity(self, ds, t, baseDataSet, baseTarget, alpha):
        device = self.learner.device

        importance = np.zeros(len(t))

        if baseDataSet is None or len(baseTarget) == 0:
            return importance
            #raise ValueError('Incremental hardness calculator shouldnt be used with empty baseDataSet ')

        xb = torch.as_tensor(ds, dtype=torch.float32, device=device)
        yb = torch.as_tensor(t, dtype=torch.int64, device=device)

        for i in range(self.nAttempts):
            if i%10 == 0:
                self.logger.logDebug(f'Calculating incremental step for #{i} of {self.nAttempts} attempts')

            model_before = self.learner.build_model().to(device)
            model_after = self.learner.update(model_before, baseDataSet, baseTarget)

            model_after.eval()

            G_attempt = per_sample_grads_vmap(model_after, xb, yb)
            G_attempt = G_attempt.detach()

            g_delta = calculateDelta(G_attempt, mode='centered_grad_norm')
            importance += g_delta

        importance /= self.nAttempts

        self.logger.logDebug(f'Finished calculating additional diversification for alpha = {alpha}')

        return importance