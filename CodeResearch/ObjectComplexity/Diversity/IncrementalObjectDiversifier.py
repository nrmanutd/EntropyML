import numpy as np
import torch

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchLearner
from CodeResearch.LearningFramework.Learners.TorchMLPLearner import TorchMLPLearner
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import per_sample_grads_vmap, calculateDelta, \
    per_sample_grads_last_layer_loop


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

            model = self.learner.train(baseDataSet, baseTarget, np.full(len(baseTarget), 1.0 / len(baseTarget)))
            model.eval()

            #G_attempt = per_sample_grads_vmap(model, xb, yb)
            G_attempt = per_sample_grads_last_layer_loop(model, xb, yb)
            g_delta = calculateDelta(G_attempt, mode='centered_grad_norm')

            importance += g_delta

            del G_attempt
            del model
            torch.cuda.empty_cache()

        importance /= self.nAttempts

        self.logger.logDebug(f'Finished calculating additional diversification for alpha = {alpha}')

        return importance