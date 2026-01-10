import numpy as np
import torch

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.Helpers.permutationHelpers import stratified_split_indices_with_min
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchMLPLearner
from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import per_sample_grads_vmap, calculateDelta


class SeparableObjectDiversifier(BaseObjectDiversifier):
    def __init__(self, learner: TorchMLPLearner, nAttempts, logger: BaseLogger):
        self.logger = logger
        self.nAttempts = nAttempts
        self.learner = learner

    def calculateObjectDiversity(self, ds, t, baseDataSet, baseTarget, alpha):

        result = np.zeros(len(t))
        device = self.learner.device

        for iAttempt in range(self.nAttempts):
            if iAttempt%10 == 0:
                self.logger.logDebug(f'Calculating object diversity for attempt #{iAttempt} of {self.nAttempts}')

            trainIdx, testIdx = stratified_split_indices_with_min(t, alpha)

            x = ds[trainIdx, :]
            y = t[trainIdx]

            xtest = ds[testIdx, :]
            ytest = t[testIdx]

            extended_x = np.concatenate([x, baseDataSet], axis=0) if baseDataSet is not None else x
            extended_y = np.concatenate([y, baseTarget]) if baseDataSet is not None else y

            model = self.learner.train(extended_x, extended_y,
                                            np.full(len(extended_y), fill_value=1.0 / len(extended_y)))

            xb = torch.as_tensor(xtest, dtype=torch.float32, device=device)
            yb = torch.as_tensor(ytest, dtype=torch.int64, device=device)

            model.eval()

            G_attempt = per_sample_grads_vmap(model, xb, yb)
            G_attempt = G_attempt.detach()

            g_delta = calculateDelta(G_attempt, mode='centered_grad_norm')

            for i in range(len(testIdx)):
                result[testIdx[i]] += g_delta[i]

        self.logger.logDebug(f'Finished calculating {self.nAttempts} diversification for alpha = {alpha}')

        return result / len(result)