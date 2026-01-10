import numpy as np
import torch

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Learners.TorchLearner import TorchMLPLearner
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import direction_from_two_models, \
    per_sample_grads_vmap_full, proj_and_orth_norm
from CodeResearch.ObjectComplexity.Hardness.BaseHardnessCalculator import BaseHardnessCalculator


class IncrementalHardnessCalculator(BaseHardnessCalculator):
    def __init__(self, learner: TorchMLPLearner, nAttempts: int, logger: BaseLogger):
        self.logger = logger
        self.nAttempts = nAttempts
        self.learner = learner

    def calculateHardness(self, ds, t, baseDataSet, baseTarget, alpha):
        device = self.learner.device

        importance = np.zeros(len(t))
        easiness = np.zeros(len(t))

        if baseDataSet is None or len(baseTarget) == 0:
            #self.logger.logWarn('Warning: baseDataSet is empty for incremental hardness calculator')
            #return np.ones(len(t)), np.ones(len(t))
            raise ValueError('Incremental hardness calculator shouldnt be used with empty baseDataSet ')

        for i in range(self.nAttempts):
            model_before = self.learner.build_model()
            model_after = self.learner.update(model_before, baseDataSet, baseTarget)

            acc, predictions = self.learner.test(model_after, ds, t)

            for j in range(len(t)):
                easiness += 1 if predictions[j] == t[j] else 0

            m, names = direction_from_two_models(model_before, model_after)

            xb = torch.as_tensor(ds, dtype=torch.float32, device=device)
            yb = torch.as_tensor(t, dtype=torch.int64, device=device)

            model_after.eval()

            G_attempt, _ = per_sample_grads_vmap_full(model_after, xb, yb, names=names)
            proj, orth = proj_and_orth_norm(G_attempt, m)

            info = orth * torch.sign(proj)
            info_np = info.detach().cpu().numpy()

            importance += info_np

        easiness /= self.nAttempts
        importance /= self.nAttempts

        self.logger.logDebug(f'Finished calculating additional diversification for alpha = {alpha}')

        return importance, easiness