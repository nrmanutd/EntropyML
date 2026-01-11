import numpy as np
import torch

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Learners.TorchMLPLearner import TorchMLPLearner
from CodeResearch.ObjectComplexity.Diversity.BaseObjectDiversifier import BaseObjectDiversifier
from CodeResearch.ObjectComplexity.Diversity.DiversifierHelpers import per_sample_grads_vmap, calculateDelta


class NNBasedObjectDiversifier(BaseObjectDiversifier):
    def __init__(self, learner: TorchMLPLearner, samplerFactory, epochs, logger: BaseLogger):
        self.logger = logger
        self.learner = learner
        self.epochs = epochs
        self.samplerFactory = samplerFactory

    def calculateObjectDiversity(self, dataSet, target, baseDataSet, baseTarget, alpha):
        self.logger.logDebug('Estimating object diversity...')

        sampler = self.samplerFactory(dataSet, target)
        currentModel = None

        device = self.learner.device
        all_epochs_scores = []

        for epoch in range(self.epochs):
            self.logger.logDebug(f'Estimating diversity for epoch #{epoch} of {self.epochs}...')

            if baseDataSet is not None and len(baseTarget) != 0:
                if currentModel is None:
                    currentModel = self.learner.train(baseDataSet, baseTarget,
                                                  np.full(len(baseTarget), 1.0 / len(baseTarget)))
                else:
                    self.learner.update(currentModel, baseDataSet, baseTarget)

            batches = sampler.sample()
            g_list = []

            for xx, yy in batches:
                # ---- 2) Перевод данных в torch на нужный device ----
                xb = torch.as_tensor(xx, dtype=torch.float32, device=device)
                yb = torch.as_tensor(yy, dtype=torch.int64, device=device)

                if currentModel is not None:
                    # ---- 3) Временно переключаем режим на eval() для стабильного измерения ----
                    was_training = currentModel.training
                    currentModel.eval()

                    # важно: НЕ оборачивай это в torch.no_grad(), т.к. градиенты нужны
                    #G_batch = self.per_sample_grads_last_layer_loop(currentModel, xb, yb)
                    G_batch = per_sample_grads_vmap(currentModel, xb, yb)

                    # ---- 4) Вернуть режим как было ----
                    currentModel.train(was_training)
                else:
                    tempModel = self.learner.build_model()
                    tempModel = tempModel.to(device)
                    tempModel.eval()

                    #G_batch = self.per_sample_grads_last_layer_loop(tempModel, xb, yb)
                    G_batch = per_sample_grads_vmap(tempModel, xb, yb)

                g_list.append(G_batch.detach())

                probs = np.full(len(yy), 1.0 / len(yy))

                currentModel = self.learner.train(xx, yy, probs) if currentModel is None else self.learner.update(
                    currentModel, xx, yy)

            g_delta = calculateDelta(g_list, mode='centered_grad_norm')
            all_epochs_scores.append(g_delta)

        final_scores = np.mean(np.stack(all_epochs_scores, axis=0), axis=0)

        self.logger.logDebug(f'Final scores: {final_scores}')
        self.logger.logDebug('Object diversity estimated.')

        return final_scores

