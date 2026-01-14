from torch import nn as nn

from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner


class NNUpdatableTorchModelLearner(TorchModelLearner):
    def train(self, x, y, probs=None):
        model = self.build_model().to(self.device)
        model, optimizer, a, p = self._trainModel(model, x, y, probs, self.epochs, None, None)

        return model, optimizer

    def update(self, m, x, y):
        model = m[0]
        optimizer = m[1]

        model = model.to(self.device)
        model, optimizer, a, p = self._trainModel(model, x, y, None, self.update_epochs, None, None, optimizer)

        return model, optimizer