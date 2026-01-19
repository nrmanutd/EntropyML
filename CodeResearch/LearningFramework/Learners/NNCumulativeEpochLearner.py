import math
import numpy as np

import torch.nn as nn

from CodeResearch.LearningFramework.Learners.NNTorchModelLearner import TorchModelLearner, ModelFactory
from CodeResearch.LearningFramework.Samplers.Batches.randomAllsetSampler import RandomAllsetSampler
from CodeResearch.ObjectComplexity.InstancePriority.standardPriorityCalculator import StandardPriorityCalculator


class NNCumulativeEpochLearner(TorchModelLearner):
    def train(self, x, y, probs=None):
        raise NotImplementedError()

    def update(self, model: nn.Module, x, y) -> nn.Module:
        raise NotImplementedError()

    def trainAndTest(self, x, y, probs, xt, yt):
        raise NotImplementedError()

    def trainAndTestOnEachEpoch(self, x, y, probs, xt, yt):
        self.fraction = 0.1
        innerEpochs = self.update_epochs
        outerEpochs = self.epochs

        sampler = RandomAllsetSampler(x, y, math.ceil(len(y) * self.fraction), StandardPriorityCalculator())
        batches = sampler.sample()

        model = None
        optimizer = None
        scaler = None

        curX = None
        curY = None

        for xb, yb in batches:
            curX = np.concatenate([curX, xb], axis=0)  if curX is not None else xb
            curY = np.concatenate([curY, yb])  if curY is not None else yb

            if model is None:
                model = self.build_model().to(self.device)

            model, optimizer, scaler, a, p = self._trainModel(model, curX, curY, probs[:len(curY)], innerEpochs, xt, yt, optimizer, scaler)

        model, optimizer, scaler, a, p = self._trainModel(model, x, y, probs, outerEpochs, xt, yt, optimizer, scaler)
        return model, a, p