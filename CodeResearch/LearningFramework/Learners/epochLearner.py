import gc
import os

import numpy as np
import torch.cuda

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.LearningFramework.NeuralNetwork.PytorchHelpers import NeuralNetwork
from CodeResearch.LearningFramework.Samplers.SamplersFactories.baseSamplersFactory import BaseSamplersFactory


class EpochLearner(BaseLearner):
    def trainAndTestOnEachEpoch(self, x, y, probs, xt, yt):
        raise NotImplementedError()

    def update(self, model, x, y):
        raise NotImplementedError()

    def __init__(self, epochs, learner: BaseLearner):
        self.learner = learner
        self.epochs = epochs
        self.trainId = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loadModel(self, model):
        raise NotImplementedError()

    def test(self, models, x, y):
        raise NotImplementedError()

    def train(self, x, y, probs):
        raise NotImplementedError()

    def trainAndTest(self, x, y, probs, xt, yt):
        model, accuracies, predictions = self.learner.trainAndTestOnEachEpoch(x, y, probs, xt, yt)
        return accuracies, predictions
