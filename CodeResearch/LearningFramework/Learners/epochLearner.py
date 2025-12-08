import gc
import os

import numpy as np
import torch.cuda
from tensorflow.python.keras.models import clone_model
import copy

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.LearningFramework.NeuralNetwork.PytorchHelpers import NeuralNetwork
from CodeResearch.LearningFramework.Samplers.SamplersFactories.baseSamplersFactory import BaseSamplersFactory


class EpochLearner(BaseLearner):
    def update(self, model, x, y):
        pass

    def __init__(self, epochs, learner: BaseLearner, samplersFactory: BaseSamplersFactory):
        self.samplersFactory = samplersFactory
        self.learner = learner
        self.epochs = epochs
        self.trainId = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loadModel(self, model):
        key = model[0]
        nFeatures = model[1]
        nClasses = model[2]

        m = NeuralNetwork(nFeatures, nClasses, 512).to(self.device)
        state = torch.load(key, map_location=self.device)
        m.load_state_dict(state)

        return m

    def test(self, models, x, y):

        accuracies = []
        for model in models:

            m = self.loadModel(model)

            accuracy = self.learner.test(m, x, y)
            accuracies.append(accuracy)

            os.remove(model[0])
            del m
            torch.cuda.empty_cache()

        gc.collect()
        return accuracies

    def train(self, x, y, probs):
        currentModel = None
        sampler = self.samplersFactory.createSampler(x, y, probs)

        trainedModels = []

        for epoch in range(self.epochs):
            batches = sampler.sample()

            for xx, yy in batches:
                currentModel = self.learner.train(xx, yy, probs) if currentModel is None else self.learner.update(currentModel, xx, yy)

            currentModelState = currentModel.state_dict()
            key = f'TempModels\\model_{self.trainId}_{epoch}.pt'
            if not os.path.exists("TempModels"):
                os.mkdir("TempModels")

            torch.save(currentModelState, key)
            trainedModels.append((key, x.shape[1], len(np.unique(y))))

        del currentModel.optimizer
        del currentModel
        torch.cuda.empty_cache()

        self.trainId += 1
        torch.cuda.empty_cache()
        return trainedModels

    def trainAndTest(self, x, y, probs, xt, yt):
        currentModel = None
        sampler = self.samplersFactory.createSampler(x, y, probs)

        accuracies = []

        for epoch in range(self.epochs):
            batches = sampler.sample()

            for xx, yy in batches:
                currentModel = self.learner.train(xx, yy, probs) if currentModel is None else self.learner.update(
                    currentModel, xx, yy)

            accuracy = self.learner.test(currentModel, xt, yt)
            accuracies.append(accuracy)
            torch.cuda.empty_cache()

        return accuracies