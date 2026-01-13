import copy

import numpy as np
import torch

from CodeResearch.Helpers.Logger.BaseLogger import BaseLogger
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class CompositeLearner(BaseLearner):
    def __init__(self, learner: BaseLearner, logger: BaseLogger):
        self.logger = logger
        self.learner = learner

    def train(self, x, y, probs):
        models = []

        for i in range(len(x)):
            currentModel = self.learner.train(x[i], y[i], probs[i])
            models.append(currentModel)

        return models

    def update(self, model, x, y):
        for i in range(len(model)):
            currentModel = model[i]
            self.learner.update(currentModel, x[i], y[i])

    def test(self, model, x, y):
        results = []
        accuracy = []

        for i in range(len(model)):
            currentModel = model[i]
            r, a = self.learner.test(currentModel, x[i], y[i])
            results.append(r)
            accuracy.append(a)

        return results, accuracy

    def trainAndTest(self, x, y, probs, xt, yt):
        resultsAccuracy = []
        resultsPredictions = []

        for i in range(len(x)):
            self.logger.logDebug(f'Training model #{i} of {len(x)}')
            accuracy, predictions = self.learner.trainAndTest(x[i], y[i], probs[i], xt[i], yt[i])
            resultsAccuracy.append(accuracy)
            resultsPredictions.append(predictions)

        return resultsAccuracy, resultsPredictions

    def trainAndTestOnEachEpoch(self, x, y, probs, xt, yt):
        resultsAccuracy = []
        resultsPredictions = []
        models = []

        for i in range(len(x)):
            self.logger.logDebug(f'Training model #{i} of {len(x)}')
            m, accuracy, predictions = self.learner.trainAndTestOnEachEpoch(x[i], y[i], probs[i], xt[i], yt[i])
            resultsAccuracy.append(accuracy)
            resultsPredictions.append(predictions)
            models.append(copy.deepcopy(m))

        return models, resultsAccuracy, resultsPredictions