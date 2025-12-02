import numpy as np

from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class CompositeLearner(BaseLearner):
    def __init__(self, learner: BaseLearner):
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
        results = np.zeros(len(model))

        for i in range(len(model)):
            currentModel = model[i]
            results[i] = self.learner.test(currentModel, x[i], y[i])

        return results