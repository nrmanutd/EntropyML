from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner
from CodeResearch.LearningFramework.Samplers.SamplersFactories.baseSamplersFactory import BaseSamplersFactory


class EpochLearner(BaseLearner):
    def update(self, model, x, y):
        pass

    def __init__(self, epochs, learner: BaseLearner, samplersFactory: BaseSamplersFactory):
        self.samplersFactory = samplersFactory
        self.learner = learner
        self.epochs = epochs

    def test(self, models, x, y):

        accuracies = []
        for model in models:
            accuracy = self.learner.test(model, x, y)
            accuracies.append(accuracy)

        return accuracies

    def train(self, x, y, probs):

        currentModel = None
        sampler = self.samplersFactory.createSampler(x, y, probs)

        trainedModels = []

        for epoch in range(self.epochs):
            batches = sampler.sample()

            for xx, yy in batches:
                currentModel = self.learner.train(xx, yy, probs) if currentModel is None else self.learner.update(currentModel, xx, yy)

            trainedModels.append(currentModel)

        return trainedModels