from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class DataTransformationParametersLearner(BaseLearner):
    def __init__(self, learner: BaseLearner, parameters, dataTransformer: BaseDataProcessor):
        self.dataTransformer = dataTransformer
        self.parameters = parameters
        self.learner = learner

    def trainAndTestOnEachEpoch(self, x, y, probs, xt, yt):
        p = self.parameters
        x, y = self.dataTransformer.applyParametersToData(x, y, p)
        xt, yt = self.dataTransformer.applyParametersToData(xt, yt, p)

        return self.learner.trainAndTestOnEachEpoch(x, y, probs, xt, yt)

    def trainAndTest(self, x, y, probs, xt, yt):
        p = self.parameters
        x, y = self.dataTransformer.applyParametersToData(x, y, p)
        xt, yt = self.dataTransformer.applyParametersToData(xt, yt, p)

        return self.learner.trainAndTest(x, y, probs, xt, yt)

    def test(self, model, x, y):
        p = self.parameters
        x, y = self.dataTransformer.applyParametersToData(x, y, p)

        return self.learner.test(model, x, y)

    def update(self, model, x, y):
        p = self.parameters
        x, y = self.dataTransformer.applyParametersToData(x, y, p)

        return self.learner.update(model, x, y)

    def train(self, x, y, probs):
        p = self.parameters
        x, y = self.dataTransformer.applyParametersToData(x, y, p)

        return self.learner.train(x, y, probs)