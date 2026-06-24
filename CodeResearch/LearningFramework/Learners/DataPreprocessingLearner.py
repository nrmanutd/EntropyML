from CodeResearch.LearningFramework.DataProcessing.BaseDataProcessor import BaseDataProcessor
from CodeResearch.LearningFramework.Learners.baseLearner import BaseLearner


class DataPreprocessingLearner(BaseLearner):
    def __init__(self, learner: BaseLearner, dataTransformer: BaseDataProcessor):
        self.dataTransformer = dataTransformer
        self.learner = learner

    def trainAndTestOnEachEpoch(self, x, y, probs, xt, yt):
        p = self.dataTransformer.estimateDataTransformationParameters(x, y)
        x, y = self.dataTransformer.applyParametersToData(x, y, p)
        xt, yt = self.dataTransformer.applyParametersToData(xt, yt, p)

        return self.learner.trainAndTestOnEachEpoch(x, y, probs, xt, yt)

    def trainAndTest(self, x, y, probs, xt, yt):
        p = self.dataTransformer.estimateDataTransformationParameters(x, y)
        x, y = self.dataTransformer.applyParametersToData(x, y, p)
        xt, yt = self.dataTransformer.applyParametersToData(xt, yt, p)

        return self.learner.trainAndTest(x, y, probs, xt, yt)

    def test(self, model, x, y):
        raise NotImplementedError()

    def update(self, model, x, y):
        raise NotImplementedError()

    def train(self, x, y, probs):
        p = self.dataTransformer.estimateDataTransformationParameters(x, y)
        x, y = self.dataTransformer.applyParametersToData(x, y, p)

        return self.learner.train(x, y, probs)