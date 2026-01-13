from abc import abstractmethod

from CodeResearch.LearningFramework.Learners.DataPreprocessingLearner import DataPreprocessingLearner
from CodeResearch.CurriculumLearning.clServices.BaseCLLearnersFactory import BaseCLLearnersFactory
from CodeResearch.LearningFramework.DataProcessing.NormalizingPyTorchCVProcessor import NormalizingPyTorchCVProcessor


class PyTorchCVCLLearnersFactory(BaseCLLearnersFactory):
    def getDataPreprocessor(self):
        return NormalizingPyTorchCVProcessor()

    def createScoreLearner(self, epochs):
        # scoring: сеть меньше, lr ниже, чтобы метрика была стабильнее
        learner = self.createScoreLearner_int(epochs)

        return learner

    def createTargetLearner(self, parameters):
        learner = self.createTargetLearner_int(parameters)
        learner = DataPreprocessingLearner(learner, self.getDataPreprocessor())

        return learner

    @abstractmethod
    def createScoreLearner_int(self, epochs):
        pass

    @abstractmethod
    def createTargetLearner_int(self, epochs):
        pass