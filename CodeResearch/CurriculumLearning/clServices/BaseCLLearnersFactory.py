from abc import ABC, abstractmethod


class BaseCLLearnersFactory(ABC):
    def __init__(self, nClasses, targetBatchSize, scoringBatchSize):
        self.scoringBatchSize = scoringBatchSize
        self.targetBatchSize = targetBatchSize
        self.nClasses = nClasses

    @abstractmethod
    def createTargetLearner(self, parameters):
        pass

    @abstractmethod
    def createScoreLearner(self, parameters):
        pass

    @abstractmethod
    def getDataPreprocessor(self):
        pass