from abc import ABC, abstractmethod


class BaseScoreCalculator(ABC):
    @abstractmethod
    def calculateScore(self, model, batches, device):
        pass

    def calculateScoreDifferentBatches(self, model, trainBatches, testBatches, device):
        return self.calculateScore(model, testBatches, device)
