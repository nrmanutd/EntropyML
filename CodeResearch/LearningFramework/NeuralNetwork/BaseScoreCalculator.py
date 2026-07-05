from abc import ABC, abstractmethod


class BaseScoreCalculator(ABC):
    @abstractmethod
    def calculateScore(self, model, batches, device):
        pass
