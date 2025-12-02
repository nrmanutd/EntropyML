from abc import ABC, abstractmethod

class BaseSamplersFactory(ABC):
    @abstractmethod
    def createSampler(self, x, y, probs):
        pass