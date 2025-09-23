from abc import ABC, abstractmethod

class BaseComplexityCalculator(ABC):
    @abstractmethod
    def updateComplexity(self, d, idx):
        pass