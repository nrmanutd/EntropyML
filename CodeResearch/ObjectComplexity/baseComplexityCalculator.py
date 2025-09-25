from abc import ABC, abstractmethod

class BaseComplexityCalculator(ABC):
    @abstractmethod
    def updateComplexity(self, diviser, classUnderDiviser, idx):
        pass