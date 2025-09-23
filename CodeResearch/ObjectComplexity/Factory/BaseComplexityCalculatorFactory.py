from abc import ABC, abstractmethod

class BaseComplexityCalculatorFactory(ABC):
    @abstractmethod
    def createComplexityCalculator(self, ds, target, idx):
        pass