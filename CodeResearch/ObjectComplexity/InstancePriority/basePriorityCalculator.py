from abc import ABC, abstractmethod


class BasePriorityCalculator(ABC):
    @abstractmethod
    def calculatePriority(self, dataSet, target):
        pass
