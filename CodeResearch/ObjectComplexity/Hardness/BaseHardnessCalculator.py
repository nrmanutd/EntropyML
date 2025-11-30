from abc import ABC, abstractmethod


class BaseHardnessCalculator(ABC):

    @abstractmethod
    def calculateHardness(self, dataSet, target):
        pass