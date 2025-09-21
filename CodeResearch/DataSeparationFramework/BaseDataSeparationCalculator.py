from abc import ABC, abstractmethod


class BaseDataSeparationCalculator(ABC):
    @abstractmethod
    def calculateDataSeparability(self, objects, iClass, jClass):
        pass

    @abstractmethod
    def serializeResults(self):
        pass







