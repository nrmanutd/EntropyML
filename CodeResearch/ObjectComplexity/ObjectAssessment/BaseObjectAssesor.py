from abc import ABC, abstractmethod


class BaseObjectAssesor(ABC):
    @abstractmethod
    def estimate(self, trainIdxes, testIdxes, testResponds, target):
        pass

    @abstractmethod
    def estimateEasyness(self, trainIdxes, testIdxes, testResponds, target):
        pass

    @abstractmethod
    def estimateImportance(self, trainIdxes, testIdxes, testResponds, target):
        pass