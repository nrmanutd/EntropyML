from abc import ABC, abstractmethod


class BaseObjectAssessor(ABC):
    @abstractmethod
    def estimate(self, trainIdxes, testIdxes, testResponds, target):
        pass

    @abstractmethod
    def estimateEasiness(self, trainIdxes, testIdxes, testResponds, target):
        pass

    @abstractmethod
    def estimateImportance(self, trainIdxes, testIdxes, testResponds, target):
        pass