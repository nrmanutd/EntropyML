from abc import ABC, abstractmethod


class BaseDataProcessor(ABC):
    @abstractmethod
    def estimateDataTransformationParameters(self, dataSet, target):
        pass

    @abstractmethod
    def applyParametersToData(self, dataSet, target, parameters):
        pass