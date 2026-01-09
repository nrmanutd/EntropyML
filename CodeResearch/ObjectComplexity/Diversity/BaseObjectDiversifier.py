from abc import ABC, abstractmethod

class BaseObjectDiversifier(ABC):

    @abstractmethod
    def calculateObjectDiversity(self, dataSet, target, baseDataSet, baseTarget):
        pass