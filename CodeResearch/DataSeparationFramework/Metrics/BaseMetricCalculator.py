from abc import ABC, abstractmethod

class BaseMetricCalculator(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculateMetric(self, ds, target):
        pass

    def getMetricName(self):
        return self.name