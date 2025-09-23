from abc import ABC, abstractmethod

class BaseMetricCalculator(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculateMetric(self, ds, target):
        pass

    @abstractmethod
    def calculateMetricPro(self, ds, target, vt1, sds1, vt2, sds2):
        pass

    def getMetricName(self):
        return self.name