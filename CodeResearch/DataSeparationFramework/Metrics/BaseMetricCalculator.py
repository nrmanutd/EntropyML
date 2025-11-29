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

    @abstractmethod
    def calculateMetricGpu(self, dsClasses, dsClasses_device, tClasses, ss1, ss1_device, vt1, bvt1, ss2, ss2_device, vt2, bvt2):
        pass

    def getMetricName(self):
        return self.name