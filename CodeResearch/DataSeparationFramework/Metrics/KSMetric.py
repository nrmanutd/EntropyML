from CodeResearch.DataSeparationFramework.Metrics.BaseMetricCalculator import BaseMetricCalculator
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba, \
    getMaximumDiviserFastNumbaCore


class KSMetric(BaseMetricCalculator):
    def __init__(self):
        super().__init__("KS")

    def calculateMetricPro(self, ds, target, vt1, sds1, vt2, sds2):
        return getMaximumDiviserFastNumbaCore(ds, target, vt1, sds1, vt2, sds2)

    def calculateMetric(self, ds, target):
        return getMaximumDiviserFastNumba(ds, target)