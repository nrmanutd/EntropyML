from CodeResearch.DataSeparationFramework.Metrics.BaseMetricCalculator import BaseMetricCalculator
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba


class KSMetric(BaseMetricCalculator):
    def __init__(self):
        super().__init__("KS")

    def calculateMetric(self, ds, target):
        return getMaximumDiviserFastNumba(ds, target)