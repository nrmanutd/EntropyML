from CodeResearch.DataSeparationFramework.Metrics.BaseMetricCalculator import BaseMetricCalculator

class StubMetric(BaseMetricCalculator):
    def calculateMetricPro(self, ds, target, vt1, sds1, vt2, sds2):
        return 0

    def calculateMetric(self, ds, target):
        return 0