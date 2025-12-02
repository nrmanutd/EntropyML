from CodeResearch.DataSeparationFramework.Metrics.BaseMetricCalculator import BaseMetricCalculator
from CodeResearch.DiviserCalculation.getDiviserFastNumba import getMaximumDiviserFastNumba, \
    getMaximumDiviserFastNumbaCore


class KSMetric(BaseMetricCalculator):
    def __init__(self):
        super().__init__("KS")

    def calculateMetricPro(self, ds, target, vt1, sds1, vt2, sds2):
        return getMaximumDiviserFastNumbaCore(ds, target, vt1, sds1, vt2, sds2)

    def calculateMetricGpu(self, dsClasses, dsClasses_device, tClasses, ss1, ss1_device, vt1, bvt1, ss2, ss2_device, vt2, bvt2):
        pass
        #return getMaximumDiviserFastCudaCore(dsClasses, dsClasses_device, tClasses, ss1, ss1_device, vt1, bvt1, ss2, ss2_device, vt2, bvt2)

    def calculateMetric(self, ds, target):
        return getMaximumDiviserFastNumba(ds, target)
