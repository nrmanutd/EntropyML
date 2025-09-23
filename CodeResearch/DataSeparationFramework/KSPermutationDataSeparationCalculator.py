from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.SimpleDataSeparationCalculator import SimpleDataSeparationCalculator
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.ObjectComplexity.Factory.KSComplexityCalculatorFactory import KSComplexityCalculatorFactory


class KSPermutationDataSeparationCalculator(SimpleDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, taskName, folder, logsFolder):
        super().__init__(dataSet, target, attempts, "KS_permutation", taskName, folder, logsFolder)
        self.pValuesCalculator = PValueCalculator(KSComplexityCalculatorFactory(), KSMetric(), attempts, True, True, False)

    def calculateMetric(self, objects, iClass, jClass):
        pValues = self.pValuesCalculator.calcPValueFastPro(objects, self.dataSet, self.target, iClass, jClass)
        return pValues
