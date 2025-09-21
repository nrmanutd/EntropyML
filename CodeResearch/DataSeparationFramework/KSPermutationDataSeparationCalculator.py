from CodeResearch.DataSeparationFramework.SimpleDataSeparationCalculator import SimpleDataSeparationCalculator
from CodeResearch.pValueCalculator import calcPValueFastPro


class KSPermutationDataSeparationCalculator(SimpleDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, taskName, folder, logsFolder):
        super().__init__(dataSet, target, attempts, "KS_permutation", taskName, folder, logsFolder)

    def calculateMetric(self, objects, iClass, jClass):
        pValues = calcPValueFastPro(objects, self.dataSet, self.target, iClass, jClass, self.attempts,
                                     True, True, False)
        return pValues
