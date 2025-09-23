from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.SimpleDataSeparationCalculator import SimpleDataSeparationCalculator
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.ObjectComplexity.Factory.KSComplexityCalculatorFactory import KSComplexityCalculatorFactory


class MLDataSeparationCalculator(SimpleDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, taskName, folder, logsFolder):
        super().__init__(dataSet, target, attempts, "ML", taskName, folder, logsFolder)
        self.pValuesCalculator = PValueCalculator(KSComplexityCalculatorFactory(), KSMetric(), attempts, False, False, True)

    def calculateMetric(self, objects, iClass, jClass):
        return self.pValuesCalculator.calcPValueFastPro(objects, self.dataSet, self.target, iClass, jClass)

    def calculateDataSeparability(self, objects, iClass, jClass):
        curPair = f'{iClass}_{jClass}'

        pValues2 = self.calculateMetric(objects, iClass, jClass)

        if len(pValues2[1]) > 0:
            self.commonPermutationPairs.append(pValues2[1])
            self.processCalculatedMetric(pValues2)

        self.labels.append(curPair)
        self.objectsCount.append(objects)
