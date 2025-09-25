from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.Metrics.StubMetric import StubMetric
from CodeResearch.DataSeparationFramework.SimpleDataSeparationCalculator import SimpleDataSeparationCalculator
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.ObjectComplexity.Factory.ShapValuesComplexityCalculatorFactory import \
    ShapValuesComplexityCalculatorFactory
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays

class ShapValueDataSeparationCalculator(SimpleDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, taskName, folder, logsFolder):
        super().__init__(dataSet, target, attempts, "KS", taskName, folder, logsFolder)

        self.commonShapValues = []
        self.commonIndexes = []
        self.pValuesCalculator = PValueCalculator(ShapValuesComplexityCalculatorFactory(), KSMetric(), attempts,  True, False, False)

    def calculateMetric(self, objects, iClass, jClass):
        pValues = self.pValuesCalculator.calcPValueFastPro(objects, self.dataSet, self.target, iClass, jClass)
        return pValues

    def processCalculatedMetric(self, data):
        complexityCalculator = data[2]
        self.commonShapValues.append(complexityCalculator.getShapValues())
        self.commonIndexes.append(complexityCalculator.getObjectsIndex())

    def serializeConcrete(self, array, subname):
        curPair = self.labels[-1]
        currentObjects = self.objectsCount[-1]
        serialize_labeled_list_of_arrays(array, self.labels, f'{self.taskName}_{subname}',
                                         self.attempts, f'{self.logsFolder}\\{subname}_{self.taskName}_{self.attempts}_{curPair}_{currentObjects}.txt')

    def serializeCalculatedData(self):
        self.serializeConcrete(self.commonShapValues, f"{self.name}_frequency")
        self.serializeConcrete([x**2 for x in self.commonShapValues], f"{self.name}_entropy")
        self.serializeConcrete(self.commonIndexes, f"{self.name}_indexes")