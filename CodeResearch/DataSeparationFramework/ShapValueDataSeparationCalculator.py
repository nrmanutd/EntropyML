from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.SimpleDataSeparationCalculator import SimpleDataSeparationCalculator
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator
from CodeResearch.ObjectComplexity.Factory.ShapValuesComplexityCalculatorFactory import \
    ShapValuesComplexityCalculatorFactory
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays

class ShapValueDataSeparationCalculator(SimpleDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, taskName, folder, logsFolder):
        super().__init__(dataSet, target, attempts, "KS", taskName, folder, logsFolder)

        self.commonImportance = []
        self.commonIndexes = []
        self.commonHardness = []
        self.pValuesCalculator = PValueCalculator(ShapValuesComplexityCalculatorFactory(), KSMetric(), attempts,  True, False, False)

    def calculateMetric(self, objects, iClass, jClass):
        pValues = self.pValuesCalculator.calcPValueFastPro(objects, self.dataSet, self.target, iClass, jClass)
        return pValues

    def processCalculatedMetric(self, data):
        complexityCalculator = data[2]
        instanceImportance, instanceHardness = complexityCalculator.getShapValues()
        self.commonImportance.append(instanceImportance)
        self.commonHardness.append(instanceHardness)
        self.commonIndexes.append(complexityCalculator.getObjectsIndex())

    def serializeConcrete(self, array, subname):
        curPair = self.labels[-1]
        currentObjects = self.objectsCount[-1]
        serialize_labeled_list_of_arrays(array, self.labels, f'{self.taskName}_{subname}',
                                         self.attempts, f'{self.logsFolder}\\{subname}_{self.taskName}_{self.attempts}_{curPair}_{currentObjects}.txt')

    def serializeCalculatedData(self):
        self.serializeConcrete(self.commonHardness, f"{self.name}_frequency")
        self.serializeConcrete(self.commonImportance, f"{self.name}_entropy")
        self.serializeConcrete(self.commonIndexes, f"{self.name}_indexes")