from CodeResearch.DataSeparationFramework.Metrics.KSMetric import KSMetric
from CodeResearch.DataSeparationFramework.SimpleDataSeparationCalculator import SimpleDataSeparationCalculator
from CodeResearch.ObjectComplexity.Factory.KSComplexityCalculatorFactory import KSComplexityCalculatorFactory
from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPair
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays
from CodeResearch.DataSeparationFramework.pValueCalculator import PValueCalculator


class KSDataSeparationCalculator(SimpleDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, taskName, folder, logsFolder):
        super().__init__(dataSet, target, attempts, "KS", taskName, folder, logsFolder)
        self.commonOutOfSamplePairs = []
        self.commonEntropies = []
        self.commonFrequences = []
        self.commonErrors = []
        self.commonIndexes = []
        self.pValuesCalculator = PValueCalculator(KSComplexityCalculatorFactory(), KSMetric(), attempts,  True, False, False)

    def calculateMetric(self, objects, iClass, jClass):
        pValues = self.pValuesCalculator.calcPValueFastPro(objects, self.dataSet, self.target, iClass, jClass)
        return pValues

    def processCalculatedMetric(self, data):
        self.commonOutOfSamplePairs.append(data[3])

        complexityCalculator = data[2]
        self.commonEntropies.append(complexityCalculator.calculateComplexity())
        self.commonFrequences.append(complexityCalculator.getObjectsFrequences())
        self.commonErrors.append([complexityCalculator.getErrorExpectation()])
        self.commonIndexes.append(complexityCalculator.getObjectsIndex())

    def serializeConcrete(self, array, subname):
        curPair = self.labels[-1]
        currentObjects = self.objectsCount[-1]
        serialize_labeled_list_of_arrays(array, self.labels, f'{self.taskName}_{subname}',
                                         self.attempts, f'{self.logsFolder}\\{subname}_{self.taskName}_{self.attempts}_{curPair}_{currentObjects}.txt')

    def serializeCalculatedData(self):
        curPair = self.labels[-1]

        visualizeAndSaveKSForEachPair(self.commonOutOfSamplePairs, self.labels, f'{self.taskName}_{self.name}_OOS', self.attempts, curPair, self.folder)

        self.serializeConcrete(self.commonOutOfSamplePairs, f"{self.name}_OOS")
        self.serializeConcrete(self.commonEntropies, f"{self.name}_entropy")
        self.serializeConcrete(self.commonFrequences, f"{self.name}_frequency")
        self.serializeConcrete(self.commonErrors, f"{self.name}_error")
        self.serializeConcrete(self.commonIndexes, f"{self.name}_indexes")
