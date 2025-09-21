from CodeResearch.DataSeparationFramework.SimpleDataSeparationCalculator import SimpleDataSeparationCalculator
from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPair
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays
from CodeResearch.pValueCalculator import calcPValueFastPro


class KSDataSeparationCalculator(SimpleDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, taskName, folder, logsFolder):
        super().__init__(dataSet, target, attempts, "KS", taskName, folder, logsFolder)
        self.commonOutOfSamplePairs = []
        self.commonEntropies = []
        self.commonFrequences = []
        self.commonErrors = []
        self.commonIndexes = []

    def calculateMetric(self, objects, iClass, jClass):
        pValues = calcPValueFastPro(objects, self.dataSet, self.target, iClass, jClass, self.attempts,
                                     True, False, False)
        return pValues

    def processCalculatedMetric(self, data):
        pValues1 = data

        self.commonOutOfSamplePairs.append(pValues1[3])
        self.commonEntropies.append(pValues1[2].calculateComplexity())
        self.commonFrequences.append(pValues1[2].getObjectsFrequences())
        self.commonErrors.append([pValues1[2].getErrorExpectation()])
        self.commonIndexes.append(pValues1[2].getObjectsIndex())

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
