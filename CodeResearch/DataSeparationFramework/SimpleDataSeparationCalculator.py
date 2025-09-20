from abc import abstractmethod

from CodeResearch.DataSeparationFramework.dataSeparationBaseCalculator import BaseDataSeparationCalculator
from CodeResearch.Visualization.VisualizeAndSaveCommonTopSubsamples import visualizeAndSaveKSForEachPair
from CodeResearch.Visualization.saveDataForVisualization import serialize_labeled_list_of_arrays
from CodeResearch.pValueCalculator import calcPValueFastPro


class SimpleDataSeparationCalculator(BaseDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, name, taskName, folder, logsFolder):

        self.logsFolder = logsFolder
        self.dataSet = dataSet
        self.target = target
        self.attempts = attempts
        self.name = name
        self.taskName = taskName
        self.folder = folder

        self.objects = []
        self.commonPermutationPairs = []
        self.labels = []

    @abstractmethod
    def calculateMetric(self, objects, iClass, jClass):
        pass

    def processCalculatedMetric(self, data):
        pass

    def serializeCalculatedData(self):
        pass

    def calculateDataSeparability(self, objects, iClass, jClass):
        curPair = f'{iClass}_{jClass}'

        pValues2 = self.calculateMetric(objects, iClass, jClass)

        if len(pValues2[0]) > 0:
            self.commonPermutationPairs.append(pValues2[0])
            self.processCalculatedMetric(pValues2)

        self.labels.append(curPair)
        self.objects.append(objects)

    def serializeResults(self):
        curPair = self.labels[-1]
        currentObjects = self.objects[-1]

        if len(self.commonPermutationPairs) > 0:
            visualizeAndSaveKSForEachPair(self.commonPermutationPairs, self.labels,
                                      f'{self.taskName}_{self.name}', self.attempts,
                                      curPair, self.folder)


            serialize_labeled_list_of_arrays(self.commonPermutationPairs, self.labels,
                                         f'{self.taskName}_{self.name}', self.attempts,
                                         f'{self.logsFolder}\\{self.name}_{self.taskName}_{self.attempts}_{curPair}_{currentObjects}.txt')

            self.serializeCalculatedData()

class KSPermutationDataSeparationCalculator(SimpleDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, taskName, folder, logsFolder):
        super().__init__(dataSet, target, attempts, "KS_permutation", taskName, folder, logsFolder)

    def calculateMetric(self, objects, iClass, jClass):
        pValues = calcPValueFastPro(objects, self.dataSet, self.target, iClass, jClass, self.attempts,
                                     True, True, False)
        return pValues

class MLDataSeparationCalculator(SimpleDataSeparationCalculator):
    def __init__(self, dataSet, target, attempts, taskName, folder, logsFolder):
        super().__init__(dataSet, target, attempts, "ML", taskName, folder, logsFolder)

    def calculateMetric(self, objects, iClass, jClass):
        return calcPValueFastPro(objects, self.dataSet, self.target, iClass, jClass, self.attempts, False, False, True)

    def calculateDataSeparability(self, objects, iClass, jClass):
        curPair = f'{iClass}_{jClass}'

        pValues2 = self.calculateMetric(objects, iClass, jClass)

        if len(pValues2[1]) > 0:
            self.commonPermutationPairs.append(pValues2[1])
            self.processCalculatedMetric(pValues2)

        self.labels.append(curPair)
        self.objects.append(objects)

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
        currentObjects = self.objects[-1]
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
